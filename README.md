# so101_physical_ai

Training two policies on SO-101 (6-DoF) teleoperation data collected on the arm, then validating them
before real-robot deployment. All training runs on RWTH HPC (`ssh rwth-gpu`, H100); the local RTX 4070
mobile is for inference/deployment only.

| Model | Dataset | What | Where trained |
|---|---|---|---|
| **Diffusion Policy** | [`shubham4413/so101_vla`](https://huggingface.co/datasets/shubham4413/so101_vla) — 149 ep, 112k frames, short stacking clips | Behavior-cloning stacking policy | 1× H100 |
| **VLA-JEPA** (Qwen3-VL-2B + V-JEPA2 world model + DiT head) | [`shubham4413/so101_wm`](https://huggingface.co/datasets/shubham4413/so101_wm) — 177 ep, 356k frames, longer clips | LoRA fine-tune from `lerobot/VLA-JEPA-Pretrain`, world model **on** | 1× H100 |

Both datasets: 2 cameras (`left`, `fpv`) 640×480 @ 30 fps, 6-DoF state/action, LeRobot **v3.0** format.

## Feasibility & GPU budget

Both are trainable on a **single H100 (80 GB)** in bf16 — **no training-time quantization needed**.
Rough budget (see `env/setup_hpc.md` and the plan for the full breakdown):

| Model | Steps | ~1 run | Budget incl. tuning |
|---|---|---|---|
| Diffusion Policy | 100–150k | ~5–7 h | ~15–20 H100-h |
| VLA-JEPA (LoRA + WM) | 25–40k | ~5–8 h | ~25–40 H100-h |
| eval / smoke / restarts | — | — | ~10 H100-h |
| **Total** | | | **≈ 50–70 H100-hours** |

Quantization only matters for **deployment** on the 4070 (see below), not for training.

## Why LoRA for VLA-JEPA (not `freeze_qwen`)

`configuration_vla_jepa.py` force-disables the world model whenever `freeze_qwen=true` (no gradient reaches
it). To keep the JEPA world-model objective active on the world-model dataset while staying cheap on one
H100, we LoRA-adapt the Qwen backbone and fully train the action head + world-model predictor. SO-101 is
6-DoF but the pretrained checkpoint is 7-DoF, so the action/state projection layers are reinitialized
(`--policy.reinit_modules`) and the gripper index is set to 5 (`--policy.gripper_dim=5`).

## Repo layout

```
env/setup_hpc.md              # one-time HPC env setup (lerobot 0.5.2-dev source, has vla_jepa)
scripts/smoke_test.sbatch     # 10-step end-to-end check of BOTH pipelines — RUN FIRST
slurm/train_diffusion.sbatch  # full Diffusion Policy training job
slurm/train_vla_jepa.sbatch   # full VLA-JEPA LoRA+WM training job
data/make_splits.py           # reproducible train/val episode splits
eval/offline_action_error.py  # offline action-error metrics + trajectory plots (gate before robot)
eval/rollout_real.md          # real-robot rollout + safety checklist
```

## Workflow

```bash
# 0. On HPC: set up the env once
#    (see env/setup_hpc.md — installs lerobot 0.5.2-dev source with the vla_jepa extra)

# 1. Make reproducible splits (writes data/splits/*.json)
python data/make_splits.py --repo-id shubham4413/so101_vla
python data/make_splits.py --repo-id shubham4413/so101_wm

# 2. Smoke test BOTH pipelines (10 steps each) — do NOT skip
sbatch scripts/smoke_test.sbatch      # check logs for exit codes 0 + PEFT trainable-params line

# 3. Full training (one H100 each)
sbatch slurm/train_diffusion.sbatch
sbatch slurm/train_vla_jepa.sbatch

# 4. Offline validation on held-out episodes
python eval/offline_action_error.py --policy-path <ckpt>/pretrained_model \
    --repo-id shubham4413/so101_vla --out-dir eval/out/diffusion
python eval/offline_action_error.py --policy-path <ckpt>/pretrained_model \
    --repo-id shubham4413/so101_wm  --out-dir eval/out/vlajepa

# 5. Real robot — follow eval/rollout_real.md (safety first)
```

To train on only the train split, capture the episode list into the `EPISODES` env var before `sbatch`:
```bash
EPISODES=$(python data/make_splits.py --repo-id shubham4413/so101_vla --print train)
sbatch --export=ALL,EPISODES="$EPISODES" slurm/train_diffusion.sbatch
```

## Fallback / PEFT loading

- **If the VLA-JEPA smoke test fails on the PEFT wrap** (LoRA + custom policy is the one uncertain piece):
  fall back to a **full fine-tune** — delete all `--peft.*` flags in `slurm/train_vla_jepa.sbatch`, keep
  everything else (`reinit_modules`, `gripper_dim=5`, world model stays on by default). Costs ~2–3× the
  hours but is the most reliable path; still fits one H100 at `--batch_size=4`–`8`.
- **Loading a LoRA checkpoint for eval/deploy:** if `make_policy` can't reattach adapters, merge them into
  the base once (`PeftModel.from_pretrained(...).merge_and_unload()`) and save a standalone checkpoint to
  point the eval/rollout tools at.

## Deployment note (RTX 4070 mobile, 8 GB)

- **Diffusion Policy** runs real-time on the 4070 — no special handling.
- **VLA-JEPA** drops the world model at inference (just Qwen-2B + action head), but a 2B VLM on 8 GB is
  tight and real-time control is not guaranteed. Options: int8/4-bit **inference** quantization, or run
  inference on a remote/HPC GPU and stream actions to the arm. This is the only place quantization belongs.

See the full plan for rationale and the verified facts it is built on.

## Gazebo and RViz

The simulation has one standard ROS 2 launch entry point. It starts Gazebo,
`ros2_control`, the joint-state broadcaster, camera bridges, TF publisher and
RViz together:

```bash
./gazebo/scripts/run_sim_rviz.sh
```

Do not also run `run_world.sh` or `run_rviz_cameras.sh` in another terminal.
Two controller/TF publishers on the same ROS domain will produce conflicting
joint transforms. Stop older launch processes with `Ctrl+C` before using the
combined command.

The only source of the initial robot state is
`gazebo/config/initial_pose.yaml`. It contains the world spawn and all six
joint values in metres/radians. The launch generates the Gazebo world, URDF
`ros2_control` initial state and RViz grid position from that file.

For a headless synchronization check:

```bash
./gazebo/scripts/run_sim_rviz.sh launch_rviz:=false gz_extra_args:=-s
./gazebo/scripts/validate_gazebo_tf_sync.py
```
