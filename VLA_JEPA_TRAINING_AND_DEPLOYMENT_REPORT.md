# VLA-JEPA fine-tuning and real-robot deployment report

Report prepared: **2026-07-30**

## 1. Executive summary

The SO-101 VLA-JEPA fine-tuning run **completed successfully** on the RWTH HPC system.

- Dataset: `shubham4413/so101_wm`
- Starting model: `lerobot/VLA-JEPA-Pretrain`
- Successful Slurm job: `2274795`
- Hardware: 1 NVIDIA H100-class Hopper GPU
- Training window: **2026-07-25 10:51:08–2026-07-26 03:24:21**
- Runtime: **16 h 33 min 13 s**
- Result: `COMPLETED`, exit code `0:0`
- Completed steps: **30,000 / 30,000**
- Final logged training loss: **0.134**
- Final checkpoint:

  ```text
  /hpcwork/dl125352/train/so101_wm_vlajepa/checkpoints/last/pretrained_model
  ```

  `last` points to checkpoint `030000`.

The optimization was stable and the training objective was learned. However, this result does **not**
yet prove that the policy performs the task reliably on unseen episodes or on the physical robot.
Validation and real-robot success-rate measurements have not yet been run.

## 2. What was done and when

The following timeline comes from RWTH Slurm accounting and the saved training logs.

| Date and time | Job | Result | Meaning |
|---|---:|---|---|
| 2026-07-24 01:04 | `2195643` | Cancelled before starting | Initial VLA-JEPA smoke-test submission |
| 2026-07-24 02:18–02:19 | `2195706` | Failed after 1m 05s | Smoke-test iteration |
| 2026-07-25 09:50–10:03 | `2270917`, `2271029`, `2271183`, `2271415` | Failed | Additional short smoke-test/debug iterations |
| 2026-07-25 10:08:32–10:10:19 | `2271897` | **Completed** | VLA-JEPA smoke test passed, exit code `0:0` |
| 2026-07-25 10:12–10:49 | `2272178`, `2272907`, `2273737`, `2274588` | Failed during startup | Full-run launch/debug iterations; none trained for more than 2m 34s |
| 2026-07-25 10:51:08 | `2274795` | Started | Successful full VLA-JEPA fine-tuning run |
| 2026-07-25 16:28:52 | `2274795` | Checkpoint | Step 10,000 checkpoint saved |
| 2026-07-25 21:56:38 | `2274795` | Checkpoint | Step 20,000 checkpoint saved |
| 2026-07-26 03:24:10 | `2274795` | Checkpoint | Step 30,000 checkpoint saved |
| 2026-07-26 03:24:14 | `2274795` | Training ended | LeRobot logged `End of training` |
| 2026-07-26 03:24:21 | `2274795` | **Completed** | Slurm exit code `0:0` |

The short failed jobs are recorded for reproducibility. They do not affect the weights from successful
job `2274795`.

## 3. Dataset and split

| Property | Value |
|---|---:|
| Dataset | `shubham4413/so101_wm` |
| Cached dataset revision | `96e2ecc061a02d2952083128350f6815de30cb9f` |
| Total episodes | 177 |
| Training episodes | **151** |
| Held-out validation episodes | **26** |
| Training frames reported by LeRobot | **302,957** |
| Cameras | `left`, `fpv` |
| Camera recording format | 640×480 at 30 FPS |
| Action dimension | 6 |
| State dimension | 6 |
| Split seed | 1000 |
| Number of task labels | 2 |

The exact episode IDs are stored in:

```text
data/splits/shubham4413__so101_wm.json
```

The run consumed 240,000 training samples (`30,000 steps × batch size 8`), equivalent to approximately
**0.79 passes** over the 302,957-frame training split.

## 4. Model configuration

The fine-tune started from `lerobot/VLA-JEPA-Pretrain` and used:

- Qwen backbone: `Qwen/Qwen3-VL-2B-Instruct`
- JEPA visual encoder: `facebook/vjepa2-vitl-fpc64-256`
- Action model: DiT-B, 16 layers
- Action chunk size: 7
- World-model video frames: 8
- Inference diffusion steps: 4
- Training dtype: bfloat16
- Batch size: 8
- Optimizer: AdamW
- Peak learning rate: `1e-4`
- Warm-up: 5,000 steps
- Cosine decay: 30,000 steps
- Final scheduled learning rate: `1e-6`
- World-model loss weight: 0.1
- Gradient clip norm: 1.0

Two physical camera keys were mapped into the names expected by the pretrained model:

```json
{
  "observation.images.left": "observation.images.exterior_1_left",
  "observation.images.fpv": "observation.images.exterior_2_left"
}
```

This mapping is part of the model interface and **must also be used during evaluation and deployment**.

## 5. Exactly which parts were fine-tuned

The run used PEFT LoRA rather than a full 3.1B-parameter fine-tune.

### 5.1 Qwen vision-language backbone

The original Qwen weights were frozen except for LoRA adapters attached to these linear projections
throughout `model.qwen`:

- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

LoRA configuration:

| Parameter | Value |
|---|---:|
| Rank `r` | 16 |
| Alpha | 32 |
| Target scope | `model.qwen` projection layers listed above |

### 5.2 Action model

`model.action_model` was trained fully and stored as a PEFT `modules_to_save` component. This includes
the DiT action head and the SO-101-specific action/state projection layers.

The pretrained checkpoint used different action/state dimensions. Four incompatible tensors were
therefore intentionally skipped and randomly reinitialized for the 6-DoF SO-101:

- `model.action_model.action_encoder.layer1.weight`
- `model.action_model.action_decoder.layer2.weight`
- `model.action_model.action_decoder.layer2.bias`
- `model.action_model.state_encoder.layer1.weight`

These reinitialized tensors were then trained as part of the full action model.

### 5.3 JEPA world model

- `model.video_predictor` was trained fully and saved.
- The world-model objective was enabled: `enable_world_model=true`.
- Its weighted contribution to total loss was `0.1 × wm_loss`.
- The pretrained V-JEPA2 encoder itself was not listed as a LoRA target or full-training module and
  therefore remained frozen under PEFT.

### 5.4 Parameter counts

| Parameter group | Count |
|---|---:|
| Total model parameters | **3,104,588,172** |
| Learnable parameters | **334,258,694** |
| Learnable fraction | **10.77%** |

In short: the run trained **Qwen LoRA adapters + the complete action model + the complete video
predictor**, while keeping the remaining pretrained Qwen weights and V-JEPA2 encoder frozen.

## 6. Numerical training results

| Step | Samples | Training loss | Gradient norm |
|---:|---:|---:|---:|
| 100 | 800 | 1.410 | 0.894 |
| 500 | 4,000 | 0.487 | 3.751 |
| 1,000 | 8,000 | 0.195 | 2.303 |
| 5,000 | 40,000 | 0.158 | 0.519 |
| 10,000 | 80,000 | 0.146 | 0.316 |
| 15,000 | 120,000 | 0.142 | 0.226 |
| 20,000 | 160,000 | 0.139 | 0.204 |
| 25,000 | 200,000 | 0.135 | 0.149 |
| 30,000 | 240,000 | **0.134** | **0.139** |

Across all 300 logged measurements:

| Statistic | Loss |
|---|---:|
| First logged loss | 1.410 |
| Final logged loss | 0.134 |
| Minimum logged loss | 0.133 |
| Maximum logged loss | 1.410 |
| Overall mean | 0.1621 |
| Mean, steps 25,100–30,000 | 0.1343 |

The first-to-final reduction was approximately **90.5%**. After the initial adaptation of the randomly
reinitialized SO-101 projection layers, the loss declined smoothly and plateaued around `0.133–0.135`.
No NaNs, CUDA out-of-memory failures, tracebacks, or fatal CUDA errors occurred in the successful run.
GPU memory usage was stable at approximately **35.6 GB**, and throughput was normally **4 samples/s**.

### Interpretation

This is a **successful and numerically stable optimization run**. It demonstrates that the selected
modules learned the combined action/world-model training objective.

It does not yet provide a measured task success rate because:

- `eval_steps=0`
- `dataset.eval_split=0.0`
- W&B logging was disabled
- no held-out offline action-error report exists
- no real-robot rollout report exists

Consequently, the current measured result is **training loss 0.134**, not a robot success percentage.

## 7. Saved artifacts

Three complete checkpoints are present:

| Checkpoint | Approximate size |
|---|---:|
| `010000` | 3.8 GB |
| `020000` | 3.8 GB |
| `030000` | 3.8 GB |
| Total | 12 GB |

Each checkpoint contains a roughly 1.3 GB deployable `pretrained_model` directory and approximately
2.5 GB of optimizer/RNG/scheduler state for resuming training.

The final deployable directory includes:

- `adapter_model.safetensors`
- `adapter_config.json`
- `config.json`
- `train_config.json`
- policy preprocessor and normalizer artifacts
- policy postprocessor and unnormalizer artifacts

The run used `push_to_hub=false`, so successful job `2274795` did **not** upload the checkpoint to
Hugging Face automatically.

## 8. Deployment prerequisites

Do not connect the policy to the robot until the offline and hardware checks below pass.

### 8.1 Use the matching LeRobot revision

The verified project revision is:

```text
v0.5.1-151-g3dd19d04
```

On the deployment computer:

```bash
cd /home/shubhamnagar/lerobot/lerobot
git checkout 3dd19d04

python -m venv .venv-deploy
source .venv-deploy/bin/activate
pip install --upgrade pip
pip install -e ".[vla_jepa,core_scripts,feetech,peft]"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use the same source revision because it contains both the VLA-JEPA policy and the `lerobot-rollout`
real-robot deployment interface used below.

### 8.2 Transfer the final checkpoint from RWTH

Run on the deployment computer:

```bash
mkdir -p "$HOME/models/so101_wm_vlajepa_030000"

rsync -av --progress \
  rwth-gpu-sj:/hpcwork/dl125352/train/so101_wm_vlajepa/checkpoints/030000/pretrained_model/ \
  "$HOME/models/so101_wm_vlajepa_030000/"
```

After transfer, confirm that `adapter_model.safetensors` is approximately 1.3 GB:

```bash
ls -lh "$HOME/models/so101_wm_vlajepa_030000"
```

Do not copy only `adapter_model.safetensors`; deployment also needs the JSON configuration and saved
pre/postprocessor normalization files.

### 8.3 Identify hardware without guessing

The arm USB port, calibration ID, and camera device indices are not stored in this repository.
Discover or reuse the exact values from data collection:

```bash
lerobot-find-port
lerobot-find-cameras opencv
```

Use stable camera paths such as `/dev/v4l/by-id/...` when available. Linux numeric camera indices can
change after reconnecting devices.

The required camera names are exactly:

- `left`: the fixed left/exterior camera used in training
- `fpv`: the wrist/first-person camera used in training

Both must produce 640×480 images at 30 FPS with the same mounting, orientation, framing, and lighting
used for `so101_wm`.

### 8.4 Verify calibration

Reuse the same follower calibration ID used during data collection. Recalibrate if the arm was rebuilt,
re-homed, or its calibration file is missing:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<FOLLOWER_CALIBRATION_ID>
```

Before autonomous deployment, verify the arm and camera configuration in a normal teleoperation session.

## 9. Required validation gate before powering the policy

Run held-out evaluation on the 26 validation episodes before the first autonomous robot trial:

First copy the updated evaluator, which supports the required camera rename map, from this workspace:

```bash
cd /home/shubhamnagar/coding/so101_physical_ai
rsync -av eval/offline_action_error.py \
  rwth-gpu-sj:/home/dl125352/so101_physical_ai/eval/offline_action_error.py
```

Then run the evaluation on RWTH:

```bash
ssh rwth-gpu-sj
cd /home/dl125352/so101_physical_ai
source /home/dl125352/lerobot/.venv/bin/activate

python eval/offline_action_error.py \
  --policy-path /hpcwork/dl125352/train/so101_wm_vlajepa/checkpoints/030000/pretrained_model \
  --repo-id shubham4413/so101_wm \
  --dataset-root /hpcwork/dl125352/hf/hub/datasets--shubham4413--so101_wm/snapshots/96e2ecc061a02d2952083128350f6815de30cb9f \
  --out-dir eval/out/vlajepa_030000_full \
  --device cuda \
  --video-backend pyav \
  --tolerance-s 1e-3 \
  --rename-map '{"observation.images.left":"observation.images.exterior_1_left","observation.images.fpv":"observation.images.exterior_2_left"}'
```

The `--rename-map` argument applies the same `left → exterior_1_left` and
`fpv → exterior_2_left` key mapping used in training.
The validation report must include:

- overall action MSE and MAE
- per-joint MSE and MAE
- worst joint
- binary gripper accuracy
- predicted-versus-ground-truth plots for all 26 held-out episodes

Do not infer deployment quality from training loss alone.

### Held-out validation result (2026-08-01)

Slurm job `2609043` completed successfully on one H100 with exit code `0:0`. It evaluated all
**26 validation episodes / 52,927 frames** in **30 minutes 13 seconds** using the saved policy
preprocessor and postprocessor, including the training-time camera rename map and action
denormalization.

| Metric | Result |
|---|---:|
| Overall MSE | **102.7091** |
| Overall RMSE | **10.1346** |
| Overall MAE | **4.6983** |
| Gripper binary agreement | **81.81%** |
| Gripper majority-class baseline | **52.19%** |
| Worst joint by MSE | **gripper.pos** |

| Joint | MSE | RMSE | MAE | MAE / observed range |
|---|---:|---:|---:|---:|
| shoulder_pan.pos | 23.6261 | 4.8607 | 2.7965 | 1.24% |
| shoulder_lift.pos | 71.5553 | 8.4590 | 5.1078 | 2.33% |
| elbow_flex.pos | 93.1362 | 9.6507 | 5.7562 | 2.95% |
| wrist_flex.pos | 78.9806 | 8.8871 | 4.7464 | 2.23% |
| wrist_roll.pos | 156.8537 | 12.5241 | 4.7035 | 1.31% |
| gripper.pos | 192.0884 | 13.8596 | 5.0797 | 5.08% |

The local artifacts are under `eval/out/vlajepa_030000_full/`: `report.json`, the Slurm logs, and
26 compressed predicted-versus-ground-truth trajectory arrays. This is teacher-forced offline action
imitation, not closed-loop task success. It supports proceeding to a guarded shadow-mode test, but it
does not establish that autonomous real-robot execution is safe or successful.

## 10. First real-robot deployment command

Replace every `<...>` placeholder with the values verified in Section 8. Run the first trial with the
robot unloaded or with soft test objects, a clear padded workspace, and a person holding the e-stop.

```bash
source /home/shubhamnagar/lerobot/lerobot/.venv-deploy/bin/activate

lerobot-rollout \
  --strategy.type=base \
  --policy.path="$HOME/models/so101_wm_vlajepa_030000" \
  --device=cuda \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<FOLLOWER_CALIBRATION_ID> \
  --robot.max_relative_target=<VALIDATED_CONSERVATIVE_POSITION_LIMIT> \
  --robot.cameras='{left: {type: opencv, index_or_path: <LEFT_CAMERA>, width: 640, height: 480, fps: 30}, fpv: {type: opencv, index_or_path: <FPV_CAMERA>, width: 640, height: 480, fps: 30}}' \
  --rename_map='{"observation.images.left":"observation.images.exterior_1_left","observation.images.fpv":"observation.images.exterior_2_left"}' \
  --task="Grab the red hexagon on the right and place it on the red hexagon on the left." \
  --fps=30 \
  --duration=10 \
  --return_to_initial_position=true \
  --display_data=true
```

Do not invent a `max_relative_target` value. Use a conservative positional-change limit already tested
with this arm and its degree/radian configuration.

The rollout implementation loads the PEFT policy before connecting to the physical robot. If adapter
loading fails, it should fail before hardware connection. Do not work around a loading failure while the
arm is powered; first verify the checkpoint, base-model access, `peft` installation, and LeRobot revision.

## 11. Safe staged rollout protocol

1. **Visual-only check:** confirm the displayed `left` and `fpv` streams are not swapped, rotated, stale,
   or mirrored.
2. **Inference timing check:** measure action latency. The model predicts seven-action chunks, but the
   RTX 4070 mobile still needs to keep up with the control loop. Abort if observations/actions queue up.
3. **No-object motion check:** use a 5–10 second rollout from a central pose. Stop on jerky,
   limit-seeking, or task-irrelevant motion.
4. **Soft-object trial:** introduce the hexagons with reduced motion limits and direct supervision.
5. **Repeatable evaluation:** run 20–30 trials over a fixed set of starting positions.
6. **Promotion:** use the model routinely only after it achieves an acceptable success rate with zero
   unsafe events.

For each real trial record:

| Field | Allowed values/example |
|---|---|
| Success | 0 or 1 |
| Failure stage | approach, grasp, transport, place |
| Unsafe event | none, collision, limit hit, e-stop |
| Inference latency | milliseconds per generated action chunk |
| Notes | camera mismatch, hesitation, gripper timing, drift |

Report:

```text
success_rate = successful_trials / total_trials
```

Training is successful, but deployment is successful only after this physical success rate is measured.

## 12. Deployment limitations and recommended next actions

1. **Completed:** offline validation on all 26 held-out episodes (Slurm job `2609043`).
2. Compare checkpoints 10k, 20k, and 30k; the lowest training loss does not guarantee the best
   held-out action accuracy.
3. Transfer the selected checkpoint to the robot computer.
4. Measure unquantized inference latency on the RTX 4070 mobile before moving the arm.
5. If the policy cannot sustain the required control rate, test a validated inference-only
   quantization path or a low-latency remote GPU architecture. Do not quantize training artifacts
   in place.
6. Run 20–30 controlled real-robot trials and add the measured success rate to this report.
7. The complete deployable `pretrained_model` directory was uploaded to the public model repository
   `shubham4413/so101_vla_jepa_stack` in commit
   `c54dedd5ed2d02c3a65fb5cdfccf282bdb86130c`.

## 13. Final status

| Stage | Status |
|---|---|
| Dataset preparation | Complete |
| Reproducible train/validation split | Complete |
| VLA-JEPA smoke test | Complete |
| 30,000-step VLA-JEPA fine-tune | **Complete** |
| Final checkpoint saved | **Complete** |
| Uploaded to Hugging Face | **Complete** — `shubham4413/so101_vla_jepa_stack` |
| Held-out offline validation | **Complete** — 26 episodes, 52,927 frames, MAE 4.6983 |
| Real-robot rollout | Not yet measured |
| Real-robot success rate | Not yet available |
