# Real-robot validation & staged deployment (SO-101)

Only run this **after** offline validation (`eval/offline_action_error.py`) looks sane: low per-joint
action MSE, trajectory plots track ground truth without large phase lag, and gripper accuracy is high.
Closed-loop success can only be measured on the real arm — there is no sim for this custom stacking task.

The checkpoint is trained on HPC; copy the `pretrained_model` dir (or `hf download` your pushed
`--policy.repo_id`) to the machine physically wired to the arm.

## 0. Task recap

"Grab the red hexagon on the right and place it on the red hexagon on the left." 6-DoF SO-101
(`so_follower`), 2 cameras: `left` + `fpv`, both 640×480 @ 30 fps. The deployment cameras and mounting
**must match** how the dataset was recorded (same views, framing, lighting) or the policy will not transfer.

## 1. Safety first (every session)

- [ ] Hand on the e-stop / power cut for the entire rollout.
- [ ] Reduce max joint velocity and torque limits in the robot config for the first sessions.
- [ ] Clear, padded workspace; nothing fragile in reach; start the arm well inside joint limits.
- [ ] Cameras plugged in, correct `left`/`fpv` assignment (swapped views = garbage policy input).
- [ ] Calibrate: `lerobot-calibrate` if the arm was re-assembled or motors re-homed.

## 2. Sanity rollout (1 episode, slow)

Run a single supervised episode at reduced speed and watch that commanded actions are smooth and roughly
task-directed before trusting it. Use lerobot's record/eval tooling to drive the policy on the real robot
(the exact `lerobot-record`/eval-on-robot invocation depends on your robot config; reuse the same
`robot.type`/port/camera setup used to collect the datasets):

- Confirm the policy loads (for a **LoRA VLA-JEPA** checkpoint, adapters must be merged/loaded — see
  README "Fallback / PEFT loading").
- Confirm control rate is acceptable: Diffusion Policy easily runs real-time; **VLA-JEPA (2B) may be slow
  on the RTX 4070 mobile** — see the deployment note in the README. If inference lags, run the policy on a
  remote GPU and stream actions, or quantize for inference.
- Abort immediately on any jerky / limit-seeking motion.

For the HPC HTTP deployment, do not use a generic LeRobot rollout command. Follow
`deployment/HTTP_INFERENCE.md` and begin with the repository's fail-closed client:

```bash
"$HOME/lerobot/.venv/bin/python" deployment/so101_http_slow_rollout.py \
  --cycles 1 --arm SAFE_SLOW_ROLLOUT
```

This executes only the first action of a validated chunk over a minimum-jerk trajectory, then observes
again. The motor loop is local; HPC/SSH latency never directly clocks the motor bus.

## 3. Success-rate evaluation (per model)

Run N = 20–30 trials with varied hexagon start positions covering the workspace. For each trial record:

| field | notes |
|---|---|
| success (0/1) | red hexagon placed stacked on the target |
| failure stage | `approach` / `grasp` / `transport` / `place` |
| unsafe event | any limit hit, collision, or e-stop |
| notes | drift, hesitation, gripper mistiming |

Report **success rate** and the failure-stage breakdown. A confusion between grasp vs place failures tells
you what to fix (more grasp demos, gripper threshold tuning, longer chunk, etc.).

## 4. Compare Diffusion Policy vs VLA-JEPA

Run both models on the **same set** of start positions. Keep the checkpoint that wins on success rate and
motion smoothness. Expectations:
- **Diffusion Policy** — strong, fast, real-time baseline on the single stacking task (trained on `so101_vla`).
- **VLA-JEPA** — language-conditioned, pretrained-backbone policy (trained on `so101_wm`); its edge is
  generalization / instruction-following, at higher inference cost.

## 5. Promote

Only after a stable success rate across a full N-trial block do you promote a checkpoint to routine use.
Record the checkpoint hash, split file, and success numbers alongside the model for reproducibility.

## 6. If it fails to transfer

Common, in rough priority order:
1. **Camera mismatch** — views/framing/lighting differ from training. Fix the rig first; it dominates.
2. **Gripper dim/threshold** — confirm SO-101 gripper is index 5 and `gripper_threshold` suits your gripper.
3. **Normalization** — ensure eval uses the training dataset stats (handled automatically when loading via
   the saved checkpoint config).
4. **Not enough / not diverse data** — collect more demos around the failure stage and re-finetune.
