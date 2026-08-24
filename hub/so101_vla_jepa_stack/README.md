---
library_name: lerobot
pipeline_tag: robotics
license: apache-2.0
base_model: lerobot/VLA-JEPA-Pretrain
datasets:
- shubham4413/so101_wm
tags:
- lerobot
- vla_jepa
- robotics
- peft
- lora
- so101
- world-model
---

# SO-101 VLA-JEPA Stack

`so101_vla_jepa_stack` is a PEFT/LoRA fine-tune of
[`lerobot/VLA-JEPA-Pretrain`](https://huggingface.co/lerobot/VLA-JEPA-Pretrain) for an SO-101
robot arm. It was trained on
[`shubham4413/so101_wm`](https://huggingface.co/datasets/shubham4413/so101_wm), which contains
teleoperated stacking and unstacking demonstrations with two synchronized RGB camera views.

VLA-JEPA combines a Qwen3-VL vision-language backbone, a frozen V-JEPA2 encoder, an
action-conditioned JEPA video predictor, and a flow-matching DiT action head.

## Status

- Training completed successfully: **30,000 / 30,000 steps**
- Final logged training loss: **0.134**
- Held-out offline evaluation: **not yet reported**
- Real-robot success rate: **not yet reported**

The loss result demonstrates stable optimization of the training objective. It must not be interpreted
as a physical task-success percentage.

## Intended use

This checkpoint is intended for research and controlled evaluation of stacking/unstacking policies on
the same 6-DoF SO-101 setup, camera arrangement, objects, and workspace represented in the training
dataset.

The model is not intended for unsupervised operation around people, fragile objects, or safety-critical
equipment. A human operator should remain at the emergency stop during every initial rollout.

## Model inputs and outputs

### Inputs

| Feature | Type | Shape | Deployment source |
|---|---|---:|---|
| `observation.images.exterior_1_left` | RGB image | `(3, 224, 224)` | Dataset/robot camera `left` |
| `observation.images.exterior_2_left` | RGB image | `(3, 224, 224)` | Dataset/robot camera `fpv` |
| Task instruction | Text | — | Stack or unstack instruction |

The deployment pipeline must apply this exact mapping:

```json
{
  "observation.images.left": "observation.images.exterior_1_left",
  "observation.images.fpv": "observation.images.exterior_2_left"
}
```

### Output

| Feature | Type | Shape |
|---|---|---:|
| `action` | SO-101 joint-position action | `(6,)` |

Joint order:

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
```

The action chunk size is 7 and the gripper index is 5.

## What this repository contains

This is a PEFT checkpoint, not a standalone copy of all 3.1B base-model parameters.

`adapter_model.safetensors` contains:

- rank-16 LoRA adapters for Qwen attention projections:
  `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- rank-16 LoRA adapters for Qwen MLP projections:
  `gate_proj`, `up_proj`, and `down_proj`;
- the complete fine-tuned `model.action_model`;
- the complete fine-tuned `model.video_predictor`.

The original Qwen weights and V-JEPA2 encoder remain in the base model and are not duplicated here.
LeRobot/PEFT loads `lerobot/VLA-JEPA-Pretrain` and applies the contents of this repository.

The small preprocessor and postprocessor safetensor files contain the normalization and
unnormalization statistics required for correct robot actions.

## Fine-tuning details

### Training data

| Property | Value |
|---|---:|
| Dataset | `shubham4413/so101_wm` |
| Total episodes | 177 |
| Training episodes | 151 |
| Held-out episodes | 26 |
| Training frames | 302,957 |
| Training samples consumed | 240,000 |
| Approximate passes over training frames | 0.79 |
| Cameras | `left`, `fpv` |
| Source resolution/rate | 640×480 at 30 FPS |
| Tasks | Stack and unstack large 3D-printed nuts |

The episode split used seed 1000. Episodes were split at episode level, not frame level.

### Trainable components

| Component | Training mode |
|---|---|
| Qwen3-VL backbone | Frozen base weights with LoRA adapters |
| DiT action model | Fully trained |
| Action/state projections | Reinitialized for 6-DoF and fully trained |
| JEPA video predictor | Fully trained |
| V-JEPA2 encoder | Frozen |

Four tensors from the 7-DoF pretrained action/state interface were intentionally reinitialized for the
6-DoF SO-101:

```text
model.action_model.action_encoder.layer1.weight
model.action_model.action_decoder.layer2.weight
model.action_model.action_decoder.layer2.bias
model.action_model.state_encoder.layer1.weight
```

### Parameter counts

| Parameters | Count |
|---|---:|
| Total | 3,104,588,172 |
| Learnable | 334,258,694 |
| Learnable fraction | 10.77% |

### Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Steps | 30,000 |
| Batch size | 8 |
| Optimizer | AdamW |
| Peak learning rate | `1e-4` |
| Warm-up | 5,000 steps |
| Schedule | Cosine decay |
| Final learning rate | `1e-6` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| World-model loss weight | 0.1 |
| Gradient clipping | 1.0 |
| Training dtype | bfloat16 |

### Compute

- One NVIDIA H100-class Hopper GPU on RWTH HPC
- Slurm job `2274795`
- 2026-07-25 10:51:08 to 2026-07-26 03:24:21
- Runtime: **16 h 33 min 13 s**
- Typical GPU memory usage: approximately **35.6 GB**
- Typical throughput: approximately **4 samples/s**

## Training results

| Step | Samples | Training loss | Gradient norm |
|---:|---:|---:|---:|
| 100 | 800 | 1.410 | 0.894 |
| 1,000 | 8,000 | 0.195 | 2.303 |
| 5,000 | 40,000 | 0.158 | 0.519 |
| 10,000 | 80,000 | 0.146 | 0.316 |
| 15,000 | 120,000 | 0.142 | 0.226 |
| 20,000 | 160,000 | 0.139 | 0.204 |
| 25,000 | 200,000 | 0.135 | 0.149 |
| 30,000 | 240,000 | **0.134** | **0.139** |

The logged loss decreased by approximately 90.5% from step 100 to step 30,000 and plateaued around
`0.133–0.135`. The successful run contained no NaNs, CUDA out-of-memory events, or fatal CUDA errors.

No validation loss, held-out action-error metric, or physical success rate is claimed because those
measurements have not yet been completed.

## Installation

Use the LeRobot revision that produced this checkpoint:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 3dd19d04

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[vla_jepa,core_scripts,feetech,peft]"
```

Download the checkpoint before powering the robot:

```bash
hf download shubham4413/so101_vla_jepa_stack \
  --local-dir "$HOME/models/so101_vla_jepa_stack"
```

## Real-robot rollout

Replace every `<...>` placeholder with values verified on the deployment computer. Do not guess the
arm port, calibration ID, camera identities, or safe positional-change limit.

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path="$HOME/models/so101_vla_jepa_stack" \
  --device=cuda \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<FOLLOWER_CALIBRATION_ID> \
  --robot.max_relative_target=<VALIDATED_CONSERVATIVE_POSITION_LIMIT> \
  --robot.cameras='{left: {type: opencv, index_or_path: <LEFT_CAMERA>, width: 640, height: 480, fps: 30}, fpv: {type: opencv, index_or_path: <FPV_CAMERA>, width: 640, height: 480, fps: 30}}' \
  --rename_map='{"observation.images.left":"observation.images.exterior_1_left","observation.images.fpv":"observation.images.exterior_2_left"}' \
  --task="<use the exact stack or unstack instruction represented in the dataset>" \
  --fps=30 \
  --duration=10 \
  --return_to_initial_position=true \
  --display_data=true
```

For initial deployment:

1. Run held-out offline action evaluation first.
2. Confirm that `left` and `fpv` are correctly assigned and reproduce the training views.
3. Start with an empty, padded workspace and a central arm pose.
4. Keep a human operator at the emergency stop.
5. Abort on jerky motion, joint-limit seeking, incorrect gripper direction, or increasing latency.
6. Measure success over 20–30 controlled trials before routine use.

## Limitations

- No held-out action MSE/MAE has been reported yet.
- No real-robot success rate has been reported yet.
- Data comes from one robot, workspace, lighting setup, and operator.
- The dataset contains successful demonstrations but no recovery/failure episodes.
- There are no force, torque, or depth observations.
- Camera mounting or key mismatches can cause immediate distribution shift.
- A 2B VLM may not sustain the desired control rate on an 8 GB mobile GPU.
- The checkpoint requires its base model; the 1.3 GB adapter is not standalone.

## License

Apache-2.0, following the upstream `lerobot/VLA-JEPA-Pretrain` model.

The training dataset is released separately under the MIT license.

## References

- [VLA-JEPA paper](https://arxiv.org/abs/2602.10098)
- [VLA-JEPA base checkpoint](https://huggingface.co/lerobot/VLA-JEPA-Pretrain)
- [Training dataset](https://huggingface.co/datasets/shubham4413/so101_wm)
- [LeRobot](https://github.com/huggingface/lerobot)
