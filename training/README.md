# SO-101 training tracks

The launchers in this directory keep large artifacts under configurable local
and HPC storage roots:

- local ACT: `${SO101_STORAGE_ROOT:-$HOME/so101_artifacts}/so101_training`
- RWTH Cosmos: `/hpcwork/$USER/so101_cosmos`
- Hugging Face and Torch caches are redirected to the same locations.

## Track 1: ACT with pretrained vision and static-frame reduction

The recommended launcher always creates a fresh run: it has no resume option
and never loads an older ACT checkpoint. It starts ResNet-18 from ImageNet
weights, leaves ACT's VAE enabled, uses batch size 8, predicts 50-action
chunks, and records an inference horizon of 10 actions in the checkpoint.

Long stationary runs are capped in the training sample index without modifying
or copying the source videos. ACT action targets still come from the original
contiguous 30 Hz timeline. The default 100-episode dataset retains 90 training
episodes after the deterministic validation/test holdout.

```bash
cd /path/to/so101_physical_ai
SO101_STORAGE_ROOT="$HOME/so101_artifacts" \
training/train_act_pretrained_trimmed.sh --dataset /path/to/lerobot_dataset
```

The run, checkpoints, model/cache downloads and trim report are written below
`$SO101_STORAGE_ROOT/so101_training`. Checkpoints are saved every
10,000 steps through step 100,000. Preview the complete command without
starting training:

```bash
training/train_act_pretrained_trimmed.sh \
  --dataset /path/to/lerobot_dataset --dry-run
```

The previous launcher below is retained only for reproducing the original
randomly initialized ResNet experiment.

## Legacy Track 1: ACT fully from scratch on the RTX 4070

The launcher selects the newest `so101_gazebo_randomized_stack_*` dataset, makes a deterministic
40/5/5 episode split for the current 50 episodes, and trains only on the 40 training episodes. It
uses both cameras and the six joint positions. It intentionally excludes
`observation.environment_state`, because that vector contains simulator ground-truth nut positions
that will not exist on the physical robot.

ACT's transformer and ResNet are initialized from scratch (no `policy.path`, and ImageNet backbone
weights are disabled):

```bash
cd /path/to/so101_physical_ai
training/train_act_local.sh --steps 100000 --batch-size 8
```

If 8 GB VRAM runs out, restart with `--batch-size 4` (or 2). A zero-write command preview is:

```bash
training/train_act_local.sh --dry-run
```

Checkpoints stay under the configured storage root. To additionally upload
policy checkpoints using the currently active Hugging Face login:

```bash
training/train_act_local.sh --push-to-hub shubham4413/so101-act-nut-stack
```

The raw dataset is not uploaded by this command.

The launcher keeps Python multiprocessing sockets under `/tmp`, which also
supports exFAT storage roots. If it reports that storage is read-only, repair
or remount that filesystem before training.

## Track 2: Cosmos Predict2.5 action-conditioned LoRA on RWTH H100

Cosmos does not consume LeRobot joint trajectories directly. First convert the dataset locally.
The converter downsamples 30 Hz to 5 Hz, encodes the selected RGB view as 320x256 H.264, and uses
the SO-101 simulation URDF for forward kinematics. Its annotation stores end-effector
`[x,y,z,roll,pitch,yaw]` and gripper closure in Cosmos' expected schema.

```bash
cd /path/to/so101_physical_ai
training/prepare_cosmos_dataset.sh
```

The command prints the new output directory under
`$SO101_STORAGE_ROOT/so101_training/cosmos_data/`. To convert the FPV camera instead:

```bash
CAMERA_KEY=observation.images.fpv training/prepare_cosmos_dataset.sh
```

Run a one-episode conversion test with `training/prepare_cosmos_dataset.sh --max-episodes 1`.

Copy the completed conversion and scripts to RWTH (quotes are important because the local drive
name contains a space):

```bash
LOCAL_DATASET="$SO101_STORAGE_ROOT/so101_training/cosmos_data/so101_stack_YYYYMMDD_HHMMSS" \
  training/sync_cosmos_to_rwth.sh
```

Set up Cosmos once:

```bash
ssh rwth-gpu
bash /hpcwork/$USER/so101_cosmos/bundle/setup_cosmos_rwth.sh
```

The NVIDIA checkpoint is gated. In the browser, accept the license for
`nvidia/Cosmos-Predict2.5-2B`, then authenticate on the cluster. Do not paste a token into any
script or Slurm log:

```bash
export HF_HOME=/hpcwork/$USER/so101_cosmos/cache/huggingface
/hpcwork/$USER/so101_cosmos/src/cosmos-predict2.5/.venv/bin/hf auth login
```

Submit one H100 job:

```bash
sbatch /hpcwork/$USER/so101_cosmos/bundle/train_cosmos_lora_rwth.sbatch
```

Useful overrides:

```bash
MAX_ITER=1000 SAVE_ITER=250 GRAD_ACCUM=8 \
  sbatch --export=ALL /hpcwork/$USER/so101_cosmos/bundle/train_cosmos_lora_rwth.sbatch
```

The job uses the official 2B action-conditioned checkpoint, inserts rank-32 PEFT LoRA adapters
into attention and MLP projections, trains at batch size 1 with gradient accumulation 8, and writes
checkpoints below `/hpcwork/$USER/so101_cosmos/output`. The base checkpoint is resolved to a concrete
local path before training; this also makes it easy to verify in the log that pretrained weights,
not random weights, were loaded.

The current 50-episode dataset is enough for a pipeline/overfit experiment, but it is not enough to
claim a broadly general world-action model. Treat the first 1,000-5,000-step Cosmos run as a
feasibility test and judge it on the five held-out layouts before collecting substantially more
diverse episodes.

At the time these scripts were prepared, RWTH reports partition `c23g` as `down`. Submission may
queue or reject until maintenance ends; that is cluster state, not a problem with the launcher.
