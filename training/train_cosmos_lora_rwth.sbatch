#!/usr/bin/env bash
#SBATCH --job-name=so101_cosmos_lora
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=/hpcwork/%u/so101_cosmos/logs/%x_%j.out
#SBATCH --error=/hpcwork/%u/so101_cosmos/logs/%x_%j.err

set -euo pipefail

HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_cosmos"}
COSMOS_DIR=${COSMOS_DIR:-"$HPC_ROOT/src/cosmos-predict2.5"}
DATASET_ROOT=${DATASET_ROOT:-"$HPC_ROOT/data/so101_stack"}
MAX_ITER=${MAX_ITER:-5000}
SAVE_ITER=${SAVE_ITER:-500}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LORA_RANK=${LORA_RANK:-32}

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0
source "$COSMOS_DIR/.venv/bin/activate"

export HF_HOME="$HPC_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$HPC_ROOT/cache/xdg"
export UV_CACHE_DIR="$HPC_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$HPC_ROOT/tools/uv-python"
export UV_TOOL_DIR="$HPC_ROOT/tools/uv-tools"
export TORCH_HOME="$HPC_ROOT/cache/torch"
export TMPDIR="$HPC_ROOT/tmp"
export IMAGINAIRE_OUTPUT_ROOT="$HPC_ROOT/output"
export WANDB_MODE=${WANDB_MODE:-disabled}
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" \
  "$IMAGINAIRE_OUTPUT_ROOT" "$HPC_ROOT/logs"

for split in train val test; do
  if [[ ! -d "$DATASET_ROOT/annotation/$split" ]]; then
    echo "Missing Cosmos annotation split: $DATASET_ROOT/annotation/$split" >&2
    exit 1
  fi
done

cd "$COSMOS_DIR"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

# Resolve through Cosmos' official registry. This downloads the gated 2B action-conditioned
# checkpoint into HF_HOME, then passes its concrete local path to avoid ambiguous checkpoint-DB
# loading during LoRA initialization.
if [[ -z ${ACTION_CKPT:-} ]]; then
  ACTION_CKPT=$(python -c 'from cosmos_predict2.config import MODEL_CHECKPOINTS,ModelKey,ModelVariant; print(MODEL_CHECKPOINTS[ModelKey(variant=ModelVariant.ROBOT_ACTION_COND)].download())' | tail -1)
fi
if [[ ! -e "$ACTION_CKPT" ]]; then
  echo "Action-conditioned checkpoint did not resolve: $ACTION_CKPT" >&2
  echo "Authenticate with HF_HOME=$HF_HOME and accept the NVIDIA model license first." >&2
  exit 1
fi
echo "Base checkpoint: $ACTION_CKPT"
echo "Dataset:        $DATASET_ROOT"
echo "Output root:    $IMAGINAIRE_OUTPUT_ROOT"

torchrun --nproc_per_node=1 --master_port=$((12000 + SLURM_JOB_ID % 20000)) -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_reason_embeddings_rectified_flow_2b_256_320 \
  '~dataloader_train.dataloaders' \
  job.name="so101_action_lora_${SLURM_JOB_ID}" \
  checkpoint.load_path="$ACTION_CKPT" \
  checkpoint.save_iter="$SAVE_ITER" \
  checkpoint.load_training_state=false \
  checkpoint.strict_resume=false \
  trainer.max_iter="$MAX_ITER" \
  trainer.grad_accum_iter="$GRAD_ACCUM" \
  trainer.logging_iter=10 \
  trainer.run_validation=true \
  trainer.validation_iter=250 \
  optimizer.lr=0.0001 \
  optimizer.weight_decay=0.001 \
  model.config.use_lora=true \
  model.config.lora_rank="$LORA_RANK" \
  model.config.lora_alpha="$LORA_RANK" \
  model.config.lora_target_modules='q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2' \
  model.config.init_lora_weights=true \
  dataloader_train.batch_size=1 \
  dataloader_train.dataset.train_annotation_path="$DATASET_ROOT/annotation/train" \
  dataloader_train.dataset.val_annotation_path="$DATASET_ROOT/annotation/val" \
  dataloader_train.dataset.test_annotation_path="$DATASET_ROOT/annotation/test" \
  dataloader_train.dataset.video_path="$DATASET_ROOT" \
  dataloader_train.dataset.video_size='[256,320]' \
  dataloader_train.sampler.dataset.train_annotation_path="$DATASET_ROOT/annotation/train" \
  dataloader_train.sampler.dataset.val_annotation_path="$DATASET_ROOT/annotation/val" \
  dataloader_train.sampler.dataset.test_annotation_path="$DATASET_ROOT/annotation/test" \
  dataloader_train.sampler.dataset.video_path="$DATASET_ROOT" \
  dataloader_train.sampler.dataset.video_size='[256,320]' \
  dataloader_val.dataset.train_annotation_path="$DATASET_ROOT/annotation/train" \
  dataloader_val.dataset.val_annotation_path="$DATASET_ROOT/annotation/val" \
  dataloader_val.dataset.test_annotation_path="$DATASET_ROOT/annotation/test" \
  dataloader_val.dataset.video_path="$DATASET_ROOT" \
  dataloader_val.dataset.video_size='[256,320]' \
  dataloader_val.sampler.dataset.train_annotation_path="$DATASET_ROOT/annotation/train" \
  dataloader_val.sampler.dataset.val_annotation_path="$DATASET_ROOT/annotation/val" \
  dataloader_val.sampler.dataset.test_annotation_path="$DATASET_ROOT/annotation/test" \
  dataloader_val.sampler.dataset.video_path="$DATASET_ROOT" \
  dataloader_val.sampler.dataset.video_size='[256,320]'
