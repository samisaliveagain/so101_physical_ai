#!/usr/bin/env bash
#SBATCH --job-name=so101_gz_100ep
#SBATCH --partition=c23ms
#SBATCH --account=default
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/hpcwork/%u/so101_gazebo/logs/%x_%j.out
#SBATCH --error=/hpcwork/%u/so101_gazebo/logs/%x_%j.err

set -euo pipefail

HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_gazebo"}
export HPC_ROOT
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HPC_ROOT/datasets"}
export OUTPUT_NAME=${OUTPUT_NAME:-"so101_gazebo_100ep_${SLURM_JOB_ID}"}
export ROS_DOMAIN_ID=$((220 + SLURM_JOB_ID % 30))
export EPISODES=100
export SEED=${SEED:-20260825}
export USE_NVIDIA=0
export SOFTWARE_RENDERING=1

mkdir -p "$HPC_ROOT/logs" "$OUTPUT_ROOT"
"$HPC_ROOT/project/apptainer/run_headless_gazebo_rwth.sh"

info="$OUTPUT_ROOT/$OUTPUT_NAME/meta/info.json"
episodes=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' "$info")
if [[ "$episodes" != 100 ]]; then
  echo "Dataset has $episodes episodes instead of 100: $info" >&2
  exit 1
fi

echo "CPU-only 100-episode collection passed: $OUTPUT_ROOT/$OUTPUT_NAME"
