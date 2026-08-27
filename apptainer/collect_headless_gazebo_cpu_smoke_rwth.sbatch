#!/usr/bin/env bash
#SBATCH --job-name=so101_gz_cpu_smoke
#SBATCH --partition=c23ms
#SBATCH --account=default
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hpcwork/%u/so101_gazebo/logs/%x_%j.out
#SBATCH --error=/hpcwork/%u/so101_gazebo/logs/%x_%j.err

set -euo pipefail
HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_gazebo"}
export HPC_ROOT
export OUTPUT_ROOT="$HPC_ROOT/smoke_datasets"
export OUTPUT_NAME="so101_gazebo_cpu_smoke_${SLURM_JOB_ID}"
export ROS_DOMAIN_ID=$((170 + SLURM_JOB_ID % 50))
export EPISODES=1
export SEED=${SEED:-20260825}
export USE_NVIDIA=0
export SOFTWARE_RENDERING=1

mkdir -p "$HPC_ROOT/logs" "$OUTPUT_ROOT"
"$HPC_ROOT/project/apptainer/run_headless_gazebo_rwth.sh"

info="$OUTPUT_ROOT/$OUTPUT_NAME/meta/info.json"
episodes=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' "$info")
if [[ "$episodes" != 1 ]]; then
  echo "Smoke dataset has $episodes episodes instead of 1: $info" >&2
  exit 1
fi
echo "CPU-only smoke test passed: $OUTPUT_ROOT/$OUTPUT_NAME"
