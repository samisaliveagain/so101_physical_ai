#!/usr/bin/env bash
#SBATCH --job-name=so101_gz_headless
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hpcwork/%u/so101_gazebo/logs/%x_%j.out
#SBATCH --error=/hpcwork/%u/so101_gazebo/logs/%x_%j.err

set -euo pipefail
HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_gazebo"}
export HPC_ROOT
export ROS_DOMAIN_ID=$((70 + SLURM_JOB_ID % 100))
export EPISODES=${EPISODES:-10}
export SEED=${SEED:-101}
export USE_NVIDIA=${USE_NVIDIA:-1}
export SOFTWARE_RENDERING=${SOFTWARE_RENDERING:-0}
mkdir -p "$HPC_ROOT/logs"
exec "$HPC_ROOT/project/apptainer/run_headless_gazebo_rwth.sh"
