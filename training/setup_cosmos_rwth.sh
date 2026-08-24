#!/usr/bin/env bash
# Run on `ssh rwth-gpu` once. Everything large goes to /hpcwork, not the full $HOME quota.
set -euo pipefail

HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_cosmos"}
COSMOS_REF=${COSMOS_REF:-a2c298b0a3df3778b973fe65e9e58877b292d8a7}
COSMOS_DIR="$HPC_ROOT/src/cosmos-predict2.5"
export HF_HOME="$HPC_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$HPC_ROOT/cache/xdg"
export UV_CACHE_DIR="$HPC_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$HPC_ROOT/tools/uv-python"
export UV_TOOL_DIR="$HPC_ROOT/tools/uv-tools"
export TMPDIR="$HPC_ROOT/tmp"
mkdir -p "$HPC_ROOT" "$HF_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
  "$UV_TOOL_DIR" "$TMPDIR" \
  "$HPC_ROOT/data" "$HPC_ROOT/output" "$HPC_ROOT/logs"

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0

if [[ ! -d "$COSMOS_DIR/.git" ]]; then
  git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_DIR"
fi
git -C "$COSMOS_DIR" fetch origin
git -C "$COSMOS_DIR" checkout "$COSMOS_REF"

if ! command -v uv >/dev/null; then
  python -m venv "$HPC_ROOT/tools/uv-venv"
  "$HPC_ROOT/tools/uv-venv/bin/pip" install 'uv>=0.8'
export PATH="$HPC_ROOT/tools/uv-venv/bin:$PATH"
fi
cd "$COSMOS_DIR"
if command -v git-lfs >/dev/null; then git lfs pull; fi
# The pinned NVIDIA flash-attn CUDA wheel in this revision is cp310-only even
# though the repository's .python-version currently says 3.13.
uv python install 3.10
uv sync --python 3.10 --extra=cu128

echo
echo "Cosmos environment ready: $COSMOS_DIR/.venv"
echo "Before the first job, authenticate on rwth-gpu and accept NVIDIA's model license:"
echo "  export HF_HOME=$HF_HOME"
echo "  $COSMOS_DIR/.venv/bin/hf auth login"
