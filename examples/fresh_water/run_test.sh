#!/bin/bash
#SBATCH --account=def-gdumas85
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=bilevel-train

set -euo pipefail

cd "$HOME/MetaMARL"

module load python/3.12
UV=/home/nadinem/.local/bin/uv
source .venv/bin/activate

# Enable WANDB OFFLINE MODE
export WANDB_MODE=offline
export WANDB_PROJECT=MetaMARL
export WANDB_DIR="$SCRATCH/metamarl/wandb/$SLURM_JOB_ID"
export WANDB_CACHE_DIR="$SCRATCH/metamarl/wandb_cache"

mkdir -p "$WANDB_DIR"
mkdir -p "$WANDB_CACHE_DIR"

# Raven
export RAVEN_CWD="$HOME/MetaMARL/examples/fresh_water/raven"
export RAVEN_CMD="$HOME/MetaMARL/examples/fresh_water/raven/2_Raven/Raven"

# run training
python run_test.py

echo "RAVEN_CWD=$RAVEN_CWD"
echo "RAVEN_CMD=$RAVEN_CMD"
echo "WANDB_DIR=$WANDB_DIR"

python run_test.py