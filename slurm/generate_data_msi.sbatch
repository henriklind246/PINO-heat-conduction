#!/bin/bash
#SBATCH --job-name=fno_datagen
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the project directory or set PROJECT_DIR}}"
export PROJECT_ROOT="$PROJECT_DIR"

# Active benchmark adapter (forcing_homogeneous|diffusion_forcing_single).
# Selects sampling + physics + item representation. generate_dataset.py reads
# $BENCHMARK as the default for --benchmark, so exporting it here is sufficient.
export BENCHMARK="${BENCHMARK:-forcing_homogeneous}"
DATA_OUT_DIR="${DATA_OUT_DIR:-$PROJECT_DIR/data/$BENCHMARK}"

GENERATOR_ARGS=(--dt 0.005)
NUM_SIMS="${NUM_SIMS:-512}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    DATA_OUT_DIR="$1"
    shift
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
    NUM_SIMS="$1"
    shift
fi

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pino}"

# Flush Python stdout line-by-line so `tail -f` shows progress in real time
export PYTHONUNBUFFERED=1

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_OUT_DIR=$DATA_OUT_DIR"
echo "NUM_SIMS=$NUM_SIMS"
echo "BENCHMARK=$BENCHMARK"
echo "Generator defaults: ${GENERATOR_ARGS[*]}"
echo "Extra generator args: $*"

# ──── Run ────
cd "$PROJECT_DIR"
mkdir -p "$DATA_OUT_DIR"
python "$PROJECT_DIR/data/generate_dataset.py" \
    --num-sims "$NUM_SIMS" \
    --save-dir "$DATA_OUT_DIR" \
    --benchmark "$BENCHMARK" \
    "${GENERATOR_ARGS[@]}" \
    "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
ls -lh "$DATA_OUT_DIR"/*.npy
