#!/bin/bash
#SBATCH --job-name=pino_gates
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

# Direct-state gates that every gated physics screen requires: the raw/exact
# gate (PHASE0_DIR) and the MG gate (PHASE4_DIR). The training launchers default
# to the same paths. An existing gate is kept; delete its directory to rerun it.
#
#   sbatch slurm/generate_gates_msi.sbatch

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the project directory or set PROJECT_DIR}}"
export PROJECT_ROOT="$PROJECT_DIR"
PHASE0_DIR="${PHASE0_DIR:-$PROJECT_DIR/runs/gates/phase0_direct_state}"
PHASE4_DIR="${PHASE4_DIR:-$PROJECT_DIR/runs/gates/phase4_direct_state_mg}"
export PYTHONUNBUFFERED=1

module load miniforge
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pino}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$PHASE0_DIR")" "$(dirname "$PHASE4_DIR")"
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PHASE0_DIR=$PHASE0_DIR PHASE4_DIR=$PHASE4_DIR"

if [[ ! -f "$PHASE0_DIR/final_metrics.json" ]]; then
    python scripts/diagnose_fv_direct_state.py --output-dir "$PHASE0_DIR"
fi
if [[ ! -f "$PHASE4_DIR/final_metrics.json" ]]; then
    python scripts/diagnose_fv_direct_state.py --mg --output-dir "$PHASE4_DIR"
fi

echo "Job $SLURM_JOB_ID finished at $(date)"
