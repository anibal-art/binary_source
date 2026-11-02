#!/bin/bash

#SBATCH --job-name=bspl_pi_ultrafine
#SBATCH --partition=milliways-int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/intrinsic_parallax_ultrafine_%j.out
#SBATCH --error=logs/intrinsic_parallax_ultrafine_%j.err

# Do not enable nounset before sourcing conda.sh:
# conda's shell initialization reads PS1 in a non-interactive batch shell.
set -eo pipefail

REPO="/share/storage3/rubin/microlensing/romanrubin/binary_source_lrt_blend_parallax_dev"

cd "$REPO"

# Make conda available in the batch shell.
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyLIMA_test

# Safe to enable nounset after Conda initialization.
set -u

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# One numerical thread per Python process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "INTRINSIC BSPL / PSPL+PARALLAX ULTRAFINE GRID"
echo "============================================================"
echo "date          = $(date)"
echo "host          = $(hostname)"
echo "python        = $(which python)"
echo "python version= $(python --version 2>&1)"
echo "job id        = ${SLURM_JOB_ID}"
echo "cpus          = ${SLURM_CPUS_PER_TASK}"
echo "git commit    = $(git rev-parse HEAD)"
echo "branch        = $(git branch --show-current)"
echo
echo "grid          = 601 x 241"
echo "nodes         = 144841"
echo "u0            = 1e-2 ... 10"
echo "P             = 10 ... 1e5 d"
echo "tE            = 30 d"
echo "workers       = ${SLURM_CPUS_PER_TASK}"
echo "============================================================"

time python -u \
    decision_tests/scan_intrinsic_xallarap_vs_pspl_parallax_ultrafine.py \
    --workers "${SLURM_CPUS_PER_TASK}"

echo
echo "============================================================"
echo "FINISHED"
echo "date = $(date)"
echo "============================================================"
