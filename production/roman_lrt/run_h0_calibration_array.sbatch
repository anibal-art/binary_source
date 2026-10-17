#!/bin/bash
#SBATCH --job-name=bspl_lrt_h0
#SBATCH --partition=milliways-int
#SBATCH --array=0-14%15
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/export/storage3/rubin/microlensing/romanrubin/binary_source_lrt_runs/logs/h0_%A_%a.out
#SBATCH --error=/export/storage3/rubin/microlensing/romanrubin/binary_source_lrt_runs/logs/h0_%A_%a.err

set -euo pipefail

BS="${BS:?BS not exported}"
RR="${RR:?RR not exported}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT not exported}"
N_NULL="${N_NULL:-500}"

PYTHON="$HOME/.conda/envs/pyLIMA_test/bin/python"

export ROMAN_RUBIN_ROOT="$RR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$BS"

MANIFEST="$BS/production/roman_lrt/h0_configs_15.txt"

LINE=$((SLURM_ARRAY_TASK_ID + 1))

read -r MAG U0 TAG < <(
    sed -n "${LINE}p" "$MANIFEST"
)

if [ -z "${TAG:-}" ]; then
    echo "Could not read config for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Unique, non-overlapping RNG stream for each configuration.
SEED0=$((2026101600 + SLURM_ARRAY_TASK_ID * 1000000))

OUTDIR="$RUN_ROOT/$TAG"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "Roman BSPL-vs-PSPL H0 calibration"
echo "============================================================"
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "host                = $(hostname)"
echo "binary_source       = $BS"
echo "binary_source HEAD  = $(git rev-parse HEAD)"
echo "roman_rubin         = $RR"
echo "roman_rubin HEAD    = $(git -C "$RR" rev-parse HEAD)"
echo "F146 AB             = $MAG"
echo "u0                   = $U0"
echo "N null               = $N_NULL"
echo "workers              = $SLURM_CPUS_PER_TASK"
echo "seed0                = $SEED0"
echo "output               = $OUTDIR"
echo "============================================================"

"$PYTHON" \
  -m binary_source.analysis.roman_bspl_pspl_lrt_calibrate_h0 \
  --source-mags "$MAG" \
  --u0s "$U0" \
  --tE 30 \
  --n-null-per-config "$N_NULL" \
  --P-min 10 \
  --P-max 100000 \
  --n-omega 60 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --seed0 "$SEED0" \
  --output-dir "$OUTDIR"

echo
echo "DONE"
