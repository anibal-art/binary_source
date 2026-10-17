#!/bin/bash
#SBATCH --job-name=bspl_lrt_l3
#SBATCH --partition=milliways-int
#SBATCH --array=0-14%15
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/export/storage3/rubin/microlensing/romanrubin/binary_source_lrt_runs/logs/ladder3_%A_%a.out
#SBATCH --error=/export/storage3/rubin/microlensing/romanrubin/binary_source_lrt_runs/logs/ladder3_%A_%a.err

set -euo pipefail

BS="/export/storage3/rubin/microlensing/romanrubin/binary_source_lrt"

RR="/export/storage3/rubin/microlensing/romanrubin/roman_rubin_lrt_pandeia_20261016"

MANIFEST="$BS/production/roman_lrt/h0_ladder3_configs_15.txt"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT must be defined}"

N_NULL="${N_NULL:-500}"

PY="$HOME/.conda/envs/pyLIMA_test/bin/python"

export PYTHONPATH="$BS${PYTHONPATH:+:$PYTHONPATH}"

export ROMAN_RUBIN_ROOT="$RR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$BS"

LINE=$((SLURM_ARRAY_TASK_ID + 1))

read -r TAG MAG U0 < <(
    sed -n "${LINE}p" "$MANIFEST"
)

if [[ -z "${TAG:-}" ]]; then
    echo "ERROR: empty manifest row for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

SEED0=$((2026101600 + SLURM_ARRAY_TASK_ID * 1000000))

OUTDIR="$RUN_ROOT/$TAG"

echo "============================================================"
echo "task_id    = $SLURM_ARRAY_TASK_ID"
echo "tag        = $TAG"
echo "F146       = $MAG"
echo "u0         = $U0"
echo "N_NULL     = $N_NULL"
echo "seed0      = $SEED0"
echo "output     = $OUTDIR"
echo "host       = $(hostname)"
echo "============================================================"

srun "$PY" \
  -m binary_source.analysis.roman_bspl_pspl_lrt_calibrate_ladder_h0 \
  --source-mag "$MAG" \
  --u0 "$U0" \
  --tE 30 \
  --n-null "$N_NULL" \
  --seed0 "$SEED0" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --p-min 10 \
  --p-max 100000 \
  --n-omega 60 \
  --max-level omega_qmass_qflux \
  --output-dir "$OUTDIR"
