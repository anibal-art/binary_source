#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --partition=milliways-int
#SBATCH --job-name="sec6_h1"
#SBATCH --output=sec6_h1_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# Section 6, H1 Asimov grid (binary_source/analysis/roman_sec6_asimov_cells.py).
# Submit from the repository root:
#   pilot:       sbatch binary_source/analysis/roman_sec6_asimov_cells.slurm --pilot
#   production:  sbatch --array=0-7 binary_source/analysis/roman_sec6_asimov_cells.slurm --workers 40
#   summary:     python binary_source/analysis/roman_sec6_asimov_cells.py --summarize
# Finished tasks are skipped (safe to resubmit). The roman_rubin checkout is the
# one of the H0 calibration unless ROMAN_RUBIN_REPO is set at submission.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PY=/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python
export PATH="/home/anibalvarela/.conda/envs/gittools/bin:$PATH"   # git is not installed on the compute nodes
export ROMAN_RUBIN_REPO="${ROMAN_RUBIN_REPO:-/share/storage3/rubin/microlensing/romanrubin/roman_rubin}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "[env] host=$(hostname) python=$PY"
echo "[env] ROMAN_RUBIN_REPO=$ROMAN_RUBIN_REPO"
echo "[env] binary_source HEAD: $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short=12 HEAD 2>&1)"
echo "[env] roman_rubin   HEAD: $(git -C "$ROMAN_RUBIN_REPO" rev-parse --short=12 HEAD 2>&1)"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    "$PY" binary_source/analysis/roman_sec6_asimov_cells.py --shard "$SLURM_ARRAY_TASK_ID" \
        --n-shards "$SLURM_ARRAY_TASK_COUNT" "$@"
else
    "$PY" binary_source/analysis/roman_sec6_asimov_cells.py "$@"
fi
