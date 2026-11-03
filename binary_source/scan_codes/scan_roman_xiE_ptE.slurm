#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --partition=milliways-int
#SBATCH --job-name="roman_xiE"
#SBATCH --output=roman_xiE_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# Roman maps in (xi_E, P/tE) with noise: PSPL and PSPL+parallax fits.
#   pilot:       sbatch scan_roman_xiE_ptE.slurm --pilot --workers 40
#   production:  sbatch --array=0-7 scan_roman_xiE_ptE.slurm --workers 40
#   summary:     python scan_roman_xiE_ptE.py --summarize
# Python environment and roman_rubin checkout are fixed here, so the job
# does not depend on the conda environment or variables of the terminal
# that submits it. ROMAN_RUBIN_REPO can still be overridden at submission.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PY=/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python
export ROMAN_RUBIN_REPO="${ROMAN_RUBIN_REPO:-/export/storage3/rubin/microlensing/romanrubin/roman_rubin_bspl_main}"
cd "$SLURM_SUBMIT_DIR"

# Provenance diagnostics (printed at the top of the log).
echo "[env] host=$(hostname) python=$PY"
echo "[env] ROMAN_RUBIN_REPO=$ROMAN_RUBIN_REPO"
echo "[env] git=$(command -v git || echo NOT_FOUND)"
echo "[env] binary_source HEAD: $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short=12 HEAD 2>&1)"
echo "[env] roman_rubin   HEAD: $(git -C "$ROMAN_RUBIN_REPO" rev-parse --short=12 HEAD 2>&1)"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    "$PY" scan_roman_xiE_ptE.py --shard "$SLURM_ARRAY_TASK_ID" \
        --n-shards "$SLURM_ARRAY_TASK_COUNT" "$@"
else
    "$PY" scan_roman_xiE_ptE.py "$@"
fi
