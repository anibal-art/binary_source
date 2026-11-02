#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --partition=PARTICION_CHE
#SBATCH --job-name="roman_xiE"
#SBATCH --output=roman_xiE_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# Roman maps in (xi_E, P/tE) with noise: PSPL and PSPL+parallax fits.
#   pilot:       sbatch scan_roman_xiE_ptE.slurm --pilot --workers 40
#   production:  sbatch --array=0-7 scan_roman_xiE_ptE.slurm --workers 40
#   summary:     python scan_roman_xiE_ptE.py --summarize
# roman_rubin must be the sibling checkout on branch bspl-xallarap-model,
# or set ROMAN_RUBIN_REPO.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PY="$CONDA_PREFIX/bin/python"
cd "$SLURM_SUBMIT_DIR"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    "$PY" scan_roman_xiE_ptE.py --shard "$SLURM_ARRAY_TASK_ID" \
        --n-shards "$SLURM_ARRAY_TASK_COUNT" "$@"
else
    "$PY" scan_roman_xiE_ptE.py "$@"
fi
