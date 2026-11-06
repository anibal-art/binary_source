#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --partition=milliways-int
#SBATCH --job-name="xiE_ptE"
#SBATCH --output=xiE_ptE_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# Intrinsic (xi_E, P/tE) maps of D_BSPL-PSPL (magnification only).
#   estimate:  sbatch scan_xiE_ptE.slurm --u0 0.01 0.3 1 --estimate --workers 40
#   run:       sbatch scan_xiE_ptE.slurm --u0 0.01 0.3 1 --workers 40
# Same Python environment as scan_roman_xiE_ptE.slurm.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PY=/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python
export PATH="/home/anibalvarela/.conda/envs/gittools/bin:$PATH"   # git is not installed on the compute nodes
cd "$SLURM_SUBMIT_DIR"

echo "[env] host=$(hostname) python=$PY"
echo "[env] binary_source HEAD: $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short=12 HEAD 2>&1)"

"$PY" scan_xiE_ptE.py "$@"
