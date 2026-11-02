#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --partition=milliways-int
#SBATCH --job-name=xiE_ptE
#SBATCH --output=scan_xiE_ptE_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com
##SBATCH --time=12:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# python of the conda env active at submission (BASH_ENV/Lmod can change PATH)
PY="${CONDA_PREFIX:?submit with the conda env active}/bin/python"
"$PY" -c "import pyLIMA, numpy, scipy" || { echo "pyLIMA not importable with $PY"; exit 1; }

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} CPUS=${SLURM_CPUS_PER_TASK} PY=$PY"
echo "COMMIT=$(git rev-parse --short=12 HEAD) DIRTY=$(git status --porcelain | wc -l)"
cd "$SLURM_SUBMIT_DIR"
"$PY" -u binary_source/scan_codes/scan_xiE_ptE.py --workers "${SLURM_CPUS_PER_TASK}" "$@"
