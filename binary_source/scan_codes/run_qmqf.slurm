#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --partition=milliways-int
#SBATCH --job-name="qmqf"
#SBATCH --output=qmqf_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# (q_M, q_f) scan over the orientations of the (xi_E, P/tE) maps
# (run_final_scans.py --only qmass_qflux). Submit from the repository root.
#   timing test: BINARY_SOURCE_QMQF_N=3 BINARY_SOURCE_QMQF_PTE=1 BINARY_SOURCE_QMQF_NGEO=1 \
#                sbatch binary_source/scan_codes/run_qmqf.slurm
#   paper run  : sbatch binary_source/scan_codes/run_qmqf.slurm
# run_final_scans.py uses at most 12 workers (final_config.MAX_WORKERS).

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

export BINARY_SOURCE_QMQF_TE=${BINARY_SOURCE_QMQF_TE:-30}
export BINARY_SOURCE_QMQF_N=${BINARY_SOURCE_QMQF_N:-60}
export BINARY_SOURCE_QMQF_PTE=${BINARY_SOURCE_QMQF_PTE:-1,10}
export BINARY_SOURCE_QMQF_NGEO=${BINARY_SOURCE_QMQF_NGEO:-24}

PY=/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python
export PATH="/home/anibalvarela/.conda/envs/gittools/bin:$PATH"   # git is not installed on the compute nodes
cd "$SLURM_SUBMIT_DIR"

echo "[env] host=$(hostname) cpus=$SLURM_CPUS_PER_TASK python=$PY"
echo "[env] binary_source HEAD: $(git rev-parse --short=12 HEAD 2>&1)"
echo "[env] tE=$BINARY_SOURCE_QMQF_TE N=$BINARY_SOURCE_QMQF_N P/tE=$BINARY_SOURCE_QMQF_PTE NGEO=$BINARY_SOURCE_QMQF_NGEO"

start=$(date +%s)
"$PY" binary_source/scan_codes/run_final_scans.py --only qmass_qflux
echo "[time] elapsed $(( $(date +%s) - start )) s"
