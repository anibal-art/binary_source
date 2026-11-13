#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --partition=milliways-int
#SBATCH --job-name="omega_prof"
#SBATCH --output=omega_prof_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# BSPL profile likelihood in omega (binary_source/analysis/lrt_omega_profile.py)
# for every runner JSON of a task run made with SAVE_ARRAYS=1.
# Submit from the repository root:
#   sbatch catalogs/roman_bspl_lrt_v1/run_omega_profiles.slurm <run_outdir> <fit_policy.json> [extra args]
# Outputs <run_outdir>/omega_profile/<event>_omega_profile.json; existing ones
# are skipped (safe to resubmit). Extra args go to lrt_omega_profile.py.

RUN=$1
POLICY=$2
shift 2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PY=/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python
export PATH="/home/anibalvarela/.conda/envs/gittools/bin:$PATH"   # git is not installed on the compute nodes
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "[env] host=$(hostname) python=$PY"
echo "[env] binary_source HEAD: $(git rev-parse --short=12 HEAD 2>&1)"
echo "[env] run=$RUN policy=$POLICY extra=$* workers=${SLURM_CPUS_PER_TASK:-40}"

WORKERS=${SLURM_CPUS_PER_TASK:-40}
EXTRA="$*"
mkdir -p "$RUN/omega_profile"

# One independent process per event (no multiprocessing.Pool: every pyLIMA fit
# starts a multiprocessing Manager, which Pool workers are not allowed to do).
run_one() {
    json=$1
    name=$(basename "${json%.json}")
    out="$RUN/omega_profile/${name}_omega_profile.json"
    [ -s "$out" ] && return 0
    "$PY" -u binary_source/analysis/lrt_omega_profile.py \
        --json "$json" --fit-policy "$POLICY" --out-dir "$RUN/omega_profile" --no-plot $EXTRA \
        > "$RUN/omega_profile/${name}.log" 2>&1 \
        && [ -s "$out" ] && echo "[ok]   $name" || echo "[FAIL] $name (see $RUN/omega_profile/${name}.log)"
}
export -f run_one
export PY POLICY RUN EXTRA

start=$(date +%s)
ls "$RUN"/json/*.json | grep -v '_omega_profile.json$' | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
echo "[done] $(ls "$RUN"/omega_profile/*_omega_profile.json 2>/dev/null | wc -l) profiles in $(( ($(date +%s) - start) / 60 )) min"
