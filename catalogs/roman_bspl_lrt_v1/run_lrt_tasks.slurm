#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --partition=milliways-int
#SBATCH --job-name="lrt_tasks"
#SBATCH --output=lrt_tasks_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# Run every line of a task file (catalogs/roman_bspl_lrt_v1/lrt_tasks.py)
# through roman_bspl_lrt_catalog_runner, WORKERS events at a time.
# Submit from the repository root:
#   sbatch catalogs/roman_bspl_lrt_v1/run_lrt_tasks.slurm <tasks.txt> <fit_policy.json> [workers]
# Events whose output JSON already exists are skipped (safe to resubmit).
# One log per event next to its JSON (<output>.log).

TASKS=$1
POLICY=$2
WORKERS=${3:-${SLURM_CPUS_PER_TASK:-40}}

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

echo "[env] host=$(hostname) python=$PY workers=$WORKERS"
echo "[env] binary_source HEAD: $(git rev-parse --short=12 HEAD 2>&1)"
echo "[env] tasks=$TASKS ($(grep -c . "$TASKS") lines) policy=$POLICY"
echo "[env] policy sha256: $(sha256sum "$POLICY" | cut -d' ' -f1)"

run_one() {
    read -r cat idx kind out <<< "$1"
    [ -s "$out" ] && return 0
    "$PY" -u binary_source/analysis/roman_bspl_lrt_catalog_runner.py \
        --catalog "$cat" --catalog-event-index "$idx" --truth-kind "$kind" \
        --fit-policy "$POLICY" --output "$out" > "$out.log" 2>&1 \
        && echo "[ok]   $kind $idx" || echo "[FAIL] $kind $idx (see $out.log)"
}
export -f run_one
export PY POLICY

start=$(date +%s)
grep . "$TASKS" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
echo "[done] $(ls "$(dirname "$TASKS")"/json/*.json 2>/dev/null | wc -l) JSON in $(( ($(date +%s) - start) / 60 )) min"
