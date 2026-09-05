#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# Binary Sources paper — complete reproduction pipeline
#
# Usage:
#
#   conda activate rubin-sim
#   ./reproduce_paper.sh
#
# Optional modes:
#
#   ./reproduce_paper.sh analysis
#   ./reproduce_paper.sh figures
#   ./reproduce_paper.sh validations
#
# The default mode is "all".
#
# Generated numerical products are written under results/.
# Generated figures are written under figures/.
#
# Neither directory is tracked by Git.
# =============================================================================


MODE="${1:-all}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/reproduction_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/reproduce_${MODE}_${TIMESTAMP}.log"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

exec > >(tee -a "${LOG_FILE}") 2>&1


echo
echo "======================================================================"
echo "BINARY SOURCES PAPER REPRODUCTION"
echo "======================================================================"
echo
echo "Repository : ${ROOT_DIR}"
echo "Mode       : ${MODE}"
echo "Started    : $(date)"
echo "Log        : ${LOG_FILE}"
echo


# -----------------------------------------------------------------------------
# Basic checks
# -----------------------------------------------------------------------------

if [[ ! -f "binary_source/degeneracy_fit.py" ]]; then
    echo "ERROR: run this script from the Binary Sources repository."
    exit 1
fi

if [[ ! -f "tools/paper_figure_manifest.txt" ]]; then
    echo "ERROR: missing tools/paper_figure_manifest.txt"
    exit 1
fi

mkdir -p results
mkdir -p figures/current
mkdir -p figures/appendix
mkdir -p figures/figures_lightcurves


run_stage () {
    local title="$1"
    shift

    echo
    echo "======================================================================"
    echo "${title}"
    echo "======================================================================"
    echo

    "$@"
}


# =============================================================================
# ANALYSIS
# =============================================================================

run_analysis () {

    # -------------------------------------------------------------------------
    # 1. Main intrinsic BSPL--PSPL scans
    # -------------------------------------------------------------------------

    run_stage \
        "1/7 — Intrinsic BSPL--PSPL scans" \
        python binary_source/scan_codes/run_final_scans.py


    # -------------------------------------------------------------------------
    # 2. Two-luminous-source qM--qf experiment
    # -------------------------------------------------------------------------

    run_stage \
        "2/7 — qM--qf numerical grid" \
        python binary_source/scan_codes/scan_qmass_qflux.py


    # -------------------------------------------------------------------------
    # 3. Photocenter-limit experiment
    # -------------------------------------------------------------------------

    run_stage \
        "3/7 — Photocenter-limit scan" \
        python binary_source/scan_codes/scan_photocenter_limit.py

    run_stage \
        "4/7 — Photocenter scaling analysis" \
        python binary_source/analysis/analyze_photocenter_scaling.py


    # -------------------------------------------------------------------------
    # 4. Roman F146 distinguishability
    # -------------------------------------------------------------------------

    mkdir -p results/roman_asimov_f146_tE30

    run_stage \
        "5/7 — Roman F146 smoke test" \
        python binary_source/analysis/roman_bspl_pspl_asimov.py \
            --mode smoke \
            --tE 30 \
            --source-mag 21

    run_stage \
        "6/7 — Roman F146 full Asimov grid" \
        python binary_source/analysis/roman_bspl_pspl_asimov.py \
            --mode grid \
            --tE 30 \
            --magnitudes 19 21 23 \
            --anchor-season 2 \
            --fit-window-tE 3.5 \
            --intrinsic-grid-dir \
                results/scan_many_tE_200x200/scan_u0_tE30 \
            --intrinsic-u0-max 1.0 \
            --output \
                results/roman_asimov_f146_tE30/roman_intrinsic_grid_f146_tE30.npz


    # -------------------------------------------------------------------------
    # 5. Physical PARSEC/F146 calculation
    #
    # This script also generates its two physical-source figures.
    # -------------------------------------------------------------------------

    run_stage \
        "7/7 — Physical PARSEC F146 calculation" \
        python isochrone_phot_cancel.py
}


# =============================================================================
# VALIDATIONS
# =============================================================================

run_validations () {

    run_stage \
        "Validation — optimizer and grid audit" \
        python binary_source/validation/validate_optimizer_and_grid_audit.py

    run_stage \
        "Validation — fitting window and blending" \
        python binary_source/validation/validate_window_and_blending.py

    run_stage \
        "Validation — Roman tE bound" \
        python binary_source/validation/validate_roman_te_bound.py

    run_stage \
        "Validation — geometry robustness" \
        python binary_source/validation/validate_geometry_robustness.py
}


# =============================================================================
# FIGURES
# =============================================================================

run_figures () {

    # -------------------------------------------------------------------------
    # Light-curve examples
    # -------------------------------------------------------------------------

    run_stage \
        "Figure production — illustrative light curves" \
        python binary_source/plot_codes/plot_lightcurve_examples.py


    # -------------------------------------------------------------------------
    # Main intrinsic figures
    # -------------------------------------------------------------------------

    run_stage \
        "Figure production — intrinsic degeneracy maps" \
        python binary_source/plot_codes/plot_intrinsic_tE_paper.py

    run_stage \
        "Figure production — intrinsic PSPL biases" \
        python binary_source/plot_codes/plot_intrinsic_biases_paper.py


    # -------------------------------------------------------------------------
    # qM--qf figures
    # -------------------------------------------------------------------------

    run_stage \
        "Figure production — qM--qf figures" \
        python binary_source/plot_codes/run_paper_figures.py


    # -------------------------------------------------------------------------
    # Photocenter figure
    # -------------------------------------------------------------------------

    run_stage \
        "Figure production — photocenter scaling" \
        python binary_source/plot_codes/plot_photocenter_scaling.py


    # -------------------------------------------------------------------------
    # Roman figure
    # -------------------------------------------------------------------------

    run_stage \
        "Figure production — Roman F146 comparison" \
        python binary_source/plot_codes/plot_roman_intrinsic_comparison.py


    # -------------------------------------------------------------------------
    # Appendix figures
    # -------------------------------------------------------------------------

    run_stage \
        "Appendix figure — optimizer audit" \
        python binary_source/plot_codes/plot_stratified_refit_audit.py

    run_stage \
        "Appendix figure — window sensitivity" \
        python binary_source/plot_codes/plot_final_validation.py

    run_stage \
        "Appendix figure — Roman tE bound" \
        python binary_source/plot_codes/plot_roman_te_bound_validation.py

    run_stage \
        "Appendix figure — geometry robustness" \
        python binary_source/plot_codes/plot_geometry_robustness_appendix.py
}


# =============================================================================
# FINAL AUDIT
# =============================================================================

run_audit () {

    run_stage \
        "Final reproducibility audit" \
        python tools/audit_paper_reproducibility.py


    echo
    echo "Checking expected paper assets..."
    echo

    missing=0

    while IFS= read -r fig; do

        [[ -z "${fig}" ]] && continue
        [[ "${fig}" =~ ^# ]] && continue

        if [[ -f "${fig}" ]]; then
            echo "OK      ${fig}"
        else
            echo "MISSING ${fig}"
            missing=1
        fi

    done < tools/paper_figure_manifest.txt


    echo

    if [[ "${missing}" -ne 0 ]]; then
        echo "ERROR: one or more expected paper figures are missing."
        exit 1
    fi

    echo "All expected paper figures are present."
}


# =============================================================================
# EXECUTION
# =============================================================================

case "${MODE}" in

    all)
        run_analysis
        run_validations
        run_figures
        run_audit
        ;;

    analysis)
        run_analysis
        ;;

    validations)
        run_validations
        ;;

    figures)
        run_figures
        run_audit
        ;;

    audit)
        run_audit
        ;;

    *)
        echo "Unknown mode: ${MODE}"
        echo
        echo "Usage:"
        echo "  ./reproduce_paper.sh"
        echo "  ./reproduce_paper.sh all"
        echo "  ./reproduce_paper.sh analysis"
        echo "  ./reproduce_paper.sh validations"
        echo "  ./reproduce_paper.sh figures"
        echo "  ./reproduce_paper.sh audit"
        exit 1
        ;;
esac


echo
echo "======================================================================"
echo "REPRODUCTION FINISHED SUCCESSFULLY"
echo "======================================================================"
echo
echo "Finished : $(date)"
echo "Log      : ${LOG_FILE}"
echo
