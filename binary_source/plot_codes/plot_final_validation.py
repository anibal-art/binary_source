#!/usr/bin/env python3
"""
Publication figures for the final inexpensive validation tests.

Inputs
------
results/validation_final_window_blending/window_sensitivity.csv
results/validation_final_window_blending/roman_true_blending.csv

Outputs
-------
figures/appendix/window_sensitivity.pdf
figures/appendix/window_sensitivity.png

figures/appendix/roman_true_blending_sensitivity.pdf
figures/appendix/roman_true_blending_sensitivity.png

The figures summarize:

1. Sensitivity of D_BSPL-PSPL to the adopted event-centered
   integration window.

2. Sensitivity of the Roman Asimov model separation to true
   positive blending.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
    / "validation_final_window_blending"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "figures"
    / "appendix"
)


# ============================================================
# Paper style
# ============================================================

def apply_style():
    """
    Prefer the repository paper style if available.
    Fall back to an equivalent compact publication style.
    """

    try:
        from binary_source.plot_codes.paper_style import apply_paper_style

        apply_paper_style()
        return

    except Exception:
        pass

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "STIXGeneral",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",

            "font.size": 11,

            "axes.labelsize": 12,
            "axes.titlesize": 12,

            "xtick.labelsize": 10,
            "ytick.labelsize": 10,

            "legend.fontsize": 9,

            "axes.linewidth": 1.0,

            "xtick.direction": "in",
            "ytick.direction": "in",

            "xtick.top": True,
            "ytick.right": True,

            "figure.facecolor": "white",
            "axes.facecolor": "white",

            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


# ============================================================
# Labels
# ============================================================

WINDOW_LABELS = {
    "one_short": "short period",
    "one_intermediate": "intermediate period",
    "one_long": "long period",
    "one_hidden_long": "hidden long period",
    "one_small_u0": r"small $u_0$",
    "two_cancel": r"$q_f=q_M$",
    "two_off_cancel": r"$q_f=0.1$",
}


BLEND_LABELS = {
    "A_hidden": "A: hidden",
    "B_near_100": r"B: near $\Delta\chi^2=100$",
    "C_near_500": r"C: near $\Delta\chi^2=500$",
    "D_clear": "D: clear",
}


# ============================================================
# Utilities
# ============================================================

def require_columns(
    df: pd.DataFrame,
    required,
    filename: Path,
):
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{filename} is missing columns: {missing}"
        )


def save_figure(
    fig,
    pdf_path: Path,
    png_path: Path,
):
    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(" ", pdf_path)
    print(" ", png_path)


# ============================================================
# Figure 1: integration-window sensitivity
# ============================================================

def plot_window_sensitivity(
    csv_path: Path,
    output_dir: Path,
):
    """
    Two-panel design.

    Left:
        broad configurations whose window dependence is tiny.
        We multiply fractional changes by 1e5 so the structure
        remains visible.

    Right:
        slowly varying long-period configurations, displayed
        directly in percent.

    This avoids compressing the broad configurations against
    the percent-level long-period sensitivity.
    """

    df = pd.read_csv(
        csv_path
    )

    require_columns(
        df,
        [
            "case",
            "window_tE",
            "D",
            "D_over_3p5",
            "fractional_D_change",
        ],
        csv_path,
    )


    broad_cases = [
        "one_short",
        "one_intermediate",
        "one_small_u0",
        "two_cancel",
        "two_off_cancel",
    ]

    long_cases = [
        "one_long",
        "one_hidden_long",
    ]


    apply_style()


    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            10.0,
            4.2,
        ),
    )


    # ========================================================
    # Panel A: broad configurations
    # ========================================================

    for case in broad_cases:

        sub = (
            df[
                df["case"]
                == case
            ]
            .sort_values(
                "window_tE"
            )
        )

        x = sub[
            "window_tE"
        ].to_numpy(
            dtype=float
        )

        y = (
            1.0e5
            * sub[
                "fractional_D_change"
            ].to_numpy(
                dtype=float
            )
        )


        ax1.plot(
            x,
            y,
            marker="o",
            linewidth=1.4,
            markersize=5,
            label=WINDOW_LABELS[
                case
            ],
        )


    ax1.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax1.axvline(
        3.5,
        linewidth=1.0,
        linestyle=":",
    )


    ax1.set_xticks(
        [
            2.5,
            3.5,
            5.0,
        ]
    )

    ax1.set_xlabel(
        r"integration half-window [$t_E$]"
    )

    ax1.set_ylabel(
        r"$10^5\,[D(W)/D(3.5t_E)-1]$"
    )

    ax1.set_title(
        "Broad configurations"
    )

    ax1.grid(
        True,
        alpha=0.20,
    )

    ax1.legend(
        frameon=False,
        loc="best",
    )


    # ========================================================
    # Panel B: long-period configurations
    # ========================================================

    for case in long_cases:

        sub = (
            df[
                df["case"]
                == case
            ]
            .sort_values(
                "window_tE"
            )
        )

        x = sub[
            "window_tE"
        ].to_numpy(
            dtype=float
        )

        y = (
            100.0
            * sub[
                "fractional_D_change"
            ].to_numpy(
                dtype=float
            )
        )


        ax2.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=5,
            label=WINDOW_LABELS[
                case
            ],
        )


    ax2.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax2.axvline(
        3.5,
        linewidth=1.0,
        linestyle=":",
    )


    ax2.set_xticks(
        [
            2.5,
            3.5,
            5.0,
        ]
    )

    ax2.set_xlabel(
        r"integration half-window [$t_E$]"
    )

    ax2.set_ylabel(
        r"$100\,[D(W)/D(3.5t_E)-1]$ [\%]"
    )

    ax2.set_title(
        "Long-period configurations"
    )

    ax2.grid(
        True,
        alpha=0.20,
    )

    ax2.legend(
        frameon=False,
        loc="best",
    )


    # ========================================================
    # Panel labels
    # ========================================================

    ax1.text(
        -0.13,
        1.03,
        "A",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )

    ax2.text(
        -0.13,
        1.03,
        "B",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )


    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.15,
        top=0.90,
        wspace=0.28,
    )


    pdf_path = (
        output_dir
        / "window_sensitivity.pdf"
    )

    png_path = (
        output_dir
        / "window_sensitivity.png"
    )


    save_figure(
        fig=fig,
        pdf_path=pdf_path,
        png_path=png_path,
    )


# ============================================================
# Figure 2: Roman true-blending sensitivity
# ============================================================

def plot_true_blending(
    csv_path: Path,
    output_dir: Path,
):
    """
    Two panels:

    A. Absolute Roman model-separation response,
       Delta chi2 / Delta chi2_0.

    B. Survey-normalized mismatch response,
       D_Roman,w / D_Roman,w,0.

    The second panel is useful because D_Roman,w does not have
    to decrease monotonically even when Delta chi2 does.
    """

    df = pd.read_csv(
        csv_path
    )

    require_columns(
        df,
        [
            "case",
            "fb_over_fs_true",
            "delta_chi2",
            "delta_chi2_over_unblended",
            "D_roman_w",
            "Droman_over_unblended",
        ],
        csv_path,
    )


    case_order = [
        "A_hidden",
        "B_near_100",
        "C_near_500",
        "D_clear",
    ]


    apply_style()


    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            10.0,
            4.2,
        ),
    )


    # ========================================================
    # Panel A: Delta chi2
    # ========================================================

    for case in case_order:

        sub = (
            df[
                df["case"]
                == case
            ]
            .sort_values(
                "fb_over_fs_true"
            )
        )

        x = sub[
            "fb_over_fs_true"
        ].to_numpy(
            dtype=float
        )

        y = sub[
            "delta_chi2_over_unblended"
        ].to_numpy(
            dtype=float
        )


        ax1.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=5,
            label=BLEND_LABELS[
                case
            ],
        )


    ax1.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
    )


    ax1.set_xticks(
        [
            0.0,
            0.1,
            0.3,
            1.0,
            3.0,
        ]
    )

    ax1.set_xlabel(
        r"true blend ratio $F_b/F_s$"
    )

    ax1.set_ylabel(
        r"$\Delta\chi^2_{\rm Roman}/"
        r"\Delta\chi^2_{\rm Roman}(F_b=0)$"
    )

    ax1.set_title(
        "Absolute model separation"
    )

    ax1.grid(
        True,
        alpha=0.20,
    )

    ax1.legend(
        frameon=False,
        loc="best",
    )


    # ========================================================
    # Panel B: normalized Roman mismatch
    # ========================================================

    for case in case_order:

        sub = (
            df[
                df["case"]
                == case
            ]
            .sort_values(
                "fb_over_fs_true"
            )
        )

        x = sub[
            "fb_over_fs_true"
        ].to_numpy(
            dtype=float
        )

        y = sub[
            "Droman_over_unblended"
        ].to_numpy(
            dtype=float
        )


        ax2.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=5,
            label=BLEND_LABELS[
                case
            ],
        )


    ax2.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
    )


    ax2.set_xticks(
        [
            0.0,
            0.1,
            0.3,
            1.0,
            3.0,
        ]
    )

    ax2.set_xlabel(
        r"true blend ratio $F_b/F_s$"
    )

    ax2.set_ylabel(
        r"$D_{\rm Roman,w}/"
        r"D_{\rm Roman,w}(F_b=0)$"
    )

    ax2.set_title(
        "Survey-normalized mismatch"
    )

    ax2.grid(
        True,
        alpha=0.20,
    )


    # No second legend: same case ordering as left panel.


    # ========================================================
    # Panel labels
    # ========================================================

    ax1.text(
        -0.13,
        1.03,
        "A",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )

    ax2.text(
        -0.13,
        1.03,
        "B",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )


    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.15,
        top=0.90,
        wspace=0.30,
    )


    pdf_path = (
        output_dir
        / "roman_true_blending_sensitivity.pdf"
    )

    png_path = (
        output_dir
        / "roman_true_blending_sensitivity.png"
    )


    save_figure(
        fig=fig,
        pdf_path=pdf_path,
        png_path=png_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )


    parser.add_argument(
        "--figure",
        choices=[
            "window",
            "blending",
            "all",
        ],
        default="all",
    )


    args = parser.parse_args()


    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else ROOT
        / args.results_dir
    )


    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT
        / args.output_dir
    )


    print("=" * 80)
    print("FINAL VALIDATION FIGURES")
    print("=" * 80)

    print(
        "results:",
        results_dir,
    )

    print(
        "output :",
        output_dir,
    )


    window_csv = (
        results_dir
        / "window_sensitivity.csv"
    )

    blending_csv = (
        results_dir
        / "roman_true_blending.csv"
    )


    if args.figure in (
        "window",
        "all",
    ):

        if not window_csv.exists():

            raise FileNotFoundError(
                window_csv
            )

        print()
        print(
            "Making window-sensitivity figure..."
        )

        plot_window_sensitivity(
            csv_path=window_csv,
            output_dir=output_dir,
        )


    if args.figure in (
        "blending",
        "all",
    ):

        if not blending_csv.exists():

            raise FileNotFoundError(
                blending_csv
            )

        print()
        print(
            "Making true-blending figure..."
        )

        plot_true_blending(
            csv_path=blending_csv,
            output_dir=output_dir,
        )


    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
