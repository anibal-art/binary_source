#!/usr/bin/env python3
"""
Publication figures for the final inexpensive validation tests.

Inputs
------
results/validation_final_window_blending/window_sensitivity.csv
results/validation_final_window_blending/roman_true_blending.csv

Outputs
-------
figures/appendix/window_sensitivity.{pdf,png}            (Fig. 11)
figures/appendix/window_sensitivity_alt.{pdf,png}        (single-panel alternative)
figures/appendix/roman_true_blending_sensitivity.{pdf,png}  (Fig. 12)

The figures summarize:

1. Sensitivity of D_BSPL-PSPL to the adopted event-centered
   integration window.
2. Sensitivity of the Roman Asimov model separation to true
   positive blending.

Plotting only: the CSV values are drawn unchanged (apart from the
fixed display scalings 1e5 and 100 noted in the axis labels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_DIR = ROOT / "results" / "validation_final_window_blending"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "appendix"
VERIFY_DIR = ROOT / "figures" / "_verify"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    apply_paper_style,
    fig_width,
    panel_label,
    save_figure,
    C_BLACK, C_BLUE, C_VERMILION, C_GREEN, C_PURPLE, C_ORANGE, C_SKY,
    LW_THIN,
)


# ============================================================
# Configuration styles (one colour + marker per configuration)
# ============================================================

WINDOW_CASES = [
    "one_short",
    "one_intermediate",
    "one_small_u0",
    "two_cancel",
    "two_off_cancel",
    "one_long",
    "one_hidden_long",
]

WINDOW_STYLE = {
    "one_short": dict(color=C_BLACK, marker="o"),
    "one_intermediate": dict(color=C_BLUE, marker="s"),
    "one_small_u0": dict(color=C_GREEN, marker="^"),
    "two_cancel": dict(color=C_VERMILION, marker="D"),
    "two_off_cancel": dict(color=C_PURPLE, marker="v"),
    "one_long": dict(color=C_ORANGE, marker="P"),
    "one_hidden_long": dict(color=C_SKY, marker="X"),
}

BLEND_CASES = ["A_hidden", "B_near_100", "C_near_500", "D_clear"]

BLEND_STYLE = {
    "A_hidden": dict(color=C_BLACK, marker="o"),
    "B_near_100": dict(color=C_BLUE, marker="s"),
    "C_near_500": dict(color=C_VERMILION, marker="^"),
    "D_clear": dict(color=C_GREEN, marker="D"),
}


def fmt_num(x, sig=2):
    """Compact value for labels: 1 (u0) or 2 (P/tE) significant figures."""
    return f"{float(f'{float(x):.{sig}g}'):g}"


def physical_label(row, show_qf=False):
    """'(u0, P/tE)' label from the stored configuration parameters."""
    lab = rf"$({fmt_num(row['u0_true'], 1)},\ {fmt_num(row['P_over_tE'], 2)})$"
    if show_qf:
        qf = float(row["qflux"])
        qm = float(row["q_mass"])
        if qf > 0 and np.isclose(qf, qm):
            lab += r", $q_f=q_M$"
        elif qf > 0:
            lab += rf", $q_f={qf:g}$"
    return lab


# ============================================================
# Utilities
# ============================================================

def require_columns(df: pd.DataFrame, required, filename: Path):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{filename} is missing columns: {missing}")


def window_axis(ax):
    ax.set_xticks([2.5, 3.5, 5.0])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f$"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(2.3, 5.2)
    ax.set_xlabel(r"half-window [$t_E$]")


# ============================================================
# Figure 11: integration-window sensitivity
# ============================================================

def plot_window_sensitivity(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    require_columns(df, ["case", "window_tE", "D", "D_over_3p5", "fractional_D_change",
                         "u0_true", "P_over_tE", "q_mass", "qflux"], csv_path)

    labels = {c: physical_label(df[df["case"] == c].iloc[0], show_qf=True)
              for c in WINDOW_CASES}

    broad_cases = ["one_short", "one_intermediate", "one_small_u0",
                   "two_cancel", "two_off_cancel"]
    long_cases = ["one_long", "one_hidden_long"]

    apply_paper_style()

    # ---------------- manuscript version: two panels, same quantities
    fig = plt.figure(figsize=(fig_width(0.78), 3.0))
    ax1 = fig.add_axes([0.12, 0.395, 0.35, 0.585])
    ax2 = fig.add_axes([0.625, 0.395, 0.35, 0.585])

    for ax, cases, scale in ((ax1, broad_cases, 1.0e5), (ax2, long_cases, 100.0)):
        ax.axhline(0.0, color="0.6", lw=LW_THIN, ls=":", zorder=0)
        for case in cases:
            sub = df[df["case"] == case].sort_values("window_tE")
            ax.plot(sub["window_tE"].to_numpy(float),
                    scale * sub["fractional_D_change"].to_numpy(float),
                    ms=3.5, label=labels[case], **WINDOW_STYLE[case])
        window_axis(ax)

    ax1.set_ylabel(r"$10^5\,[D(W)/D(3.5\,t_E)-1]$")
    ax2.set_ylabel(r"$[D(W)/D(3.5\,t_E)-1]$ (\%)")
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f$"))
    ax1.set_ylim(-9.0, 4.0)
    ax2.set_ylim(-2.2, 1.0)
    panel_label(ax1, "a")
    panel_label(ax2, "b")

    handles = [h for ax in (ax1, ax2) for h in ax.get_legend_handles_labels()[0]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, title=r"$(u_0,\ P/t_E)$", handlelength=1.6,
               columnspacing=1.0)

    save_figure(fig, output_dir / "window_sensitivity")
    plt.close(fig)

    # ---------------- alternative: one panel, |D(W)/D(3.5 tE) - 1|, log y
    fig = plt.figure(figsize=(fig_width(0.78), 2.45))
    ax = fig.add_axes([0.13, 0.17, 0.48, 0.79])
    plotted_alt = {}
    for case in WINDOW_CASES:
        sub = df[(df["case"] == case) & (df["window_tE"] != 3.5)].sort_values("window_tE")
        y = np.abs(sub["fractional_D_change"].to_numpy(float))
        x = sub["window_tE"].to_numpy(float)
        ax.plot(x, y, ms=3.5, label=labels[case], **WINDOW_STYLE[case])
        plotted_alt[case] = y
    ax.set_yscale("log")
    ax.set_ylim(5e-8, 0.1)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.axvline(3.5, color="0.6", lw=LW_THIN, ls=":", zorder=0)
    window_axis(ax)
    ax.set_ylabel(r"$|D(W)/D(3.5\,t_E)-1|$")
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5),
              title=r"$(u_0,\ P/t_E)$", handlelength=1.6)
    save_figure(fig, output_dir / "window_sensitivity_alt")
    plt.close(fig)

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(VERIFY_DIR / "window_sensitivity_plotted_values.csv", index=False, float_format="%.17g")
    print("Saved window_sensitivity(.pdf/.png) and window_sensitivity_alt")


# ============================================================
# Figure 12: Roman true-blending sensitivity
# ============================================================

def plot_true_blending(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    require_columns(df, ["case", "fb_over_fs_true", "delta_chi2",
                         "delta_chi2_over_unblended", "D_roman_w",
                         "Droman_over_unblended", "u0_true", "P_over_tE"], csv_path)

    labels = {c: physical_label(df[df["case"] == c].iloc[0]) for c in BLEND_CASES}

    apply_paper_style()
    fig = plt.figure(figsize=(fig_width(0.88), 2.45))
    ax1 = fig.add_axes([0.105, 0.17, 0.375, 0.80])
    ax2 = fig.add_axes([0.605, 0.17, 0.375, 0.80])

    for ax, col in ((ax1, "delta_chi2_over_unblended"), (ax2, "Droman_over_unblended")):
        ax.axhline(1.0, color="0.6", lw=LW_THIN, ls=":", zorder=0)
        for case in BLEND_CASES:
            sub = df[df["case"] == case].sort_values("fb_over_fs_true")
            ax.plot(sub["fb_over_fs_true"].to_numpy(float), sub[col].to_numpy(float),
                    ms=3.5, label=labels[case], **BLEND_STYLE[case])
        # symlog keeps F_b/F_s = 0 on the axis while separating 0, 0.1 and 0.3
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.6)
        ax.set_xlim(-0.02, 3.6)
        ax.set_xticks([0.0, 0.1, 0.3, 1.0, 3.0])
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(
            ["$0$", "$0.1$", "$0.3$", "$1$", "$3$"]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(r"$F_b/F_s$")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2g$"))

    ax1.set_ylim(0.4, 1.08)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f$"))
    ax2.set_ylim(0.88, 1.008)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f$"))
    ax1.set_ylabel(r"$\Delta\chi^2_{\rm Roman}/\Delta\chi^2_{\rm Roman}(F_b=0)$")
    ax2.set_ylabel(r"$D_{\rm Roman,w}/D_{\rm Roman,w}(F_b=0)$")
    panel_label(ax1, "a", y=0.08).set_va("bottom")
    panel_label(ax2, "b", y=0.08).set_va("bottom")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.14),
               title=r"$(u_0,\ P/t_E)$", handlelength=1.6)

    save_figure(fig, output_dir / "roman_true_blending_sensitivity")
    plt.close(fig)

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(VERIFY_DIR / "roman_true_blending_plotted_values.csv", index=False, float_format="%.17g")
    print("Saved roman_true_blending_sensitivity(.pdf/.png)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", choices=["window", "blending", "all"], default="all")
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    window_csv = results_dir / "window_sensitivity.csv"
    blending_csv = results_dir / "roman_true_blending.csv"

    if args.figure in ("window", "all"):
        if not window_csv.exists():
            raise FileNotFoundError(window_csv)
        plot_window_sensitivity(window_csv, output_dir)

    if args.figure in ("blending", "all"):
        if not blending_csv.exists():
            raise FileNotFoundError(blending_csv)
        plot_true_blending(blending_csv, output_dir)


if __name__ == "__main__":
    main()
