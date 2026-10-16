#!/usr/bin/env python3
"""
Plot the 60-point stratified optimizer/refit audit used in Appendix B.

Run from the repository root:

    python binary_source/plot_codes/plot_stratified_refit_audit.py

Input
-----
results/validation_optimizer_grid_audit/stratified_refit_audit.csv

Output
------
figures/appendix/optimizer_stratified_refit_audit.pdf
figures/appendix/optimizer_stratified_refit_audit.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (LogLocator, FuncFormatter, FixedLocator,
                               FixedFormatter, NullLocator)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (apply_paper_style, fig_width, panel_label,
                         save_figure, C_BLUE, C_VERMILION, LW_THIN)


ROOT = Path.cwd()
CSV = ROOT / "results" / "validation_optimizer_grid_audit" / "stratified_refit_audit.csv"
OUTDIR = ROOT / "figures" / "appendix"
OUTDIR.mkdir(parents=True, exist_ok=True)

PDF = OUTDIR / "optimizer_stratified_refit_audit.pdf"
PNG = OUTDIR / "optimizer_stratified_refit_audit.png"


def require_column(df, name):
    if name not in df.columns:
        raise KeyError(
            f"Required column {name!r} not found. Available columns: "
            + ", ".join(df.columns)
        )


def positive_floor(x, floor=1e-16):
    x = np.asarray(x, dtype=float)
    return np.maximum(np.abs(x), floor)


def main():
    if not CSV.exists():
        raise FileNotFoundError(
            f"Missing {CSV}. This figure uses the already-computed "
            "60-point stratified re-fit audit."
        )

    df = pd.read_csv(CSV)

    for col in ["D_stored", "D_nm_fresh", "D_trf_same_objective"]:
        require_column(df, col)

    good = (
        np.isfinite(df["D_stored"])
        & np.isfinite(df["D_nm_fresh"])
        & np.isfinite(df["D_trf_same_objective"])
        & (df["D_stored"] > 0)
        & (df["D_nm_fresh"] > 0)
        & (df["D_trf_same_objective"] > 0)
    )
    d = df.loc[good].copy()

    if len(d) == 0:
        raise RuntimeError("No finite positive audit points found.")

    # Recompute the two fractional comparisons so the plot does not depend
    # on optional pre-computed CSV columns.
    d["rel_nm_stored"] = (
        d["D_nm_fresh"] - d["D_stored"]
    ) / d["D_stored"]

    d["rel_trf_nm"] = (
        d["D_trf_same_objective"] - d["D_nm_fresh"]
    ) / d["D_nm_fresh"]

    apply_paper_style()
    fig = plt.figure(figsize=(fig_width(0.92), 2.55))
    ax_a = fig.add_axes([0.10, 0.165, 0.37, 0.80])
    ax_b = fig.add_axes([0.605, 0.165, 0.37, 0.80])

    nm_style = dict(marker="o", s=12, facecolors="none", edgecolors=C_BLUE,
                    linewidths=0.7, zorder=3)
    trf_style = dict(marker="x", s=12, color=C_VERMILION, linewidths=0.7,
                     zorder=4)

    # ------------------------------------------------------------
    # Panel (a): fresh NM and TRF against stored production D
    # ------------------------------------------------------------
    ax = ax_a
    lo = min(d["D_stored"].min(), d["D_nm_fresh"].min(),
             d["D_trf_same_objective"].min())
    hi = max(d["D_stored"].max(), d["D_nm_fresh"].max(),
             d["D_trf_same_objective"].max())
    ref = np.logspace(np.log10(lo) - 0.3, np.log10(hi) + 0.3, 300)
    ax.plot(ref, ref, color="0.4", ls="--", lw=LW_THIN, zorder=1)
    ax.scatter(d["D_stored"], d["D_nm_fresh"], label="fresh Nelder--Mead", **nm_style)
    ax.scatter(d["D_stored"], d["D_trf_same_objective"], label="TRF", **trf_style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ref[0], ref[-1])
    ax.set_ylim(ref[0], ref[-1])
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(
            lambda v, _p: rf"$10^{{{int(round(np.log10(v)))}}}$"
            if int(round(np.log10(v))) % 2 == 0 else ""))
    ax.set_xlabel(r"stored $D_{\rm BSPL-PSPL}$")
    ax.set_ylabel(r"re-fit $D_{\rm BSPL-PSPL}$")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    # ------------------------------------------------------------
    # Panel (b): absolute fractional differences; exact zeros are
    # drawn in a separate band below the numerical floor 1e-16.
    # ------------------------------------------------------------
    ax = ax_b
    FLOOR = 1e-16
    ZERO_Y = 10 ** -17.0
    y_nm = np.abs(d["rel_nm_stored"].to_numpy())
    y_trf = np.abs(d["rel_trf_nm"].to_numpy())
    x = d["D_stored"].to_numpy()

    def show(y, style, label):
        zero = y == 0.0
        nz = ~zero
        if np.any(nz & (y < FLOOR)):
            raise RuntimeError("non-zero difference below the plotted floor")
        ax.scatter(x[nz], y[nz], label=label, **style)
        ax.scatter(x[zero], np.full(zero.sum(), ZERO_Y), **style)
        return int(zero.sum())

    n0_nm = show(y_nm, nm_style, r"fresh NM vs.\ stored")
    n0_trf = show(y_trf, trf_style, r"TRF vs.\ fresh NM")
    print(f"exact zeros: NM-stored {n0_nm}, TRF-NM {n0_trf}")

    ax.axhspan(10 ** -17.5, 10 ** -16.5, color="0.92", lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(10 ** -17.5, 1e-3)
    ax.set_xlim(ref[0], ref[-1])
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _p: rf"$10^{{{int(round(np.log10(v)))}}}$"
        if int(round(np.log10(v))) % 2 == 0 else ""))
    yt = [ZERO_Y, 1e-16, 1e-13, 1e-10, 1e-7, 1e-4]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter(
        ["$0$"] + [rf"$10^{{{int(np.log10(v))}}}$" for v in yt[1:]]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"stored $D_{\rm BSPL-PSPL}$")
    ax.set_ylabel(r"$|\Delta D|/D$")
    ax.legend(loc="upper right", ncol=1)
    panel_label(ax, "b")

    save_figure(fig, PDF.with_suffix(""))
    plt.close(fig)

    np.savez(ROOT / "figures" / "_verify" / "optimizer_stratified_refit_audit_plotted_values.npz",
             D_stored=x, D_nm=d["D_nm_fresh"].to_numpy(),
             D_trf=d["D_trf_same_objective"].to_numpy(), y_nm=y_nm, y_trf=y_trf)

    print(f"N plotted = {len(d)}")
    print(
        "max |NM fresh - stored| / stored =",
        np.max(np.abs(d["rel_nm_stored"])),
    )
    print(
        "max |TRF - NM fresh| / NM fresh =",
        np.max(np.abs(d["rel_trf_nm"])),
    )
    print("Saved:", PDF)
    print("Saved:", PNG)


if __name__ == "__main__":
    main()
