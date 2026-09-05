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

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # ------------------------------------------------------------
    # Panel (a): fresh NM and TRF against stored production D
    # ------------------------------------------------------------
    ax = axes[0]

    ax.scatter(
        d["D_stored"],
        d["D_nm_fresh"],
        s=28,
        alpha=0.75,
        label="fresh Nelder--Mead",
    )

    ax.scatter(
        d["D_stored"],
        d["D_trf_same_objective"],
        s=22,
        alpha=0.65,
        marker="x",
        label="TRF, same objective",
    )

    lo = min(
        d["D_stored"].min(),
        d["D_nm_fresh"].min(),
        d["D_trf_same_objective"].min(),
    )
    hi = max(
        d["D_stored"].max(),
        d["D_nm_fresh"].max(),
        d["D_trf_same_objective"].max(),
    )

    ref = np.logspace(np.log10(lo), np.log10(hi), 300)
    ax.plot(ref, ref, "--", lw=1.0, label="1:1")

    ax.axvline(1e-2, ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"stored $D_{\rm BSPL-PSPL}$")
    ax.set_ylabel(r"independent re-fit $D_{\rm BSPL-PSPL}$")
    ax.set_title("(a) Reproduction of stored minima")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------
    # Panel (b): fractional differences
    # ------------------------------------------------------------
    ax = axes[1]

    y_nm = positive_floor(d["rel_nm_stored"].to_numpy())
    y_trf = positive_floor(d["rel_trf_nm"].to_numpy())

    ax.scatter(
        d["D_stored"],
        y_nm,
        s=28,
        alpha=0.75,
        label=r"$|D_{\rm NM,fresh}-D_{\rm stored}|/D_{\rm stored}$",
    )

    ax.scatter(
        d["D_stored"],
        y_trf,
        s=22,
        alpha=0.65,
        marker="x",
        label=r"$|D_{\rm TRF}-D_{\rm NM,fresh}|/D_{\rm NM,fresh}$",
    )

    ax.axvline(1e-2, ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"stored $D_{\rm BSPL-PSPL}$")
    ax.set_ylabel("absolute fractional difference")
    ax.set_title("(b) Numerical agreement")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
