#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    REPO_ROOT
    / "results"
    / "intrinsic_xallarap_vs_pspl_parallax_tE30_ultrafine_u0_10"
)

INPUT = RESULTS_DIR / "intrinsic_parallax_grid.parquet"

OUT_PNG = RESULTS_DIR / "intrinsic_parallax_two_panel_ultrafine.png"
OUT_PDF = RESULTS_DIR / "intrinsic_parallax_two_panel_ultrafine.pdf"


# ============================================================
# Helpers
# ============================================================

def build_grid(df, value_col):
    u0 = np.sort(df["u0_true"].unique())
    pte = np.sort(df["P_over_tE"].unique())

    pivot = (
        df.pivot(
            index="P_over_tE",
            columns="u0_true",
            values=value_col,
        )
        .reindex(index=pte, columns=u0)
    )

    z = pivot.to_numpy(dtype=float)
    x, y = np.meshgrid(u0, pte)

    return u0, pte, x, y, z


def nice_log_levels(z0, z1):
    allz = np.concatenate(
        [
            z0[np.isfinite(z0)],
            z1[np.isfinite(z1)],
        ]
    )

    vmin_raw = np.nanmin(allz)
    vmax_raw = np.nanmax(allz)

    vmin = np.floor(vmin_raw * 2.0) / 2.0
    vmax = np.ceil(vmax_raw * 2.0) / 2.0

    step = 0.5
    levels = np.arange(vmin, vmax + 0.5 * step, step)

    if len(levels) < 5:
        step = 0.25
        levels = np.arange(vmin, vmax + 0.5 * step, step)

    filled = np.linspace(vmin, vmax, 256)

    return vmin, vmax, levels, filled


def apply_common_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$u_0$")
    ax.set_ylabel(r"$P/t_E$")
    ax.grid(False)

    ax.set_xlim(1e-2, 1e1)
    ax.set_ylim(1/3, 3333.3333333333335)


def label_contours(ax, contour_set, fmt_func):
    labels = ax.clabel(
        contour_set,
        contour_set.levels,
        inline=True,
        fmt=fmt_func,
        fontsize=9,
        colors="black",
    )
    for txt in labels:
        txt.set_path_effects(
            [
                pe.Stroke(linewidth=3.0, foreground="white"),
                pe.Normal(),
            ]
        )


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_parquet(INPUT)

    required = [
        "u0_true",
        "P_over_tE",
        "D0",
        "Dpi",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    if len(df) != 601 * 241:
        print(
            "WARNING: unexpected number of rows:",
            len(df),
            "(expected 144841)"
        )

    u0, pte, X, Y, D0 = build_grid(df, "D0")
    _, _, _, _, Dpi = build_grid(df, "Dpi")

    if np.any(~np.isfinite(D0)):
        raise RuntimeError("D0 has non-finite values")
    if np.any(~np.isfinite(Dpi)):
        raise RuntimeError("Dpi has non-finite values")

    logD0 = np.log10(D0)
    logDpi = np.log10(Dpi)

    vmin, vmax, line_levels, filled_levels = nice_log_levels(logD0, logDpi)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    cmap = "viridis"

    # --------------------------------------------------------
    # Panel 1: D0
    # --------------------------------------------------------
    ax = axes[0]

    cf0 = ax.contourf(
        X,
        Y,
        logD0,
        levels=filled_levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )

    c0 = ax.contour(
        X,
        Y,
        logD0,
        levels=line_levels,
        colors="black",
        linewidths=0.9,
        linestyles="solid",
    )

    label_contours(
        ax,
        c0,
        fmt_func=lambda lev: rf"$10^{{{lev:.1f}}}$",
    )

    apply_common_axes(ax)
    ax.set_title(r"(a) intrinsic mismatch $D_0$ (BSPL vs PSPL)")

    # --------------------------------------------------------
    # Panel 2: Dpi
    # --------------------------------------------------------
    ax = axes[1]

    cf1 = ax.contourf(
        X,
        Y,
        logDpi,
        levels=filled_levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )

    c1 = ax.contour(
        X,
        Y,
        logDpi,
        levels=line_levels,
        colors="black",
        linewidths=0.9,
        linestyles="solid",
    )

    label_contours(
        ax,
        c1,
        fmt_func=lambda lev: rf"$10^{{{lev:.1f}}}$",
    )

    # Reference D0 contours on the right panel
    c1_ref = ax.contour(
        X,
        Y,
        logD0,
        levels=line_levels,
        colors="white",
        linewidths=1.0,
        linestyles="dashed",
        alpha=0.95,
    )

    for coll in c1_ref.collections:
        coll.set_path_effects(
            [
                pe.Stroke(linewidth=2.2, foreground="black"),
                pe.Normal(),
            ]
        )

    apply_common_axes(ax)
    ax.set_title(r"(b) intrinsic mismatch $D_\pi$ (BSPL vs PSPL+parallax)")

    # Small legend only for the overlay convention
    ax.plot(
        [],
        [],
        color="black",
        lw=1.0,
        ls="solid",
        label=r"solid: $D_\pi$ contours",
    )
    ax.plot(
        [],
        [],
        color="white",
        lw=1.2,
        ls="dashed",
        label=r"dashed: $D_0$ contours",
        path_effects=[
            pe.Stroke(linewidth=2.2, foreground="black"),
            pe.Normal(),
        ],
    )
    ax.legend(
        loc="lower left",
        fontsize=9,
        frameon=True,
    )

    # --------------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------------
    cbar = fig.colorbar(
        cf1,
        ax=axes,
        shrink=0.96,
        pad=0.02,
    )
    cbar.set_label(r"$\log_{10} D$")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("loaded:", INPUT)
    print("rows:", len(df))
    print("u0 nodes:", len(u0))
    print("P/tE nodes:", len(pte))
    print("common log10(D) range:", (vmin, vmax))
    print("contour levels:", line_levels.tolist())
    print("saved:", OUT_PNG)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
