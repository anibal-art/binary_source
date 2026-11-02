#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


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
OUT_PNG = RESULTS_DIR / "intrinsic_parallax_two_panel_clean.png"
OUT_PDF = RESULTS_DIR / "intrinsic_parallax_two_panel_clean.pdf"


# ============================================================
# Science configuration
# ============================================================

TE_TRUE = 30.0
P_OVER_TE_ONE_YEAR = 365.25 / TE_TRUE

# Color scale for D
VMIN = 1e-5
VMAX = 1e-1

# Keep only a few scientifically useful contours.
LEVELS_MAIN = np.array([1e-2, 1e-3, 1e-4], dtype=float)

# Same levels for the comparison contours of D0 on panel (b)
LEVELS_COMPARE = np.array([1e-2, 1e-3, 1e-4], dtype=float)


# ============================================================
# Helpers
# ============================================================

def load_grid(path):
    df = pd.read_parquet(path)

    need = ["u0_true", "P_over_tE", "D0", "Dpi"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Sort so the grid is monotonic in both axes.
    df = df.sort_values(["P_over_tE", "u0_true"]).reset_index(drop=True)

    u0_vals = np.sort(df["u0_true"].unique())
    pte_vals = np.sort(df["P_over_tE"].unique())

    D0 = (
        df.pivot(index="P_over_tE", columns="u0_true", values="D0")
        .reindex(index=pte_vals, columns=u0_vals)
        .to_numpy(dtype=float)
    )

    Dpi = (
        df.pivot(index="P_over_tE", columns="u0_true", values="Dpi")
        .reindex(index=pte_vals, columns=u0_vals)
        .to_numpy(dtype=float)
    )

    return u0_vals, pte_vals, D0, Dpi


def add_contour_labels(cs, fmt, color="black", fontsize=12):
    texts = cs.clabel(
        fmt=fmt,
        inline=True,
        inline_spacing=6,
        fontsize=fontsize,
        colors=color,
    )
    for txt in texts:
        txt.set_path_effects(
            [pe.withStroke(linewidth=3, foreground="white")]
        )


def setup_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$u_0$", fontsize=16)
    ax.set_ylabel(r"$P/t_E$", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)


# ============================================================
# Main
# ============================================================

def main():
    print("loading:", INPUT)
    u0_vals, pte_vals, D0, Dpi = load_grid(INPUT)

    # Mesh with x = u0 and y = P/tE
    X, Y = np.meshgrid(u0_vals, pte_vals)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cmap = "viridis"
    norm = LogNorm(vmin=VMIN, vmax=VMAX)

    # --------------------------------------------------------
    # Panel (a): D0
    # --------------------------------------------------------
    ax = axes[0]
    im0 = ax.contourf(
        X,
        Y,
        D0,
        levels=np.logspace(np.log10(VMIN), np.log10(VMAX), 30),
        norm=norm,
        cmap=cmap,
        extend="both",
    )

    c0 = ax.contour(
        X,
        Y,
        D0,
        levels=np.sort(LEVELS_MAIN),
        colors="black",
        linewidths=1.8,
    )

    add_contour_labels(
        c0,
        fmt={
            1e-2: r"$10^{-2}$",
            1e-3: r"$10^{-3}$",
            1e-4: r"$10^{-4}$",
        },
        color="black",
        fontsize=12,
    )

    ax.axhline(
        P_OVER_TE_ONE_YEAR,
        color="red",
        linestyle=":",
        linewidth=2.2,
    )

    ax.text(
        7.0,
        P_OVER_TE_ONE_YEAR * 1.04,
        r"$P=1\,{\rm yr}$",
        color="red",
        fontsize=14,
        ha="left",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    ax.set_title(r"(a) intrinsic mismatch $D_0$ (BSPL vs. PSPL)", fontsize=17)
    setup_axes(ax)

    # --------------------------------------------------------
    # Panel (b): Dpi with D0 comparison contours
    # --------------------------------------------------------
    ax = axes[1]
    im1 = ax.contourf(
        X,
        Y,
        Dpi,
        levels=np.logspace(np.log10(VMIN), np.log10(VMAX), 30),
        norm=norm,
        cmap=cmap,
        extend="both",
    )

    # Solid black = Dpi contours
    c1 = ax.contour(
        X,
        Y,
        Dpi,
        levels=np.sort(LEVELS_MAIN),
        colors="black",
        linewidths=1.8,
    )

    add_contour_labels(
        c1,
        fmt={
            1e-2: r"$10^{-2}$",
            1e-3: r"$10^{-3}$",
            1e-4: r"$10^{-4}$",
        },
        color="black",
        fontsize=12,
    )

    # Dashed white = D0 contours, same levels, only as comparison
    c2 = ax.contour(
        X,
        Y,
        D0,
        levels=np.sort(LEVELS_COMPARE),
        colors="white",
        linestyles="--",
        linewidths=2.0,
    )

    add_contour_labels(
        c2,
        fmt={
            1e-2: r"$10^{-2}$",
            1e-3: r"$10^{-3}$",
            1e-4: r"$10^{-4}$",
        },
        color="white",
        fontsize=12,
    )

    ax.axhline(
        P_OVER_TE_ONE_YEAR,
        color="red",
        linestyle=":",
        linewidth=2.2,
    )

    ax.text(
        7.0,
        P_OVER_TE_ONE_YEAR * 1.04,
        r"$P=1\,{\rm yr}$",
        color="red",
        fontsize=14,
        ha="left",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    ax.set_title(r"(b) intrinsic mismatch $D_\pi$ (BSPL vs. PSPL+parallax)", fontsize=17)
    setup_axes(ax)
    ax.set_ylabel("")

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------
    handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", label=r"solid: $D_\pi$ contours"),
        Line2D([0], [0], color="white", lw=2.0, linestyle="--", label=r"dashed: $D_0$ contours"),
        Line2D([0], [0], color="red", lw=2.2, linestyle=":", label=r"$P=1\,{\rm yr}$"),
    ]

    axes[1].legend(
        handles=handles,
        loc="lower left",
        fontsize=12,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    # --------------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------------
    cbar = fig.colorbar(
        im1,
        ax=axes,
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label(r"intrinsic mismatch $D$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------
    fig.suptitle(
        "Intrinsic BSPL/xallarap mismatch before and after allowing annual parallax\n"
        + r"$t_E = 30\,{\rm d},\ \pi_{E,{\rm true}} = 0$",
        fontsize=20,
    )

    # Optional explicit limits, to match the full ultrafine grid
    axes[0].set_xlim(u0_vals.min(), u0_vals.max())
    axes[0].set_ylim(pte_vals.min(), pte_vals.max())

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print("saved:", OUT_PNG)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
