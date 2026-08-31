#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import (
    LogNorm,
    TwoSlopeNorm,
)


SCRIPT = Path(__file__).resolve()

PLOT_DIR = SCRIPT.parent
SOURCE_DIR = PLOT_DIR.parent
REPO_ROOT = SOURCE_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"

if str(PLOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PLOT_DIR),
    )

from paper_style import apply_paper_style


FEATURE_QM_MIN = 3e-2
FEATURE_QM_MAX = 3e-1


def find_summary():

    candidates = list(
        RESULTS_ROOT.glob(
            "diagnostic_u0_qmass_PoverTE1_qf0_*/"
            "summary_u0_qmass_fixed_period.npz"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            "No u0-qM diagnostic summary found."
        )

    return max(
        candidates,
        key=lambda p:
            p.stat().st_mtime,
    )


def log_limits(
    values,
):

    valid = np.asarray(
        values,
        dtype=float,
    )

    valid = valid[
        np.isfinite(valid)
        & (valid > 0.0)
    ]

    return (
        10.0 ** np.floor(
            np.log10(
                np.percentile(
                    valid,
                    1
                )
            )
        ),
        10.0 ** np.ceil(
            np.log10(
                np.percentile(
                    valid,
                    99.5
                )
            )
        ),
    )


def symmetric_limit(
    values,
):

    valid = np.abs(
        np.asarray(
            values,
            dtype=float,
        )
    )

    valid = valid[
        np.isfinite(valid)
    ]

    if len(valid) == 0:
        return 1.0

    value = np.percentile(
        valid,
        99.0,
    )

    if value <= 0.0:
        value = 1.0

    return float(
        value
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=Path,
    default=None,
)

args = parser.parse_args()


filename = (
    args.input.expanduser().resolve()
    if args.input is not None
    else find_summary().resolve()
)


with np.load(
    filename,
    allow_pickle=False,
) as d:

    u0 = np.asarray(
        d["u0_grid"],
        dtype=float,
    )

    qM = np.asarray(
        d["qM_grid"],
        dtype=float,
    )

    D = np.asarray(
        d["D"],
        dtype=float,
    )

    U1MIN = np.asarray(
        d["U1MIN"],
        dtype=float,
    )

    DTMIN1 = np.asarray(
        d["DT_U1MIN_OVER_TE"],
        dtype=float,
    )

    DU0 = np.asarray(
        d["DU0"],
        dtype=float,
    )

    XI1_U0 = np.asarray(
        d["xi1_over_u0"],
        dtype=float,
    )

    SUCCESS = np.asarray(
        d["SUCCESS"],
        dtype=bool,
    )

    P_over_tE = float(
        d["P_over_tE"].item()
    )

    qf = float(
        d["qf"].item()
    )

    commit = str(
        d["code_commit"].item()
    )


DU0_REL = (
    DU0
    / u0[:, None]
)


# ============================================================
# Local feature ridge and closest-approach locus
# ============================================================

feature_q_mask = (
    (qM >= FEATURE_QM_MIN)
    & (qM <= FEATURE_QM_MAX)
)


q_Dmax = np.full(
    len(u0),
    np.nan,
)

q_U1min = np.full(
    len(u0),
    np.nan,
)


for i_u in range(
    len(u0)
):

    good = (
        feature_q_mask
        & SUCCESS[i_u]
        & np.isfinite(
            D[i_u]
        )
        & np.isfinite(
            U1MIN[i_u]
        )
    )


    idx = np.where(
        good
    )[0]


    if len(idx) == 0:
        continue


    q_Dmax[i_u] = qM[
        idx[
            np.argmax(
                D[
                    i_u,
                    idx,
                ]
            )
        ]
    ]


    q_U1min[i_u] = qM[
        idx[
            np.argmin(
                U1MIN[
                    i_u,
                    idx,
                ]
            )
        ]
    ]


# ============================================================
# Plot
# ============================================================

apply_paper_style()


fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        10.5,
        8.4,
    ),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)


axes = axes.ravel()


# ============================================================
# A. D
# ============================================================

D_plot = np.where(
    SUCCESS
    & np.isfinite(D)
    & (D > 0.0),
    D,
    np.nan,
)


vmin, vmax = log_limits(
    D_plot
)


pcm = axes[0].pcolormesh(
    qM,
    u0,
    D_plot,
    shading="auto",
    cmap="viridis",
    norm=LogNorm(
        vmin=vmin,
        vmax=vmax,
    ),
    rasterized=True,
)


cb = fig.colorbar(
    pcm,
    ax=axes[0],
    pad=0.02,
)

cb.set_label(
    r"$D_{\rm BSPL-PSPL}$"
)


# Orbital amplitude of the luminous source.
levels_xi1 = [
    0.01,
    0.1,
    1.0,
]


finite_xi = XI1_U0[
    np.isfinite(
        XI1_U0
    )
]


levels_here = [
    level
    for level in levels_xi1
    if (
        np.min(
            finite_xi
        )
        <= level
        <= np.max(
            finite_xi
        )
    )
]


if levels_here:

    axes[0].contour(
        qM,
        u0,
        XI1_U0,
        levels=levels_here,
        colors="white",
        linewidths=0.8,
        alpha=0.75,
    )


axes[0].plot(
    q_Dmax,
    u0,
    linewidth=1.8,
    label=r"local max of $D$",
)


axes[0].plot(
    q_U1min,
    u0,
    linestyle="--",
    linewidth=1.8,
    label=r"min of $u_{1,\min}$",
)


axes[0].legend(
    frameon=False,
    fontsize=9,
)


axes[0].set_title(
    r"$D_{\rm BSPL-PSPL}$"
)


# ============================================================
# B. closest approach of luminous source
# ============================================================

u1_vmin, u1_vmax = log_limits(
    U1MIN
)


pcm = axes[1].pcolormesh(
    qM,
    u0,
    U1MIN,
    shading="auto",
    cmap="viridis",
    norm=LogNorm(
        vmin=u1_vmin,
        vmax=u1_vmax,
    ),
    rasterized=True,
)


cb = fig.colorbar(
    pcm,
    ax=axes[1],
    pad=0.02,
)

cb.set_label(
    r"$u_{1,\min}$"
)


axes[1].plot(
    q_Dmax,
    u0,
    linewidth=1.8,
)


axes[1].plot(
    q_U1min,
    u0,
    linestyle="--",
    linewidth=1.8,
)


axes[1].set_title(
    r"Closest approach of luminous source"
)


# ============================================================
# C. time of closest approach
# ============================================================

limit_t = symmetric_limit(
    DTMIN1
)


pcm = axes[2].pcolormesh(
    qM,
    u0,
    DTMIN1,
    shading="auto",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-limit_t,
        vcenter=0.0,
        vmax=limit_t,
    ),
    rasterized=True,
)


cb = fig.colorbar(
    pcm,
    ax=axes[2],
    pad=0.02,
)

cb.set_label(
    r"$(t_{u_1,\min}-t_0)/t_E$"
)


axes[2].set_title(
    r"Time of closest approach"
)


# ============================================================
# D. PSPL u0 bias
# ============================================================

limit_du0 = symmetric_limit(
    DU0_REL
)


pcm = axes[3].pcolormesh(
    qM,
    u0,
    DU0_REL,
    shading="auto",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-limit_du0,
        vcenter=0.0,
        vmax=limit_du0,
    ),
    rasterized=True,
)


cb = fig.colorbar(
    pcm,
    ax=axes[3],
    pad=0.02,
)

cb.set_label(
    r"$\Delta u_0/u_0$"
)


axes[3].set_title(
    r"PSPL absorption"
)


# ============================================================
# Common formatting
# ============================================================

for i, ax in enumerate(
    axes
):

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        qM.min(),
        qM.max(),
    )

    ax.set_ylim(
        u0.min(),
        u0.max(),
    )

    ax.text(
        0.05,
        0.94,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


axes[2].set_xlabel(
    r"$q_M$"
)

axes[3].set_xlabel(
    r"$q_M$"
)

axes[0].set_ylabel(
    r"$u_0$"
)

axes[2].set_ylabel(
    r"$u_0$"
)


fig.suptitle(
    rf"$P/t_E={P_over_tE:g}$, "
    rf"$q_f={qf:g}$",
)


# ============================================================
# Save
# ============================================================

outdir = (
    REPO_ROOT
    / "figures"
    / f"draft_{commit}"
)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)


png = (
    outdir
    / "u0_qmass_fixed_period_diagnostic.png"
)

pdf = (
    outdir
    / "u0_qmass_fixed_period_diagnostic.pdf"
)


fig.savefig(
    png,
    dpi=600,
)

fig.savefig(
    pdf,
)

plt.close(
    fig
)


print(
    "Input:",
    filename,
)

print(
    "PNG  :",
    png,
)

print(
    "PDF  :",
    pdf,
)
