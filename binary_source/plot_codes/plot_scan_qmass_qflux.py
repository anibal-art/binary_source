#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot joint q_M -- q_f scan.

Main figure
-----------
D_BSPL-PSPL(q_M, q_f) for each P/tE.

The line

    q_f = q_M

is highlighted because the first-order photocenter displacement

    xi_phot =
        |q_M - q_f| /
        [(1 + q_M)(1 + q_f)] * xi_rel

vanishes there.

Diagnostic figure
-----------------
Compare:

    D(q_f = 0)

with

    D(q_f = q_M)

to quantify how the luminosity of the companion changes the
BSPL--PSPL degeneracy.
"""

# ============================================================
# Imports
# ============================================================

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe


# ============================================================
# Matplotlib style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",

        "font.size": 12,

        "axes.labelsize": 13,
        "axes.titlesize": 13,

        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "legend.fontsize": 10,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "axes.linewidth": 1.0,

        "savefig.bbox": "tight",
    }
)


# ============================================================
# Input
# ============================================================

home = Path.home()

results_dir = (
    home
    / "binary_source"
    / "results"
    / "scan_qM_qf_Mtotfixed_tE150"
)

input_file = (
    results_dir
    / "summary_qM_qf.npz"
)


# ============================================================
# Output
# ============================================================

figure_dir = (
    home
    / "binary_source"
    / "figures"
    / "figures_qM_qf"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Plot configuration
# ============================================================

# Reference mismatch contour
D_REFERENCE = 1e-2

# ------------------------------------------------------------
# Color normalization
#
# Set to None for automatic limits.
#
# If later you want exactly the same scale in several paper
# figures, replace these by fixed values.
# ------------------------------------------------------------

VMIN = None
VMAX = None


# ------------------------------------------------------------
# Optional contour levels
# ------------------------------------------------------------

SHOW_EXTRA_CONTOURS = False

EXTRA_CONTOURS = [
    1e-3,
    1e-2,
    1e-1,
]


# ============================================================
# Load
# ============================================================

if not input_file.exists():

    raise FileNotFoundError(
        f"No encontré:\n{input_file}"
    )


d = np.load(
    input_file,
    allow_pickle=False,
)


print()
print("=" * 70)
print("LOADING q_M -- q_f SCAN")
print("=" * 70)

print(
    "File:",
    input_file,
)

print(
    "Keys:",
    d.files,
)


# ============================================================
# Read grids
# ============================================================

qM_grid = np.asarray(
    d["qM_grid"],
    dtype=float,
)

qf_grid = np.asarray(
    d["qf_grid"],
    dtype=float,
)

P_grid = np.asarray(
    d["P_grid"],
    dtype=float,
)

P_over_tE_grid = np.asarray(
    d["P_over_tE_grid"],
    dtype=float,
)

D_cube = np.asarray(
    d["D"],
    dtype=float,
)

SUCCESS = np.asarray(
    d["SUCCESS"],
    dtype=bool,
)


# ============================================================
# Physical parameters
# ============================================================

Mtot_source = float(
    d["Mtot_source"]
)

u0_true = float(
    d["u0_true"]
)

tE_true = float(
    d["tE_true"]
)

rEhat_AU = float(
    d["rEhat_AU"]
)


# ============================================================
# Check dimensions
# ============================================================

expected_shape = (
    len(qM_grid),
    len(qf_grid),
    len(P_grid),
)

if D_cube.shape != expected_shape:

    raise ValueError(
        "\nShape inconsistente.\n"
        f"D.shape      = {D_cube.shape}\n"
        f"expected     = {expected_shape}"
    )


print()
print(
    "D shape      =",
    D_cube.shape,
)

print(
    "qM range     =",
    qM_grid.min(),
    qM_grid.max(),
)

print(
    "qf range     =",
    qf_grid.min(),
    qf_grid.max(),
)

print(
    "P/tE         =",
    P_over_tE_grid,
)

print()


# ============================================================
# Mask failed simulations
# ============================================================

D_plot_cube = D_cube.copy()

D_plot_cube[
    ~SUCCESS
] = np.nan

D_plot_cube[
    D_plot_cube <= 0
] = np.nan


# ============================================================
# Determine common color normalization
# ============================================================

valid_D = D_plot_cube[
    np.isfinite(
        D_plot_cube
    )
]

if len(valid_D) == 0:

    raise RuntimeError(
        "No hay valores válidos de D."
    )


# ------------------------------------------------------------
# Robust automatic limits
# ------------------------------------------------------------

if VMIN is None:

    low = np.nanpercentile(
        valid_D,
        1.0,
    )

    VMIN = 10.0 ** np.floor(
        np.log10(low)
    )


if VMAX is None:

    high = np.nanpercentile(
        valid_D,
        99.0,
    )

    VMAX = 10.0 ** np.ceil(
        np.log10(high)
    )


# Safety
if VMAX <= VMIN:

    VMAX = 10.0 * VMIN


norm = colors.LogNorm(
    vmin=VMIN,
    vmax=VMAX,
)


print(
    f"Color scale: "
    f"{VMIN:.3e} -- {VMAX:.3e}"
)


# ============================================================
# q_f axis
#
# q_f contains zero, therefore a standard log axis cannot be
# used. symlog preserves q_f=0 while becoming logarithmic above
# the smallest positive q_f.
# ============================================================

qf_positive = qf_grid[
    qf_grid > 0
]

if len(qf_positive) == 0:

    raise RuntimeError(
        "q_f grid no contiene valores positivos."
    )

qf_min_positive = np.min(
    qf_positive
)


# ============================================================
# Mesh
# ============================================================

QM, QF = np.meshgrid(
    qM_grid,
    qf_grid,
    indexing="xy",
)


# ============================================================
# MAIN FIGURE
# ============================================================

N_P = len(
    P_over_tE_grid
)

fig_width = (
    4.2 * N_P
)

fig, axes = plt.subplots(
    1,
    N_P,
    figsize=(
        fig_width,
        4.5,
    ),
    sharex=True,
    sharey=True,
)

if N_P == 1:

    axes = np.array(
        [axes]
    )


# ============================================================
# Panels
# ============================================================

panel_labels = [
    f"({chr(ord('a') + i)})"
    for i in range(N_P)
]

mappable = None


for k, ax in enumerate(axes):

    # --------------------------------------------------------
    # D for this P
    #
    # D cube is:
    #
    #   [qM, qf, P]
    #
    # pcolormesh wants:
    #
    #   [qf, qM]
    # --------------------------------------------------------

    Z = (
        D_plot_cube[
            :,
            :,
            k,
        ].T
    )


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    mappable = ax.pcolormesh(
        qM_grid,
        qf_grid,
        Z,
        shading="nearest",
        cmap="viridis",
        norm=norm,
        rasterized=True,
    )


    # --------------------------------------------------------
    # Reference D contour
    # --------------------------------------------------------

    finite_Z = Z[
        np.isfinite(Z)
    ]

    if len(finite_Z) > 0:

        zmin = np.nanmin(
            finite_Z
        )

        zmax = np.nanmax(
            finite_Z
        )

        if (
            zmin
            <= D_REFERENCE
            <= zmax
        ):

            cs = ax.contour(
                qM_grid,
                qf_grid,
                Z,
                levels=[
                    D_REFERENCE
                ],
                colors="red",
                linewidths=1.8,
            )

            # Optional direct label
            labels = ax.clabel(
                cs,
                fmt={
                    D_REFERENCE:
                    r"$D=10^{-2}$"
                },
                fontsize=9,
                inline=False,
            )

            for txt in labels:

                txt.set_path_effects(
                    [
                        pe.Stroke(
                            linewidth=2.5,
                            foreground="white",
                        ),
                        pe.Normal(),
                    ]
                )


    # --------------------------------------------------------
    # Optional other D contours
    # --------------------------------------------------------

    if SHOW_EXTRA_CONTOURS:

        available_levels = [
            level
            for level in EXTRA_CONTOURS
            if (
                len(finite_Z) > 0
                and
                np.nanmin(finite_Z)
                <= level
                <= np.nanmax(finite_Z)
            )
        ]

        if len(
            available_levels
        ) > 0:

            ax.contour(
                qM_grid,
                qf_grid,
                Z,
                levels=available_levels,
                colors="white",
                linewidths=0.8,
                alpha=0.8,
            )


    # --------------------------------------------------------
    # q_f = q_M
    #
    # First-order photocenter cancellation line.
    # --------------------------------------------------------

    ax.plot(
        qM_grid,
        qM_grid,
        linestyle="--",
        linewidth=1.7,
        color="white",
        zorder=20,
    )


    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    text = ax.text(
        0.04,
        0.95,
        (
            panel_labels[k]
            + "\n"
            + rf"$P/t_E={P_over_tE_grid[k]:g}$"
            + "\n"
            + rf"$P={P_grid[k]:.0f}\,$d"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    text.set_path_effects(
        [
            pe.Stroke(
                linewidth=3,
                foreground="white",
            ),
            pe.Normal(),
        ]
    )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "symlog",
        linthresh=qf_min_positive,
        linscale=1.0,
        base=10,
    )

    ax.set_xlim(
        qM_grid.min(),
        qM_grid.max(),
    )

    ax.set_ylim(
        0,
        qf_grid.max(),
    )

    ax.set_xlabel(
        r"Source mass ratio "
        r"$q_M=M_{S,2}/M_{S,1}$"
    )

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )


# ============================================================
# y label
# ============================================================

axes[0].set_ylabel(
    r"Source flux ratio "
    r"$q_{f,\xi}=F_{S,2}/F_{S,1}$"
)


# ============================================================
# y ticks including zero
# ============================================================

candidate_yticks = np.array(
    [
        0.0,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ]
)

yticks = candidate_yticks[
    candidate_yticks
    <= qf_grid.max()
]

axes[0].set_yticks(
    yticks
)

ytick_labels = []

for value in yticks:

    if value == 0:

        ytick_labels.append(
            r"$0$"
        )

    else:

        exponent = int(
            np.log10(value)
        )

        ytick_labels.append(
            rf"$10^{{{exponent}}}$"
        )

axes[0].set_yticklabels(
    ytick_labels
)


# ============================================================
# Shared colorbar
# ============================================================

cbar = fig.colorbar(
    mappable,
    ax=axes,
    orientation="horizontal",
    location="top",
    fraction=0.07,
    pad=0.08,
    aspect=40,
)

cbar.set_label(
    r"Normalized BSPL--PSPL mismatch "
    r"$D_{\rm BSPL-PSPL}$"
)

cbar.ax.tick_params(
    direction="in",
)


# ============================================================
# Figure-level annotation
# ============================================================

fig.text(
    0.5,
    0.015,
    (
        rf"$M_{{S,\rm tot}}={Mtot_source:g}\,M_\odot$, "
        rf"$u_0={u0_true:g}$, "
        rf"$t_E={tE_true:g}\,$d, "
        rf"$\hat r_E={rEhat_AU:g}\,$AU"
    ),
    ha="center",
    va="bottom",
    fontsize=11,
)


# ============================================================
# Save main figure
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.17,
    top=0.78,
    wspace=0.08,
)

png_file = (
    figure_dir
    / "D_qM_qf_three_periods.png"
)

pdf_file = (
    figure_dir
    / "D_qM_qf_three_periods.pdf"
)

fig.savefig(
    png_file,
    dpi=600,
)

fig.savefig(
    pdf_file,
)

print()
print(
    "Saved:",
    png_file,
)

print(
    "Saved:",
    pdf_file,
)

plt.show()


# ============================================================
# SECOND FIGURE
#
# Compare dark-companion limit q_f=0 with q_f=q_M.
# ============================================================

# ------------------------------------------------------------
# q_f = 0 is explicitly first element in our scan
# ------------------------------------------------------------

i_qf_zero = int(
    np.argmin(
        np.abs(qf_grid)
    )
)


# ------------------------------------------------------------
# Find q_f index corresponding to each q_M.
#
# This works even if the grids are later changed.
# ------------------------------------------------------------

diag_indices = np.array(
    [
        np.argmin(
            np.abs(
                qf_grid - qM
            )
        )
        for qM in qM_grid
    ],
    dtype=int,
)


# ============================================================
# Extract D
# ============================================================

D_dark = np.full(
    (
        len(qM_grid),
        N_P,
    ),
    np.nan,
)

D_diag = np.full_like(
    D_dark,
    np.nan,
)


for i_qM in range(
    len(qM_grid)
):

    for k in range(
        N_P
    ):

        D_dark[
            i_qM,
            k,
        ] = D_plot_cube[
            i_qM,
            i_qf_zero,
            k,
        ]

        D_diag[
            i_qM,
            k,
        ] = D_plot_cube[
            i_qM,
            diag_indices[i_qM],
            k,
        ]


# ============================================================
# Ratio
#
# < 1:
#     making source 2 luminous makes the event MORE degenerate.
#
# > 1:
#     making source 2 luminous BREAKS the degeneracy.
# ============================================================

ratio = (
    D_diag
    / D_dark
)


# ============================================================
# Plot diagnostic
# ============================================================

fig2, axes2 = plt.subplots(
    1,
    2,
    figsize=(
        9.0,
        4.0,
    ),
)


# ============================================================
# Left: actual D values
# ============================================================

ax = axes2[0]

for k in range(
    N_P
):

    line, = ax.plot(
        qM_grid,
        D_dark[
            :,
            k,
        ],
        linestyle="--",
        linewidth=1.5,
        label=(
            rf"$P/t_E={P_over_tE_grid[k]:g}$,"
            r" $q_f=0$"
        ),
    )

    ax.plot(
        qM_grid,
        D_diag[
            :,
            k,
        ],
        linestyle="-",
        linewidth=2.0,
        color=line.get_color(),
        label=(
            rf"$P/t_E={P_over_tE_grid[k]:g}$,"
            r" $q_f=q_M$"
        ),
    )


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$q_M$"
)

ax.set_ylabel(
    r"$D_{\rm BSPL-PSPL}$"
)

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
)

ax.legend(
    frameon=False,
    fontsize=8,
    ncol=1,
)

ax.text(
    0.04,
    0.95,
    "(a)",
    transform=ax.transAxes,
    ha="left",
    va="top",
)


# ============================================================
# Right: ratio D(qf=qM) / D(qf=0)
# ============================================================

ax = axes2[1]

for k in range(
    N_P
):

    ax.plot(
        qM_grid,
        ratio[
            :,
            k,
        ],
        linewidth=2.0,
        label=(
            rf"$P/t_E={P_over_tE_grid[k]:g}$"
        ),
    )


ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$q_M$"
)

ax.set_ylabel(
    r"$D(q_f=q_M)/D(q_f=0)$"
)

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
)

ax.legend(
    frameon=False,
)

ax.text(
    0.04,
    0.95,
    "(b)",
    transform=ax.transAxes,
    ha="left",
    va="top",
)

ax.text(
    0.98,
    0.06,
    (
        r"$<1$: companion light "
        "\n"
        r"increases degeneracy"
        "\n"
        r"$>1$: companion light "
        "\n"
        r"breaks degeneracy"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
)


# ============================================================
# Save diagnostic
# ============================================================

fig2.tight_layout()

png_file2 = (
    figure_dir
    / "D_qM_qf_photocenter_comparison.png"
)

pdf_file2 = (
    figure_dir
    / "D_qM_qf_photocenter_comparison.pdf"
)

fig2.savefig(
    png_file2,
    dpi=600,
)

fig2.savefig(
    pdf_file2,
)

print()
print(
    "Saved:",
    png_file2,
)

print(
    "Saved:",
    pdf_file2,
)

plt.show()
