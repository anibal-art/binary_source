#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize


# ============================================================
# Paths
# ============================================================

INPUT = Path(
    "results/roman_asimov_f146_tE30/"
    "roman_intrinsic_grid_f146_tE30.npz"
)

OUTDIR = Path(
    "figures/current"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PDF = OUTDIR / "roman_intrinsic_comparison_f146.pdf"
OUT_PNG = OUTDIR / "roman_intrinsic_comparison_f146.png"


# ============================================================
# Style
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
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)


# ============================================================
# Load
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

d = np.load(
    INPUT,
    allow_pickle=False,
)

mags = np.asarray(
    d["f146_magnitudes"],
    dtype=float,
)

u0 = np.asarray(
    d["u0_grid"],
    dtype=float,
)

P_over_tE = np.asarray(
    d["P_over_tE"],
    dtype=float,
)

D = np.asarray(
    d["D_INTRINSIC"],
    dtype=float,
)

CHI2 = np.asarray(
    d["DELTA_CHI2"],
    dtype=float,
)

SUCCESS = np.asarray(
    d["SUCCESS"],
    dtype=bool,
)


# ============================================================
# Validation
# ============================================================

expected = (
    len(mags),
    len(u0),
    len(P_over_tE),
)

if CHI2.shape != expected:
    raise RuntimeError(
        f"DELTA_CHI2 shape={CHI2.shape}, "
        f"expected={expected}"
    )

if D.shape != (
    len(u0),
    len(P_over_tE),
):
    raise RuntimeError(
        f"D_INTRINSIC shape={D.shape}"
    )

if not np.all(SUCCESS):
    raise RuntimeError(
        "Some Roman fits are unsuccessful."
    )

if not np.all(
    np.isfinite(CHI2)
):
    raise RuntimeError(
        "Non-finite DeltaChi2 values."
    )

if not np.all(
    np.isfinite(D)
):
    raise RuntimeError(
        "Non-finite intrinsic D values."
    )


# ============================================================
# Coordinates
# ============================================================

X, Y = np.meshgrid(
    P_over_tE,
    u0,
)


# ============================================================
# Display quantity
# ============================================================

# Plot log10 DeltaChi2.
# Clip only for visualization; original values are untouched.
CHI2_FLOOR = 1.0e-2

logchi = np.log10(
    np.maximum(
        CHI2,
        CHI2_FLOOR,
    )
)

global_min = np.nanmin(
    logchi
)

global_max = np.nanmax(
    logchi
)

print(
    "log10 DeltaChi2 global range:",
    global_min,
    global_max,
)


# A fixed common range makes all three panels directly comparable.
VMIN = -2.0
VMAX = 8.0

norm = Normalize(
    vmin=VMIN,
    vmax=VMAX,
)


# ============================================================
# Contours
# ============================================================

D_LEVELS = [
    1.0e-3,
    1.0e-2,
]

CHI2_LEVEL = 100.0


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    len(mags),
    figsize=(13.2, 4.25),
    sharex=True,
    sharey=True,
)

if len(mags) == 1:
    axes = [axes]


mesh = None

for im, (ax, mag) in enumerate(
    zip(
        axes,
        mags,
    )
):

    # --------------------------------------------------------
    # Roman DeltaChi2 background
    # --------------------------------------------------------

    mesh = ax.pcolormesh(
        X,
        Y,
        logchi[im],
        shading="auto",
        cmap="viridis",
        norm=norm,
        rasterized=True,
    )

    # --------------------------------------------------------
    # Intrinsic D contours
    # --------------------------------------------------------

    # --------------------------------------------------------
    # White underlay for intrinsic D contours
    #
    # Drawing the same contours twice produces a robust halo:
    # a thick white line underneath and a thinner black line
    # on top. This keeps D visible across the full viridis map.
    # --------------------------------------------------------

    ax.contour(
        X,
        Y,
        D,
        levels=D_LEVELS,
        colors="white",
        linewidths=3.4,
        linestyles=[
            "--",
            "-",
        ],
        zorder=4,
    )

    cs_D = ax.contour(
        X,
        Y,
        D,
        levels=D_LEVELS,
        colors="black",
        linewidths=1.35,
        linestyles=[
            "--",
            "-",
        ],
        zorder=5,
    )

    labels_D = ax.clabel(
        cs_D,
        inline=False,
        fontsize=9,
        fmt={
            1.0e-3: r"$D=10^{-3}$",
            1.0e-2: r"$D=10^{-2}$",
        },
    )

    # White halo around the contour labels.
    for txt in labels_D:
        txt.set_path_effects(
            [
                pe.Stroke(
                    linewidth=3.0,
                    foreground="white",
                ),
                pe.Normal(),
            ]
        )

    # --------------------------------------------------------
    # Roman model-separation contour
    # --------------------------------------------------------

    cs_chi = ax.contour(
        X,
        Y,
        CHI2[im],
        levels=[CHI2_LEVEL],
        colors="white",
        linewidths=2.0,
    )

    labels_chi = ax.clabel(
        cs_chi,
        inline=False,
        fontsize=9,
        fmt={
            CHI2_LEVEL:
                r"$\Delta\chi^2=100$"
        },
    )

    # Dark halo improves the white-label contrast on the
    # brightest parts of the colormap.
    for txt in labels_chi:
        txt.set_path_effects(
            [
                pe.Stroke(
                    linewidth=2.5,
                    foreground="black",
                ),
                pe.Normal(),
            ]
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(
        P_over_tE.min(),
        P_over_tE.max(),
    )

    ax.set_ylim(
        u0.min(),
        u0.max(),
    )

    ax.set_title(
        rf"$F146={mag:.0f}$"
    )

    ax.set_xlabel(
        r"$P/t_E$"
    )

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )


axes[0].set_ylabel(
    r"$u_0$"
)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.885,
    bottom=0.18,
    top=0.90,
    wspace=0.08,
)


# ============================================================
# Shared colorbar
# ============================================================

cax = fig.add_axes(
    [
        0.905,
        0.18,
        0.015,
        0.72,
    ]
)

cbar = fig.colorbar(
    mesh,
    cax=cax,
    orientation="vertical",
)

cbar.set_label(
    r"$\log_{10}\Delta\chi^2_{\rm Roman}$"
)

cbar.ax.tick_params(
    which="both",
    direction="in",
)


# ============================================================
# Compact annotation
# ============================================================

fig.text(
    0.48,
    0.015,
    (
        r"Black contours: intrinsic $D_{\rm BSPL-PSPL}$; "
        r"white contour: representative Roman model-separation level."
    ),
    ha="center",
    va="bottom",
    fontsize=10,
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PNG,
    bbox_inches="tight",
    dpi=600,
)

plt.close(fig)

print()
print("Saved:")
print(" ", OUT_PDF)
print(" ", OUT_PNG)
