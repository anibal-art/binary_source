#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ============================================================
# Paths
# ============================================================

INPUT = Path(
    "results/roman_asimov/combined_intrinsic_grid/"
    "roman_intrinsic_grid_W149_19_21_23.npz"
)

OUTDIR = Path(
    "figures/roman_asimov"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PDF = OUTDIR / "roman_intrinsic_comparison.pdf"
OUT_PNG = OUTDIR / "roman_intrinsic_comparison.png"


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
    d["w149_magnitudes"],
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

    cs_D = ax.contour(
        X,
        Y,
        D,
        levels=D_LEVELS,
        colors="black",
        linewidths=1.25,
        linestyles=[
            "--",
            "-",
        ],
    )

    ax.clabel(
        cs_D,
        inline=True,
        fontsize=9,
        fmt={
            1.0e-3: r"$D=10^{-3}$",
            1.0e-2: r"$D=10^{-2}$",
        },
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

    ax.clabel(
        cs_chi,
        inline=True,
        fontsize=9,
        fmt={
            CHI2_LEVEL:
                r"$\Delta\chi^2=100$"
        },
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
        rf"$W149={mag:.0f}$"
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
# Shared colorbar
# ============================================================

cbar = fig.colorbar(
    mesh,
    ax=axes,
    orientation="vertical",
    fraction=0.025,
    pad=0.025,
)

cbar.set_label(
    r"$\log_{10}\Delta\chi^2_{\rm Roman}$"
)


# ============================================================
# Compact annotation
# ============================================================

fig.text(
    0.5,
    0.015,
    (
        r"Black contours: intrinsic $D_{\rm BSPL-PSPL}$; "
        r"white contour: representative Roman model-separation level."
    ),
    ha="center",
    va="bottom",
    fontsize=10,
)


fig.subplots_adjust(
    left=0.075,
    right=0.91,
    bottom=0.18,
    top=0.90,
    wspace=0.08,
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
