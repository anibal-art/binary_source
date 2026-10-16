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

OUT_PDF = OUTDIR / "roman_intrinsic_comparison.pdf"
OUT_PNG = OUTDIR / "roman_intrinsic_comparison.png"


# ============================================================
# Style (shared paper style)
# ============================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (apply_paper_style, fig_width, heatmap, panel_label,
                         save_figure, halo, d_norm, place_contour_labels,
                         CHI2_CMAP, D_CMAP, D_LABEL, F146_STYLE, C_ORANGE,
                         LW_MAIN, LW_THIN, FS_ANNOT)
import matplotlib.ticker as mticker

apply_paper_style()



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

X, Y = np.meshgrid(P_over_tE, u0)

# Display quantity: log10 DeltaChi2, clipped only for the colour scale.
CHI2_FLOOR = 1.0e-2
logchi = np.log10(np.maximum(CHI2, CHI2_FLOOR))
print("log10 DeltaChi2 global range:", np.nanmin(logchi), np.nanmax(logchi))

# Fixed common range so the three panels are directly comparable.
VMIN = -2.0
VMAX = 8.0
norm = Normalize(vmin=VMIN, vmax=VMAX)

D_LEVELS = [1.0e-3, 1.0e-2]
CHI2_LEVEL = 100.0

U0_TICKS = [0.01, 0.03, 0.1, 0.3]
U0_LABELS = ["$0.01$", "$0.03$", "$0.1$", "$0.3$"]


def format_u0_axis(axis):
    axis.set_major_locator(mticker.FixedLocator(U0_TICKS))
    axis.set_major_formatter(mticker.FixedFormatter(U0_LABELS))
    axis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
    axis.set_minor_formatter(mticker.NullFormatter())


# ============================================================
# Manuscript version: three panels, log10 DeltaChi2 + contours
# ============================================================

fig = plt.figure(figsize=(fig_width(1.0), 2.7))
left, gap, right_cb = 0.075, 0.018, 0.875
w = (right_cb - 0.02 - left - 2 * gap) / 3
bottom, top = 0.15, 0.81
axes = [fig.add_axes([left + k * (w + gap), bottom, w, top - bottom]) for k in range(len(mags))]
cax = fig.add_axes([right_cb, bottom, 0.016, top - bottom])

mesh = None
for im, (ax, mag) in enumerate(zip(axes, mags)):
    mesh = heatmap(ax, X, Y, logchi[im], cmap=CHI2_CMAP, norm=norm)

    ax.contour(X, Y, D, levels=D_LEVELS, colors="white", linewidths=2.4,
               linestyles=["--", "-"], zorder=4)
    cs_D = ax.contour(X, Y, D, levels=D_LEVELS, colors="black",
                      linewidths=LW_THIN + 0.2, linestyles=["--", "-"], zorder=5)
    cs_chi = ax.contour(X, Y, CHI2[im], levels=[CHI2_LEVEL], colors="white",
                        linewidths=LW_MAIN, zorder=6)
    cs_chi.set_path_effects(halo(2.2, "black"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(P_over_tE.min(), P_over_tE.max())
    ax.set_ylim(u0.min(), u0.max())
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    format_u0_axis(ax.yaxis)
    if im > 0:
        ax.tick_params(labelleft=False)
    ax.set_xlabel(r"$P/t_E$")
    ax.set_title(rf"F146 $={mag:.0f}$")
    panel_label(ax, "abc"[im])


axes[0].set_ylabel(r"$u_0$")

# Contours are identified by a legend (lines are too close for inline labels)
legend_handles = [
    plt.Line2D([], [], color="black", ls="-", lw=LW_THIN + 0.2,
               path_effects=halo(2.4, "white"), label=r"$D_{\rm BSPL-PSPL}=10^{-2}$"),
    plt.Line2D([], [], color="black", ls="--", lw=LW_THIN + 0.2,
               path_effects=halo(2.4, "white"), label=r"$D_{\rm BSPL-PSPL}=10^{-3}$"),
    plt.Line2D([], [], color="white", ls="-", lw=LW_MAIN,
               path_effects=halo(2.2, "black"), label=r"$\Delta\chi^2_{\rm Roman}=100$"),
]
leg = fig.legend(handles=legend_handles, loc="upper center", ncol=3,
                 bbox_to_anchor=(0.47, 1.0), frameon=True, fancybox=False,
                 edgecolor="none", facecolor="0.80", borderpad=0.35,
                 columnspacing=1.5)

cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
cbar.set_label(r"$\log_{10}\Delta\chi^2_{\rm Roman}$")
cbar.ax.tick_params(which="both", direction="in")

save_figure(fig, OUT_PDF.with_suffix(""))
plt.close(fig)

# ============================================================
# Alternative: one D map in (u0, P/tE) with the three
# DeltaChi2_Roman = 100 contours (F146 = 19, 21, 23)
# ============================================================

fig = plt.figure(figsize=(fig_width(0.6), 3.05))
ax = fig.add_axes([0.16, 0.13, 0.80, 0.68])
cax = fig.add_axes([0.16, 0.885, 0.80, 0.025])

pcm = heatmap(ax, u0, P_over_tE, D.T, cmap=D_CMAP, norm=d_norm())
cs_list, fmts, cols, halos = [], [], [], []
for im, mag in enumerate(mags):
    st = F146_STYLE[int(round(mag))]
    cs = ax.contour(u0, P_over_tE, CHI2[im].T, levels=[CHI2_LEVEL],
                    colors=st["color"], linestyles=st["ls"], linewidths=LW_MAIN + 0.2,
                    zorder=6)
    outline = "black" if st["color"].upper() in ("#FFFFFF", "WHITE", C_ORANGE.upper()) else "white"
    cs.set_path_effects(halo(2.4, outline))
    cs_list.append(cs)
    fmts.append({CHI2_LEVEL: rf"${mag:.0f}$"})
    cols.append(st["color"])
    halos.append(halo(1.8, outline))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(u0.min(), u0.max())
ax.set_ylim(P_over_tE.min(), P_over_tE.max())
ax.xaxis.set_major_locator(mticker.FixedLocator(U0_TICKS))
ax.xaxis.set_major_formatter(mticker.FixedFormatter(U0_LABELS))
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel(r"$u_0$")
ax.set_ylabel(r"$P/t_E$")
place_contour_labels(ax, cs_list, fmts=fmts, colors=cols, label_halo=halos,
                     box=(0.12, 0.88, 0.15, 0.85))
ax.text(0.96, 0.95, r"$\Delta\chi^2_{\rm Roman}=100$, F146 $=19,21,23$",
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
        color="white")

cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.tick_params(which="both", direction="in", pad=1.5)
cbar.set_label(D_LABEL, labelpad=3)

save_figure(fig, OUTDIR / "roman_intrinsic_comparison_alt")
plt.close(fig)

VERIFY = Path("figures/_verify")
VERIFY.mkdir(parents=True, exist_ok=True)
np.savez(VERIFY / "roman_intrinsic_comparison_plotted_values.npz",
         u0=u0, P_over_tE=P_over_tE, D=D, CHI2=CHI2, logchi=logchi, mags=mags,
         D_levels=np.array(D_LEVELS), chi2_level=CHI2_LEVEL)

print()
print("Saved:")
for p in (OUT_PDF, OUT_PNG, OUTDIR / "roman_intrinsic_comparison_alt.pdf",
          OUTDIR / "roman_intrinsic_comparison_alt.png"):
    print(" ", p)
