#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photocenter-cancellation scaling figure.

Scientific goal
---------------
Show the perturbative behavior

    q_f = 0       -> D ~ xi_rel
    q_f = q_M     -> D ~ xi_rel^2

and measure the corresponding power-law exponent alpha.

This script only reads an existing numerical summary.
It never runs simulations or fits.
"""

from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# ============================================================
# Paths
# ============================================================

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

from paper_style import (apply_paper_style, fig_width, panel_label,
                         save_figure, C_BLUE, C_VERMILION, C_GREEN,
                         LS_DARK, LS_CANCEL, LW_THIN, FS_ANNOT)
from matplotlib.ticker import (LogLocator, NullFormatter, MultipleLocator,
                               FormatStrFormatter)


# ============================================================
# Perturbative-fit configuration
# ============================================================

XI_FIT_MIN = 1e-4
XI_FIT_MAX = 1e-2

D_FLOOR = 1e-13
MIN_POINTS = 8


# ============================================================
# Helpers
# ============================================================

def current_commit():

    try:

        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short=12",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except Exception:

        return None


def find_summary():

    candidates = list(
        RESULTS_ROOT.glob(
            "final_*/photocenter_small_xi_tE150/"
            "summary_photocenter_small_xi.npz"
        )
    )

    candidates = [
        p
        for p in candidates
        if "_dirty" not in p.parts[-3]
    ]

    if not candidates:

        raise FileNotFoundError(
            "No photocenter summary found."
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def nearest_index(
    array,
    target,
):

    array = np.asarray(
        array,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                np.log10(array)
                - np.log10(target)
            )
        )
    )


def powerlaw_slope(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= XI_FIT_MIN)
        & (x <= XI_FIT_MAX)
        & (y > D_FLOOR)
    )

    if (
        np.count_nonzero(valid)
        < MIN_POINTS
    ):
        return np.nan

    slope, _ = np.polyfit(
        np.log10(
            x[valid]
        ),
        np.log10(
            y[valid]
        ),
        1,
    )

    return float(
        slope
    )


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=Path,
    default=None,
)

args = parser.parse_args()


# ============================================================
# Input
# ============================================================

filename = (
    args.input
    .expanduser()
    .resolve()
    if args.input is not None
    else find_summary().resolve()
)


if not filename.exists():

    raise FileNotFoundError(
        filename
    )


with np.load(
    filename,
    allow_pickle=False,
) as d:

    families = np.asarray(
        d["family_names"]
    ).astype(str)

    qM = np.asarray(
        d["qM_grid"],
        dtype=float,
    )

    xi = np.asarray(
        d["xi_over_u0_grid"],
        dtype=float,
    )

    D = np.asarray(
        d["D"],
        dtype=float,
    )

    success = np.asarray(
        d["SUCCESS"],
        dtype=bool,
    )

    dataset_commit = str(
        d["code_commit"].item()
    )


if not np.all(
    success
):

    raise RuntimeError(
        "Input photocenter dataset contains failed fits."
    )


# ============================================================
# Identify families robustly
# ============================================================

family_to_index = {
    family: i
    for i, family
    in enumerate(families)
}


required_families = (
    "dark",
    "photocenter_cancel",
)


for family in required_families:

    if family not in family_to_index:

        raise KeyError(
            f"Missing family: {family}"
        )


i_dark = family_to_index[
    "dark"
]

i_cancel = family_to_index[
    "photocenter_cancel"
]


# ============================================================
# Power-law slopes
# ============================================================

alpha = np.full(
    (
        len(families),
        len(qM),
    ),
    np.nan,
    dtype=float,
)


for i_family in range(
    len(families)
):

    for i_q in range(
        len(qM)
    ):

        alpha[
            i_family,
            i_q,
        ] = powerlaw_slope(
            xi,
            D[
                i_family,
                i_q,
                :,
            ],
        )


# ============================================================
# Output
# ============================================================

figure_dir = REPO_ROOT / "figures" / f"draft_{dataset_commit}"
figure_dir.mkdir(parents=True, exist_ok=True)
verify_dir = REPO_ROOT / "figures" / "_verify"
verify_dir.mkdir(parents=True, exist_ok=True)

# The small-separation scaling exponent is called gamma in the paper
# (alpha denotes an orbital-orientation angle).
gamma = alpha

apply_paper_style()

q_targets = [1e-2, 1e-1, 1.0]
q_styles = [
    dict(color=C_BLUE, marker="o"),
    dict(color=C_VERMILION, marker="s"),
    dict(color=C_GREEN, marker="^"),
]
MARK_EVERY = 0.12


def draw_scaling_panel(ax, label_panel=True):
    # slope-fit range (shaded) used for panel (b) / Table 7
    ax.axvspan(XI_FIT_MIN, XI_FIT_MAX, color="0.5", alpha=0.08, lw=0, zorder=0)

    for target, st in zip(q_targets, q_styles):
        iq = nearest_index(qM, target)
        ax.plot(xi, D[i_dark, iq, :], color=st["color"], ls=LS_DARK,
                marker=st["marker"], markevery=MARK_EVERY, ms=3.0, mfc="white")
        ax.plot(xi, D[i_cancel, iq, :], color=st["color"], ls=LS_CANCEL,
                marker=st["marker"], markevery=MARK_EVERY, ms=3.0)

    # Reference slopes (visual guides only), placed in empty regions
    x_lin = np.logspace(np.log10(4e-3), np.log10(3e-2), 40)
    y_lin = 6e-3 * (x_lin / x_lin[0])
    ax.plot(x_lin, y_lin, color="0.35", lw=LW_THIN)
    ax.text(x_lin[-1] * 1.15, y_lin[-1], r"$\propto\xi_{\rm rel}$",
            fontsize=FS_ANNOT, ha="left", va="center", color="0.25")
    x_quad = np.logspace(np.log10(1.5e-2), np.log10(1.2e-1), 40)
    y_quad = 3e-9 * (x_quad / x_quad[0]) ** 2
    ax.plot(x_quad, y_quad, color="0.35", lw=LW_THIN)
    ax.text(x_quad[-1] * 1.15, y_quad[-1], r"$\propto\xi_{\rm rel}^2$",
            fontsize=FS_ANNOT, ha="left", va="center", color="0.25")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xi.min(), xi.max())
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=30))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\xi_{\rm rel}/u_0$")
    ax.set_ylabel(r"$D_{\rm BSPL-PSPL}$")

    q_handles = [Line2D([0], [0], color=st["color"], ls="-", marker=st["marker"],
                        ms=3.0, label=rf"${t:g}$")
                 for t, st in zip(q_targets, q_styles)]
    family_handles = [
        Line2D([0], [0], color="0.15", ls=LS_DARK, label=r"$q_f=0$"),
        Line2D([0], [0], color="0.15", ls=LS_CANCEL, label=r"$q_f=q_M$"),
    ]
    leg_q = ax.legend(handles=q_handles, title=r"$q_M$", loc="lower right")
    ax.add_artist(leg_q)
    ax.legend(handles=family_handles, loc="upper left",
              bbox_to_anchor=(0.0, 0.89))
    if label_panel:
        panel_label(ax, "a")


# ---------------- manuscript version: (a) D(xi), (b) gamma(q_M)
fig = plt.figure(figsize=(fig_width(1.0), 2.7))
ax_a = fig.add_axes([0.085, 0.15, 0.40, 0.83])
ax_b = fig.add_axes([0.585, 0.15, 0.40, 0.83])

draw_scaling_panel(ax_a)

ax_b.axhline(1.0, color="0.55", lw=LW_THIN, ls=":", zorder=0)
ax_b.axhline(2.0, color="0.55", lw=LW_THIN, ls=":", zorder=0)
ax_b.plot(qM, gamma[i_dark, :], color="k", ls=LS_DARK, label=r"$q_f=0$")
ax_b.plot(qM, gamma[i_cancel, :], color="k", ls=LS_CANCEL, label=r"$q_f=q_M$")
ax_b.set_xscale("log")
ax_b.set_xlim(qM.min(), qM.max())
ax_b.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
ax_b.set_ylim(0.9, 2.2)
ax_b.yaxis.set_major_locator(MultipleLocator(0.2))
ax_b.yaxis.set_minor_locator(MultipleLocator(0.05))
ax_b.yaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
ax_b.set_xlabel(r"$q_M$")
ax_b.set_ylabel(r"$\gamma$ in $D\propto(\xi_{\rm rel}/u_0)^{\gamma}$")
ax_b.legend(loc="center right")
panel_label(ax_b, "b")

png, pdf = save_figure(fig, figure_dir / "photocenter_scaling")[::-1]
plt.close(fig)

# ---------------- alternative: panel (a) only (slopes are in Table 7)
fig = plt.figure(figsize=(fig_width(0.6), 2.7))
ax = fig.add_axes([0.155, 0.15, 0.825, 0.83])
draw_scaling_panel(ax, label_panel=False)
png_alt, pdf_alt = save_figure(fig, figure_dir / "photocenter_scaling_alt")[::-1]
plt.close(fig)

np.savez(verify_dir / "photocenter_scaling_plotted_values.npz",
         xi=xi, qM=qM, D_dark=D[i_dark], D_cancel=D[i_cancel],
         gamma_dark=gamma[i_dark], gamma_cancel=gamma[i_cancel],
         q_indices=np.array([nearest_index(qM, t) for t in q_targets]))

metadata = figure_dir / "photocenter_scaling.txt"
metadata.write_text("\n".join([
    f"dataset={filename}",
    f"dataset_commit={dataset_commit}",
    "figure=photocenter_scaling",
    f"xi_fit_min={XI_FIT_MIN}",
    f"xi_fit_max={XI_FIT_MAX}",
    f"D_floor={D_FLOOR}",
    "dark_expected_gamma=1",
    "photocenter_cancel_expected_gamma=2",
    "",
]))
print("Input :", filename)
print("PDF :", pdf, pdf_alt)
print("PNG :", png, png_alt)
print("Meta :", metadata)
