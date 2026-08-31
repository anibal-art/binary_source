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

from paper_style import apply_paper_style


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

figure_dir = (
    REPO_ROOT
    / "figures"
    / f"draft_{dataset_commit}"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Plot style
# ============================================================

apply_paper_style()


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        11.5,
        4.7,
    ),
    constrained_layout=True,
)


# ============================================================
# Use color only for qM.
# Use linestyle only for luminous-source family.
# ============================================================

color_cycle = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()
    .get(
        "color",
        [],
    )
)


if len(color_cycle) < 3:

    raise RuntimeError(
        "Matplotlib color cycle must contain at least 3 colors."
    )


q_targets = [
    1e-2,
    1e-1,
    1.0,
]


q_colors = (
    color_cycle[:3]
)


# ============================================================
# Panel A:
# D vs xi_rel/u0
# ============================================================

ax = axes[0]


# ------------------------------------------------------------
# Mark the range used to measure alpha.
# ------------------------------------------------------------

ax.axvspan(
    XI_FIT_MIN,
    XI_FIT_MAX,
    color="0.5",
    alpha=0.07,
    zorder=0,
)


ax.text(
    np.sqrt(
        XI_FIT_MIN
        * XI_FIT_MAX
    ),
    0.975,
    "slope-fit range",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=10,
)


# ------------------------------------------------------------
# Numerical curves
# ------------------------------------------------------------

for target, color in zip(
    q_targets,
    q_colors,
):

    iq = nearest_index(
        qM,
        target,
    )


    # Dark companion:
    # first-order photocenter displacement survives.
    ax.plot(
        xi,
        D[
            i_dark,
            iq,
            :,
        ],
        color=color,
        linestyle="-",
        linewidth=2.0,
    )


    # qf=qM:
    # first-order photocenter displacement cancels.
    ax.plot(
        xi,
        D[
            i_cancel,
            iq,
            :,
        ],
        color=color,
        linestyle="--",
        linewidth=2.0,
    )


# ------------------------------------------------------------
# Reference slopes
#
# These are visual guides only.
# ------------------------------------------------------------

x_ref_linear = np.logspace(
    -3.8,
    -2.7,
    40,
)

x0_linear = x_ref_linear[0]

y_ref_linear = (
    5e-6
    * (
        x_ref_linear
        / x0_linear
    )
)


ax.plot(
    x_ref_linear,
    y_ref_linear,
    color="0.40",
    linewidth=1.1,
)


ax.text(
    x_ref_linear[-1] * 1.08,
    y_ref_linear[-1],
    r"slope $1$",
    fontsize=9,
    ha="left",
    va="center",
)


x_ref_quad = np.logspace(
    -3.8,
    -2.7,
    40,
)

x0_quad = x_ref_quad[0]

y_ref_quad = (
    3e-10
    * (
        x_ref_quad
        / x0_quad
    )**2
)


ax.plot(
    x_ref_quad,
    y_ref_quad,
    color="0.40",
    linewidth=1.1,
)


ax.text(
    x_ref_quad[-1] * 1.08,
    y_ref_quad[-1],
    r"slope $2$",
    fontsize=9,
    ha="left",
    va="center",
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)


ax.set_xlabel(
    r"$\xi_{\rm rel}/u_0$"
)

ax.set_ylabel(
    r"$D_{\rm BSPL-PSPL}$"
)


# ------------------------------------------------------------
# Separate legends:
#
# color     -> qM
# linestyle -> qf family
# ------------------------------------------------------------

q_handles = [

    Line2D(
        [0],
        [0],
        color=color,
        linestyle="-",
        linewidth=2.2,
        label=rf"${target:g}$",
    )

    for target, color
    in zip(
        q_targets,
        q_colors,
    )
]


family_handles = [

    Line2D(
        [0],
        [0],
        color="0.15",
        linestyle="-",
        linewidth=2.2,
        label=r"$q_f=0$",
    ),

    Line2D(
        [0],
        [0],
        color="0.15",
        linestyle="--",
        linewidth=2.2,
        label=r"$q_f=q_M$",
    ),
]


legend_q = ax.legend(
    handles=q_handles,
    title=r"$q_M$",
    frameon=False,
    loc="lower right",
    fontsize=10,
    title_fontsize=10,
)


ax.add_artist(
    legend_q
)


ax.legend(
    handles=family_handles,
    title="Flux ratio",
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.82),
    fontsize=10,
    title_fontsize=10,
)


ax.text(
    0.04,
    0.95,
    "(a)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)


# ============================================================
# Panel B:
# measured alpha(qM)
# ============================================================

ax = axes[1]


family_colors = (
    color_cycle[:2]
)


ax.plot(
    qM,
    alpha[
        i_dark,
        :,
    ],
    color=family_colors[0],
    linewidth=2.2,
    label=r"$q_f=0$",
)


ax.plot(
    qM,
    alpha[
        i_cancel,
        :,
    ],
    color=family_colors[1],
    linewidth=2.2,
    label=r"$q_f=q_M$",
)


# ------------------------------------------------------------
# Expected perturbative orders
# ------------------------------------------------------------

ax.axhline(
    1.0,
    color="0.4",
    linewidth=1.1,
    linestyle="--",
    zorder=0,
)

ax.axhline(
    2.0,
    color="0.4",
    linewidth=1.1,
    linestyle="--",
    zorder=0,
)


ax.set_xscale(
    "log"
)


ax.set_xlabel(
    r"$q_M$"
)

ax.set_ylabel(
    r"$\alpha$ in "
    r"$D\propto(\xi_{\rm rel}/u_0)^\alpha$"
)


# The interesting information is concentrated around alpha=1 and 2.
ax.set_ylim(
    0.93,
    2.07,
)


ax.legend(
    frameon=False,
    loc="center right",
)


ax.text(
    0.04,
    0.90,
    "(b)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)


# ============================================================
# Save
# ============================================================

png = (
    figure_dir
    / "photocenter_scaling.png"
)

pdf = (
    figure_dir
    / "photocenter_scaling.pdf"
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


# ============================================================
# Metadata
# ============================================================

metadata = (
    figure_dir
    / "photocenter_scaling.txt"
)


metadata.write_text(
    "\n".join(
        [
            f"dataset={filename}",
            f"dataset_commit={dataset_commit}",
            "figure=photocenter_scaling",
            f"xi_fit_min={XI_FIT_MIN}",
            f"xi_fit_max={XI_FIT_MAX}",
            f"D_floor={D_FLOOR}",
            "dark_expected_alpha=1",
            "photocenter_cancel_expected_alpha=2",
            "",
        ]
    )
)


print(
    "Input :",
    filename,
)

print(
    "PNG   :",
    png,
)

print(
    "PDF   :",
    pdf,
)

print(
    "Meta  :",
    metadata,
)
