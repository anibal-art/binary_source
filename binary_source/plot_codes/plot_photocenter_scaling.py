#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT = Path(__file__).resolve()

SOURCE_DIR = SCRIPT.parents[1]
REPO_ROOT = SOURCE_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"

sys.path.insert(
    0,
    str(SCRIPT.parent),
)

from paper_style import apply_paper_style


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
    xmin=1e-4,
    xmax=1e-2,
    floor=1e-13,
):

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xmin)
        & (x <= xmax)
        & (y > floor)
    )

    if np.count_nonzero(valid) < 8:
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
    args.input.resolve()
    if args.input is not None
    else find_summary()
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


if not np.all(success):

    raise RuntimeError(
        "Input dataset contains failed fits."
    )


# ============================================================
# Slopes
# ============================================================

alpha = np.full(
    (
        len(families),
        len(qM),
    ),
    np.nan,
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
# Plot
# ============================================================

apply_paper_style()


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        11.5,
        4.6,
    ),
)


# ============================================================
# Panel A: D vs xi/u0
# ============================================================

ax = axes[0]


q_targets = [
    1e-2,
    1e-1,
    1.0,
]


line_styles = [
    "-",
    "--",
    ":",
]


for target, ls in zip(
    q_targets,
    line_styles,
):

    iq = nearest_index(
        qM,
        target,
    )


    # Dark companion
    ax.plot(
        xi,
        D[
            0,
            iq,
            :,
        ],
        linestyle=ls,
        linewidth=1.8,
        label=(
            rf"$q_M={qM[iq]:.2g}$, "
            r"$q_f=0$"
        ),
    )


    # Photocenter cancellation
    ax.plot(
        xi,
        D[
            1,
            iq,
            :,
        ],
        linestyle=ls,
        linewidth=1.8,
        alpha=0.65,
        label=(
            rf"$q_M={qM[iq]:.2g}$, "
            r"$q_f=q_M$"
        ),
    )


# Reference slopes
x_ref = np.array(
    [
        2e-4,
        2e-3,
    ]
)


ax.plot(
    x_ref,
    5e-6
    * (
        x_ref
        / x_ref[0]
    ),
    linewidth=1.2,
    label=r"$\propto \xi_{\rm rel}$",
)


ax.plot(
    x_ref,
    5e-10
    * (
        x_ref
        / x_ref[0]
    )**2,
    linewidth=1.2,
    label=r"$\propto \xi_{\rm rel}^{2}$",
)


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


ax.legend(
    frameon=False,
    fontsize=9,
    ncol=2,
)


ax.text(
    0.04,
    0.95,
    r"\textbf{(a)}",
    transform=ax.transAxes,
    ha="left",
    va="top",
)


# ============================================================
# Panel B: fitted exponent
# ============================================================

ax = axes[1]


for i_family, family in enumerate(
    families
):

    if family == "dark":

        label = r"$q_f=0$"

    else:

        label = r"$q_f=q_M$"


    ax.plot(
        qM,
        alpha[
            i_family,
            :,
        ],
        linewidth=2.0,
        label=label,
    )


ax.axhline(
    1.0,
    linewidth=1.0,
    linestyle="--",
)

ax.axhline(
    2.0,
    linewidth=1.0,
    linestyle="--",
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


ax.set_ylim(
    0.8,
    2.2,
)


ax.legend(
    frameon=False,
)


ax.text(
    0.04,
    0.95,
    r"\textbf{(b)}",
    transform=ax.transAxes,
    ha="left",
    va="top",
)


# ============================================================
# Save
# ============================================================

fig.tight_layout()


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
# Metadata sidecar
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
            "xi_fit_range=1e-4,1e-2",
            "objective=intrinsic_magnification_trapezoid",
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
