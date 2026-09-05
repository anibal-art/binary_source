#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS = (
    ROOT
    / "results"
    / "validation_geometry_robustness"
)

OUTDIR = (
    ROOT
    / "figures"
    / "appendix"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


BROAD_FILE = (
    RESULTS
    / "geometry_broad.csv"
)

SLOPES_FILE = (
    RESULTS
    / "geometry_scaling_slopes.csv"
)


# ============================================================
# Paper style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "STIXGeneral",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",

        "font.size": 12,

        "axes.labelsize": 14,
        "axes.titlesize": 14,

        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "legend.fontsize": 10,

        "axes.linewidth": 1.0,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "xtick.major.size": 5,
        "ytick.major.size": 5,

        "xtick.minor.size": 3,
        "ytick.minor.size": 3,

        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


# ============================================================
# Load data
# ============================================================

broad = pd.read_csv(
    BROAD_FILE
)

slopes = pd.read_csv(
    SLOPES_FILE
)


# ============================================================
# Prepare broad table
# ============================================================

cases = [
    "one_short",
    "one_intermediate",
    "one_long",
]

case_labels = [
    "short",
    "intermediate",
    "long",
]


pivot = (
    broad[
        broad["case"].isin(
            cases
        )
    ]
    .pivot(
        index="geometry_id",
        columns="case",
        values="D",
    )
)


# Keep geometry ordering deterministic
geometry_ids = sorted(
    pivot.index
)

pivot = pivot.loc[
    geometry_ids
]


# ============================================================
# Fiducial geometry
# ============================================================

fid_rows = broad[
    broad[
        "fiducial_geometry"
    ].astype(bool)
]

if len(
    fid_rows
) == 0:

    raise RuntimeError(
        "Could not identify fiducial geometry."
    )


fiducial_id = str(
    fid_rows.iloc[0][
        "geometry_id"
    ]
)


# ============================================================
# Summary statistics
# ============================================================

D_arrays = [
    pivot[
        case
    ].to_numpy(
        dtype=float
    )
    for case in cases
]


medians = np.array(
    [
        np.median(x)
        for x in D_arrays
    ]
)


p16 = np.array(
    [
        np.percentile(
            x,
            16,
        )
        for x in D_arrays
    ]
)


p84 = np.array(
    [
        np.percentile(
            x,
            84,
        )
        for x in D_arrays
    ]
)


# Ordering fraction
ordering_ok = (
    (
        pivot[
            "one_intermediate"
        ]
        >
        pivot[
            "one_short"
        ]
    )
    &
    (
        pivot[
            "one_intermediate"
        ]
        >
        pivot[
            "one_long"
        ]
    )
)


n_ok = int(
    ordering_ok.sum()
)

n_total = int(
    len(
        ordering_ok
    )
)


# ============================================================
# Prepare slope data
# ============================================================

dark = slopes[
    slopes["family"]
    == "dark"
].copy()

cancel = slopes[
    slopes["family"]
    == "cancel"
].copy()


dark[
    "delta_gamma"
] = (
    dark["slope"]
    - 1.0
)


cancel[
    "delta_gamma"
] = (
    cancel["slope"]
    - 2.0
)


# Express in 1e-3 units
dark_delta = (
    1.0e3
    * dark[
        "delta_gamma"
    ].to_numpy(
        dtype=float
    )
)

cancel_delta = (
    1.0e3
    * cancel[
        "delta_gamma"
    ].to_numpy(
        dtype=float
    )
)


# ============================================================
# Figure
# ============================================================

fig, (
    ax1,
    ax2,
) = plt.subplots(
    1,
    2,
    figsize=(
        10.8,
        4.6,
    ),
)


# ============================================================
# Panel A
# ============================================================

x = np.arange(
    3,
    dtype=float,
)


rng = np.random.default_rng(
    12345
)


# ------------------------------------------------------------
# All geometries
# ------------------------------------------------------------

for j, case in enumerate(
    cases
):

    values = pivot[
        case
    ].to_numpy(
        dtype=float
    )

    jitter = rng.uniform(
        -0.09,
        0.09,
        size=len(
            values
        ),
    )

    ax1.scatter(
        x[j]
        + jitter,
        values,
        s=18,
        facecolors="none",
        edgecolors="0.55",
        linewidths=0.7,
        alpha=0.65,
        zorder=2,
    )


# ------------------------------------------------------------
# p16-p84 + median
# ------------------------------------------------------------

yerr = np.vstack(
    [
        medians
        - p16,

        p84
        - medians,
    ]
)


ax1.errorbar(
    x,
    medians,
    yerr=yerr,
    fmt="o",
    markersize=7,
    capsize=4,
    linewidth=1.8,
    color="tab:red",
    label=r"median and $p_{16}$--$p_{84}$",
    zorder=5,
)


ax1.plot(
    x,
    medians,
    "-",
    linewidth=1.8,
    color="tab:red",
    zorder=4,
)


# ------------------------------------------------------------
# Fiducial geometry
# ------------------------------------------------------------

fid_values = np.array(
    [
        pivot.loc[
            fiducial_id,
            case,
        ]
        for case in cases
    ],
    dtype=float,
)


ax1.scatter(
    x,
    fid_values,
    marker="*",
    s=110,
    color="black",
    label="fiducial geometry",
    zorder=6,
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

ax1.set_yscale(
    "log"
)

ax1.set_xticks(
    x
)

ax1.set_xticklabels(
    case_labels
)

ax1.set_ylabel(
    r"$D_{\rm BSPL-PSPL}$"
)

ax1.set_xlabel(
    "orbital-period regime"
)

ax1.set_title(
    "Broad period structure"
)

ax1.grid(
    axis="y",
    which="both",
    alpha=0.18,
)

ax1.set_axisbelow(
    True
)


ax1.legend(
    loc="upper right",
    frameon=False,
)


# ------------------------------------------------------------
# Small unobtrusive annotation
# ------------------------------------------------------------

ax1.text(
    0.04,
    0.06,
    (
        rf"$N={n_total}$ geometries"
        "\n"
        rf"$D_{{\rm int}}>D_{{\rm short}},D_{{\rm long}}$: "
        rf"{n_ok}/{n_total}"
    ),
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=10,
)


# ============================================================
# Panel B
# ============================================================

x_dark = 0.0
x_cancel = 1.0


rng = np.random.default_rng(
    67890
)


jitter_dark = rng.uniform(
    -0.10,
    0.10,
    size=len(
        dark_delta
    ),
)


jitter_cancel = rng.uniform(
    -0.10,
    0.10,
    size=len(
        cancel_delta
    ),
)


# ------------------------------------------------------------
# Individual geometries
# ------------------------------------------------------------

ax2.scatter(
    x_dark
    + jitter_dark,
    dark_delta,
    s=22,
    alpha=0.65,
    color="tab:blue",
    edgecolors="none",
    label="individual geometries",
    zorder=2,
)


ax2.scatter(
    x_cancel
    + jitter_cancel,
    cancel_delta,
    s=22,
    alpha=0.65,
    color="tab:orange",
    edgecolors="none",
    zorder=2,
)


# ------------------------------------------------------------
# Medians + p16-p84
# ------------------------------------------------------------

for xpos, values, color in [
    (
        x_dark,
        dark_delta,
        "tab:blue",
    ),
    (
        x_cancel,
        cancel_delta,
        "tab:orange",
    ),
]:

    med = np.median(
        values
    )

    lo = np.percentile(
        values,
        16,
    )

    hi = np.percentile(
        values,
        84,
    )

    ax2.errorbar(
        xpos,
        med,
        yerr=[
            [
                med
                - lo
            ],
            [
                hi
                - med
            ],
        ],
        fmt="D",
        markersize=7,
        capsize=5,
        linewidth=2.0,
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
    )


# ------------------------------------------------------------
# Expected value
# ------------------------------------------------------------

ax2.axhline(
    0.0,
    color="black",
    linewidth=1.0,
    linestyle="--",
    zorder=1,
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

ax2.set_xticks(
    [
        x_dark,
        x_cancel,
    ]
)


ax2.set_xticklabels(
    [
        r"dark companion" "\n" r"($q_f=0$)",
        r"photocenter cancellation" "\n" r"($q_f=q_M$)",
    ]
)


ax2.set_ylabel(
    r"$10^3\,(\gamma-\gamma_{\rm expected})$"
)


ax2.set_title(
    "Small-separation scaling"
)


ax2.grid(
    axis="y",
    alpha=0.18,
)


ax2.set_axisbelow(
    True
)


# ------------------------------------------------------------
# Useful limits
# ------------------------------------------------------------

all_delta = np.concatenate(
    [
        dark_delta,
        cancel_delta,
    ]
)


limit = (
    1.15
    * np.max(
        np.abs(
            all_delta
        )
    )
)


ax2.set_ylim(
    -limit,
    limit,
)


# ============================================================
# Panel labels
# ============================================================

ax1.text(
    -0.12,
    1.03,
    "A",
    transform=ax1.transAxes,
    fontsize=15,
    fontweight="bold",
    va="bottom",
)


ax2.text(
    -0.12,
    1.03,
    "B",
    transform=ax2.transAxes,
    fontsize=15,
    fontweight="bold",
    va="bottom",
)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.99,
    bottom=0.18,
    top=0.88,
    wspace=0.30,
)


# ============================================================
# Save
# ============================================================

pdf_path = (
    OUTDIR
    / "geometry_robustness_appendix.pdf"
)

png_path = (
    OUTDIR
    / "geometry_robustness_appendix.png"
)


fig.savefig(
    pdf_path
)


fig.savefig(
    png_path,
    dpi=600,
)


plt.close(
    fig
)


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 80)
print("GEOMETRY ROBUSTNESS FIGURE")
print("=" * 80)

print(
    f"Fiducial geometry: "
    f"{fiducial_id}"
)

print()

print(
    f"Intermediate > short,long: "
    f"{n_ok}/{n_total} "
    f"({100*n_ok/n_total:.1f}%)"
)

print()

print(
    "Dark slope:"
)

print(
    f"  median = "
    f"{np.median(dark['slope']):.6f}"
)

print(
    f"  range  = "
    f"[{dark['slope'].min():.6f}, "
    f"{dark['slope'].max():.6f}]"
)

print()

print(
    "Cancellation slope:"
)

print(
    f"  median = "
    f"{np.median(cancel['slope']):.6f}"
)

print(
    f"  range  = "
    f"[{cancel['slope'].min():.6f}, "
    f"{cancel['slope'].max():.6f}]"
)

print()

print(
    "Saved:"
)

print(
    " ",
    pdf_path
)

print(
    " ",
    png_path
)
