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
# Paper style (shared)
# ============================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (apply_paper_style, fig_width, panel_label, save_figure,
                         C_BLUE, C_VERMILION, LW_MAIN, LW_THIN)
import matplotlib.ticker as mticker



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
# Figure (printed width \textwidth = 6.14 in)
# ============================================================

apply_paper_style()

fig = plt.figure(figsize=(fig_width(1.0), 2.65))
ax1 = fig.add_axes([0.085, 0.165, 0.39, 0.815])
ax2 = fig.add_axes([0.585, 0.165, 0.405, 0.815])

# Physical period of each regime (stored configuration parameters)
P_over_tE_case = [float(broad.loc[broad["case"] == c, "P_over_tE"].iloc[0]) for c in cases]

# ---------------- Panel (a): strip plot + median and 16-84% interval
x = np.arange(3, dtype=float)
rng = np.random.default_rng(12345)          # same jitter seed as before

for j, case in enumerate(cases):
    values = pivot[case].to_numpy(dtype=float)
    jitter = rng.uniform(-0.09, 0.09, size=len(values))
    ax1.scatter(x[j] + jitter, values, s=9, facecolors="none", edgecolors="0.55",
                linewidths=0.5, zorder=2)

yerr = np.vstack([medians - p16, p84 - medians])
ax1.errorbar(x + 0.22, medians, yerr=yerr, fmt="o", ms=4, capsize=2.5,
             lw=LW_MAIN, color="black", zorder=5,
             label=r"median, 16--84\%")

fid_values = np.array([pivot.loc[fiducial_id, case] for case in cases], dtype=float)
ax1.scatter(x - 0.22, fid_values, marker="*", s=40, color=C_VERMILION,
            edgecolors="none", label="fiducial geometry", zorder=6)

ax1.set_yscale("log")
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(3e-6, 0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([rf"${fmt:.2g}$".replace("e", "") for fmt in P_over_tE_case])
ax1.xaxis.set_minor_locator(mticker.NullLocator())
ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
ax1.set_xlabel(r"$P/t_E$")
ax1.set_ylabel(r"$D_{\rm BSPL-PSPL}$")
ax1.legend(loc="lower left")
panel_label(ax1, "a")

# ---------------- Panel (b): slope deviations
x_dark, x_cancel = 0.0, 1.0
rng = np.random.default_rng(67890)          # same jitter seed as before
jitter_dark = rng.uniform(-0.10, 0.10, size=len(dark_delta))
jitter_cancel = rng.uniform(-0.10, 0.10, size=len(cancel_delta))

ax2.axhline(0.0, color="0.55", lw=LW_THIN, ls=":", zorder=1)
ax2.scatter(x_dark + jitter_dark, dark_delta, s=9, facecolors="none",
            edgecolors=C_BLUE, linewidths=0.5, zorder=2)
ax2.scatter(x_cancel + jitter_cancel, cancel_delta, s=9, facecolors="none",
            edgecolors=C_VERMILION, linewidths=0.5, zorder=2)
for xpos, values in ((x_dark, dark_delta), (x_cancel, cancel_delta)):
    med = np.median(values)
    lo, hi = np.percentile(values, 16), np.percentile(values, 84)
    ax2.errorbar(xpos + 0.22, med, yerr=[[med - lo], [hi - med]], fmt="D", ms=3.5,
                 capsize=2.5, lw=LW_MAIN, color="black", zorder=5)

ax2.set_xlim(-0.5, 1.5)
ax2.set_xticks([x_dark, x_cancel])
ax2.set_xticklabels([r"$q_f=0$", r"$q_f=q_M$"])
ax2.xaxis.set_minor_locator(mticker.NullLocator())
ax2.set_ylabel(r"$10^3\,(\gamma-\gamma_{\rm expected})$")
all_delta = np.concatenate([dark_delta, cancel_delta])
limit = 1.15 * np.max(np.abs(all_delta))
ax2.set_ylim(-limit, limit * 1.25)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f$"))
panel_label(ax2, "b")

save_figure(fig, OUTDIR / "geometry_robustness_appendix")
plt.close(fig)

VERIFY = OUTDIR.parent / "_verify"
VERIFY.mkdir(parents=True, exist_ok=True)
np.savez(VERIFY / "geometry_robustness_appendix_plotted_values.npz",
         D_short=pivot["one_short"].to_numpy(float),
         D_intermediate=pivot["one_intermediate"].to_numpy(float),
         D_long=pivot["one_long"].to_numpy(float),
         medians=medians, p16=p16, p84=p84, fid_values=fid_values,
         dark_delta=dark_delta, cancel_delta=cancel_delta)

# ============================================================
# Console summary
# ============================================================

print()
print("=" * 80)
print("GEOMETRY ROBUSTNESS FIGURE")
print("=" * 80)
print(f"Fiducial geometry: {fiducial_id}")
print(f"Intermediate > short,long: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%)")
print(f"Dark slope: median = {np.median(dark['slope']):.6f} "
      f"range = [{dark['slope'].min():.6f}, {dark['slope'].max():.6f}]")
print(f"Cancellation slope: median = {np.median(cancel['slope']):.6f} "
      f"range = [{cancel['slope'].min():.6f}, {cancel['slope'].max():.6f}]")
print("Saved:", OUTDIR / "geometry_robustness_appendix.pdf")
