#!/usr/bin/env python3

from pathlib import Path
import argparse
import subprocess

import numpy as np


# ============================================================
# Repository paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

SOURCE_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SOURCE_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"


# ============================================================
# Configuration for perturbative fit
# ============================================================

XI_MIN = 1e-4
XI_MAX = 1e-2

# Avoid fitting values already at numerical zero.
D_FLOOR = 1e-13

MIN_POINTS = 8


# ============================================================
# Input discovery
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


def find_default_summary():

    commit = current_commit()

    # --------------------------------------------------------
    # Preferred case:
    # results generated from the current HEAD while clean.
    #
    # Deliberately DO NOT append "_dirty".
    # Analysis scripts should be able to inspect completed
    # production even when the working tree later changes.
    # --------------------------------------------------------

    if commit is not None:

        candidate = (
            RESULTS_DIR
            / f"final_{commit}"
            / "photocenter_small_xi_tE150"
            / "summary_photocenter_small_xi.npz"
        )

        if candidate.exists():
            return candidate


    # --------------------------------------------------------
    # Fallback:
    # locate the most recently modified clean production.
    # --------------------------------------------------------

    candidates = list(
        RESULTS_DIR.glob(
            "final_*/photocenter_small_xi_tE150/"
            "summary_photocenter_small_xi.npz"
        )
    )

    candidates = [
        path
        for path in candidates
        if "_dirty" not in path.parts[-3]
    ]

    if not candidates:

        raise FileNotFoundError(
            "Could not find any "
            "summary_photocenter_small_xi.npz "
            f"below {RESULTS_DIR}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


# ============================================================
# Helpers
# ============================================================

def fit_power_law(
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
        & (x >= XI_MIN)
        & (x <= XI_MAX)
        & (y > D_FLOOR)
    )


    n_valid = np.count_nonzero(
        valid
    )


    if n_valid < MIN_POINTS:

        return (
            np.nan,
            np.nan,
            n_valid,
        )


    lx = np.log10(
        x[valid]
    )

    ly = np.log10(
        y[valid]
    )


    slope, intercept = np.polyfit(
        lx,
        ly,
        1,
    )


    predicted = (
        slope * lx
        + intercept
    )


    residual = (
        ly
        - predicted
    )


    rms = np.sqrt(
        np.mean(
            residual**2
        )
    )


    return (
        float(slope),
        float(rms),
        n_valid,
    )


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=Path,
    default=None,
    help=(
        "Explicit path to "
        "summary_photocenter_small_xi.npz"
    ),
)

args = parser.parse_args()


# ============================================================
# Locate input
# ============================================================

if args.input is not None:

    filename = (
        args.input
        .expanduser()
        .resolve()
    )

else:

    filename = (
        find_default_summary()
        .resolve()
    )


if not filename.exists():

    raise FileNotFoundError(
        filename
    )


# ============================================================
# Load
# ============================================================

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


    commit = str(
        d["code_commit"].item()
    )


    objective = str(
        d["fit_objective"].item()
    )


# ============================================================
# Basic validation
# ============================================================

print()

print(
    "=" * 88
)

print(
    "PHOTOCENTER SMALL-XI ANALYSIS"
)

print(
    "=" * 88
)


print(
    "file       =",
    filename,
)

print(
    "commit     =",
    commit,
)

print(
    "objective  =",
    objective,
)

print(
    "families   =",
    families,
)

print(
    "shape D    =",
    D.shape,
)

print(
    "success    =",
    np.count_nonzero(success),
    "/",
    success.size,
)

print(
    "xi range   =",
    xi.min(),
    "->",
    xi.max(),
)

print(
    "fit range  =",
    XI_MIN,
    "->",
    XI_MAX,
)

print(
    "=" * 88
)


# ============================================================
# Fit alpha(qM)
#
# D ~ (xi/u0)^alpha
# ============================================================

alpha = np.full(
    (
        len(families),
        len(qM),
    ),
    np.nan,
)


scatter = np.full_like(
    alpha,
    np.nan,
)


n_used = np.zeros_like(
    alpha,
    dtype=int,
)


for i_family in range(
    len(families)
):


    for i_q in range(
        len(qM)
    ):


        y = D[
            i_family,
            i_q,
            :,
        ]


        (
            alpha[
                i_family,
                i_q,
            ],
            scatter[
                i_family,
                i_q,
            ],
            n_used[
                i_family,
                i_q,
            ],
        ) = fit_power_law(
            xi,
            y,
        )


# ============================================================
# Representative qM values
# ============================================================

q_targets = [
    1e-3,
    1e-2,
    1e-1,
    0.3,
    1.0,
]


print()

print(
    "=" * 88
)

print(
    "REPRESENTATIVE POWER-LAW SLOPES"
)

print(
    "=" * 88
)


print(
    "family"
    "                  qM"
    "          alpha"
    "       scatter[dex]"
    "    N"
)


print(
    "-" * 88
)


for i_family, family in enumerate(
    families
):


    for target in q_targets:


        i_q = int(
            np.argmin(
                np.abs(
                    np.log10(qM)
                    - np.log10(target)
                )
            )
        )


        print(
            f"{family:22s} "
            f"{qM[i_q]:10.4e} "
            f"{alpha[i_family, i_q]:12.5f} "
            f"{scatter[i_family, i_q]:14.5e} "
            f"{n_used[i_family, i_q]:4d}"
        )


    print()


# ============================================================
# Global summaries
# ============================================================

print()

print(
    "=" * 88
)

print(
    "GLOBAL SLOPE SUMMARY"
)

print(
    "=" * 88
)


for i_family, family in enumerate(
    families
):


    a = alpha[
        i_family
    ]


    valid = np.isfinite(
        a
    )


    if not np.any(
        valid
    ):

        print(
            family,
            ": no valid slopes",
        )

        continue


    print(
        f"{family:22s}: "
        f"N={np.count_nonzero(valid):3d}  "
        f"median={np.nanmedian(a):.4f}  "
        f"p16={np.nanpercentile(a, 16):.4f}  "
        f"p84={np.nanpercentile(a, 84):.4f}"
    )


# ============================================================
# qM >= 1e-2
# ============================================================

print()

print(
    "=" * 88
)

print(
    "SLOPES FOR qM >= 1e-2"
)

print(
    "=" * 88
)


q_mask = (
    qM >= 1e-2
)


for i_family, family in enumerate(
    families
):


    a = alpha[
        i_family,
        q_mask,
    ]


    valid = np.isfinite(
        a
    )


    if not np.any(
        valid
    ):

        continue


    print(
        f"{family:22s}: "
        f"median={np.nanmedian(a):.4f}  "
        f"p16={np.nanpercentile(a, 16):.4f}  "
        f"p84={np.nanpercentile(a, 84):.4f}"
    )


# ============================================================
# Additional consistency check:
# compare D at the smallest xi
# ============================================================

print()

print(
    "=" * 88
)

print(
    "SMALLEST-XI VALUES"
)

print(
    "=" * 88
)


i_xi = 0


for i_family, family in enumerate(
    families
):


    values = D[
        i_family,
        :,
        i_xi,
    ]


    finite = np.isfinite(
        values
    )


    print(
        f"{family:22s}: "
        f"xi/u0={xi[i_xi]:.3e}, "
        f"D median={np.nanmedian(values[finite]):.6e}, "
        f"D min={np.nanmin(values[finite]):.6e}, "
        f"D max={np.nanmax(values[finite]):.6e}"
    )


# ============================================================
# Save diagnostic
# ============================================================

outfile = (
    filename.parent
    / "photocenter_powerlaw_slopes.npz"
)


np.savez_compressed(

    outfile,

    family_names=families,

    qM_grid=qM,

    xi_over_u0_grid=xi,

    alpha=alpha,

    scatter_dex=scatter,

    n_used=n_used,

    xi_fit_min=np.float64(
        XI_MIN
    ),

    xi_fit_max=np.float64(
        XI_MAX
    ),

    D_floor=np.float64(
        D_FLOOR
    ),

    source_summary=np.array(
        filename.name
    ),

    code_commit=np.array(
        commit
    ),
)


print()

print(
    "Saved:",
    outfile,
)
