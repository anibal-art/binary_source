#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_parallax as par,
)


# ============================================================
# Science configuration
# ============================================================

TE_TRUE = 30.0
U0_TRUE = 0.01

PIEN_TRUE = -0.24
PIEE_TRUE = +0.02

WINDOWS_FORWARD = [
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
    60.0,
    75.0,
    90.0,
    105.0,
]

WINDOW_BACKWARD = 8.0
ANCHOR_WINDOW = 10.0

MAX_WINDOW = 105.0

# Same dense sampling as previous 20001-point +/-105 d test.
N_FULL = 20001

OUTDIR = Path(
    "/tmp/jerk_branch_window_continuation"
)


# ============================================================
# Reference points already established
# ============================================================

# Interior W=10 d minimum from previous experiment.
ANCHOR_GUESS = {
    "u0": 0.00756981,
    "tE": 29.98858273,
    "piEN": -0.01939711,
    "piEE": +0.02475214,
}

# Exact local u0=0 C4 root from direct trajectory test.
LOCAL_C4_ROOT = np.array(
    [
        -0.0019446907407941493,
        +0.026352926698214995,
    ],
    dtype=float,
)

# Full +/-105 d secondary minimum.
GLOBAL_REFERENCE = np.array(
    [
        -0.049942335065522404,
        +0.024190626569186493,
    ],
    dtype=float,
)


# ============================================================
# Diagnostic continuation trust region
# ============================================================

TRUST_DT0 = 0.05       # days
TRUST_U0 = 0.004
TRUST_TE = 0.50        # days

TRUST_PIEN = 0.040
TRUST_PIEE = 0.020

# Reject an optimum if it lies this close to a trust bound,
# expressed as a fraction of total allowed interval.
BOUND_FRACTION_TOL = 5e-4

OPTIMALITY_MAX = 1e-2


# ============================================================
# Numerical utilities
# ============================================================

def trapz(y, x):
    try:
        return float(
            np.trapezoid(y, x)
        )
    except AttributeError:
        return float(
            np.trapz(y, x)
        )


def trapezoid_sqrt_weights(t):
    t = np.asarray(
        t,
        dtype=float,
    )

    if len(t) < 2:
        raise RuntimeError(
            "Need >=2 samples."
        )

    w = np.empty_like(t)

    w[0] = (
        t[1] - t[0]
    ) / 2.0

    w[-1] = (
        t[-1] - t[-2]
    ) / 2.0

    if len(t) > 2:
        w[1:-1] = (
            t[2:] - t[:-2]
        ) / 2.0

    if np.any(w <= 0):
        raise RuntimeError(
            "Invalid trapezoid weights."
        )

    # Pure scaling; does not change minimizer.
    w /= np.mean(w)

    return np.sqrt(w)


def intrinsic_metrics(
    t,
    A_truth,
    A_fit,
):
    r = (
        np.asarray(A_fit)
        - np.asarray(A_truth)
    )

    num = trapz(
        r**2,
        t,
    )

    den = trapz(
        (
            np.asarray(A_truth)
            - 1.0
        )**2,
        t,
    )

    D = float(
        np.sqrt(
            num / den
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                r**2
            )
        )
    )

    return D, rms


# ============================================================
# Fit parametrization
#
# x =
# [dt0, u0, log(tE/TE_TRUE), piEN, piEE]
# ============================================================

def encode(
    *,
    t0,
    u0,
    tE,
    piEN,
    piEE,
    t0_ref,
):
    return np.array(
        [
            float(t0) - float(t0_ref),
            float(u0),
            np.log(
                float(tE)
                / TE_TRUE
            ),
            float(piEN),
            float(piEE),
        ],
        dtype=float,
    )


def decode(
    x,
    t0_ref,
):
    return {
        "t0":
            float(
                t0_ref
                + x[0]
            ),

        "u0":
            float(
                x[1]
            ),

        "tE":
            float(
                TE_TRUE
                * np.exp(
                    x[2]
                )
            ),

        "piEN":
            float(
                x[3]
            ),

        "piEE":
            float(
                x[4]
            ),
    }


def make_trust_bounds(
    center,
    t0_ref,
):
    dt0_center = (
        float(center["t0"])
        - float(t0_ref)
    )

    lower_dt0 = max(
        -1.0,
        dt0_center - TRUST_DT0,
    )

    upper_dt0 = min(
        +1.0,
        dt0_center + TRUST_DT0,
    )

    lower_u0 = max(
        1e-6,
        float(center["u0"])
        - TRUST_U0,
    )

    upper_u0 = min(
        0.05,
        float(center["u0"])
        + TRUST_U0,
    )

    lower_tE = max(
        25.0,
        float(center["tE"])
        - TRUST_TE,
    )

    upper_tE = min(
        35.0,
        float(center["tE"])
        + TRUST_TE,
    )

    lower_piEN = max(
        -0.15,
        float(center["piEN"])
        - TRUST_PIEN,
    )

    upper_piEN = min(
        +0.05,
        float(center["piEN"])
        + TRUST_PIEN,
    )

    lower_piEE = max(
        -0.10,
        float(center["piEE"])
        - TRUST_PIEE,
    )

    upper_piEE = min(
        +0.10,
        float(center["piEE"])
        + TRUST_PIEE,
    )

    lower = np.array(
        [
            lower_dt0,
            lower_u0,
            np.log(
                lower_tE
                / TE_TRUE
            ),
            lower_piEN,
            lower_piEE,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            upper_dt0,
            upper_u0,
            np.log(
                upper_tE
                / TE_TRUE
            ),
            upper_piEN,
            upper_piEE,
        ],
        dtype=float,
    )

    return lower, upper


def explicit_bound_diagnostics(
    x,
    lower,
    upper,
):
    width = (
        upper - lower
    )

    frac_lower = (
        x - lower
    ) / width

    frac_upper = (
        upper - x
    ) / width

    margins = np.minimum(
        frac_lower,
        frac_upper,
    )

    names = [
        "dt0",
        "u0",
        "log_tE",
        "piEN",
        "piEE",
    ]

    near = {
        name:
            bool(
                margins[i]
                < BOUND_FRACTION_TOL
            )
        for i, name
        in enumerate(names)
    }

    return {
        "margins":
            margins,

        "near":
            near,

        "any_near":
            any(
                near.values()
            ),
    }


def pi_distance(
    row,
    ref,
):
    return float(
        np.hypot(
            float(row["piEN"])
            - float(ref[0]),
            float(row["piEE"])
            - float(ref[1]),
        )
    )


def pi_step(
    a,
    b,
):
    return float(
        np.hypot(
            float(a["piEN"])
            - float(b["piEN"]),
            float(a["piEE"])
            - float(b["piEE"]),
        )
    )


# ============================================================
# Time grid
# ============================================================

DT_FULL = (
    2.0
    * MAX_WINDOW
    / (
        N_FULL - 1
    )
)


def make_times(
    t0_ref,
    window,
):
    n_half = max(
        100,
        int(
            np.ceil(
                float(window)
                / DT_FULL
            )
        ),
    )

    return np.linspace(
        t0_ref - float(window),
        t0_ref + float(window),
        2*n_half + 1,
    )


# ============================================================
# One local fit
# ============================================================

def fit_one(
    *,
    evaluator,
    t,
    A_truth,
    t0_ref,
    center,
    start,
    start_name,
):
    lower, upper = (
        make_trust_bounds(
            center,
            t0_ref,
        )
    )

    sqrt_w = (
        trapezoid_sqrt_weights(
            t
        )
    )

    x0 = encode(
        t0=start["t0"],
        u0=start["u0"],
        tE=start["tE"],
        piEN=start["piEN"],
        piEE=start["piEE"],
        t0_ref=t0_ref,
    )

    x0 = np.clip(
        x0,
        lower + 1e-10,
        upper - 1e-10,
    )

    def residual(
        x,
    ):
        p = decode(
            x,
            t0_ref,
        )

        A = evaluator.magnification(
            t0=p["t0"],
            u0=p["u0"],
            tE=p["tE"],
            piEN=p["piEN"],
            piEE=p["piEE"],
        )

        return (
            sqrt_w
            * (
                A - A_truth
            )
        )

    opt = least_squares(
        residual,
        x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        x_scale="jac",
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=2500,
    )

    p = decode(
        opt.x,
        t0_ref,
    )

    A_fit = evaluator.magnification(
        t0=p["t0"],
        u0=p["u0"],
        tE=p["tE"],
        piEN=p["piEN"],
        piEE=p["piEE"],
    )

    D, rms = intrinsic_metrics(
        t,
        A_truth,
        A_fit,
    )

    bound = (
        explicit_bound_diagnostics(
            opt.x,
            lower,
            upper,
        )
    )

    strict_good = (
        bool(opt.success)
        and np.isfinite(D)
        and np.isfinite(
            opt.optimality
        )
        and float(
            opt.optimality
        )
        < OPTIMALITY_MAX
        and not bound[
            "any_near"
        ]
    )

    return {
        "start_name":
            start_name,

        "success":
            bool(
                opt.success
            ),

        "strict_good":
            bool(
                strict_good
            ),

        "status":
            int(
                opt.status
            ),

        "nfev":
            int(
                opt.nfev
            ),

        "optimality":
            float(
                opt.optimality
            ),

        "D":
            D,

        "RMS":
            rms,

        "near_any_bound":
            bool(
                bound[
                    "any_near"
                ]
            ),

        "near_dt0_bound":
            bool(
                bound["near"]["dt0"]
            ),

        "near_u0_bound":
            bool(
                bound["near"]["u0"]
            ),

        "near_tE_bound":
            bool(
                bound["near"]["log_tE"]
            ),

        "near_piEN_bound":
            bool(
                bound["near"]["piEN"]
            ),

        "near_piEE_bound":
            bool(
                bound["near"]["piEE"]
            ),

        "min_bound_fraction":
            float(
                np.min(
                    bound[
                        "margins"
                    ]
                )
            ),

        **p,
    }


# ============================================================
# Local starts around previous branch point
# ============================================================

def make_starts(
    previous,
):
    starts = []

    starts.append(
        (
            "previous",
            dict(
                previous
            ),
        )
    )

    perturbations = [
        (
            "piEN_minus",
            -0.010,
            0.0,
        ),
        (
            "piEN_plus",
            +0.010,
            0.0,
        ),
        (
            "piEE_minus",
            0.0,
            -0.003,
        ),
        (
            "piEE_plus",
            0.0,
            +0.003,
        ),
    ]

    for (
        name,
        dN,
        dE,
    ) in perturbations:

        p = dict(
            previous
        )

        p["piEN"] = (
            float(
                p["piEN"]
            )
            + dN
        )

        p["piEE"] = (
            float(
                p["piEE"]
            )
            + dE
        )

        starts.append(
            (
                name,
                p,
            )
        )

    return starts


# ============================================================
# Fit one window while following previous solution
# ============================================================

def run_window(
    *,
    window,
    previous,
    t0_ref,
):
    t = make_times(
        t0_ref,
        window,
    )

    evaluator = (
        par.PSPLParallaxEvaluator(
            t=t,
            t0_reference=t0_ref,
        )
    )

    A_truth = evaluator.magnification(
        t0=t0_ref,
        u0=U0_TRUE,
        tE=TE_TRUE,
        piEN=PIEN_TRUE,
        piEE=PIEE_TRUE,
    )

    results = []

    print()
    print("=" * 116)
    print(
        f"WINDOW +/- {window:g} d "
        f"N={len(t)}"
    )
    print("=" * 116)

    for (
        start_name,
        start,
    ) in make_starts(
        previous
    ):

        row = fit_one(
            evaluator=evaluator,
            t=t,
            A_truth=A_truth,
            t0_ref=t0_ref,
            center=previous,
            start=start,
            start_name=start_name,
        )

        row[
            "window_days"
        ] = float(
            window
        )

        row[
            "N_time"
        ] = int(
            len(t)
        )

        row[
            "step_pi_from_previous"
        ] = pi_step(
            row,
            previous,
        )

        row[
            "dist_local_pi"
        ] = pi_distance(
            row,
            LOCAL_C4_ROOT,
        )

        row[
            "dist_global_pi"
        ] = pi_distance(
            row,
            GLOBAL_REFERENCE,
        )

        results.append(
            row
        )

        print(
            f"{start_name:12s} "
            f"good={str(row['strict_good']):5s} "
            f"D={row['D']:.10e} "
            f"u0={row['u0']:+.8f} "
            f"tE={row['tE']:.8f} "
            f"piEN={row['piEN']:+.8f} "
            f"piEE={row['piEE']:+.8f} "
            f"step={row['step_pi_from_previous']:.4e} "
            f"nfev={row['nfev']:4d} "
            f"opt={row['optimality']:.3e} "
            f"bound={row['near_any_bound']}"
        )

    good = [
        r
        for r in results
        if r[
            "strict_good"
        ]
    ]

    if not good:

        print()
        print(
            "NO TRUSTWORTHY LOCAL SOLUTION"
        )

        raise RuntimeError(
            f"Continuation failed "
            f"at window={window} d."
        )

    # Primary continuation rule:
    # keep the solution reached from the previous branch point.
    tracked = next(
        (
            r
            for r in good
            if r[
                "start_name"
            ]
            == "previous"
        ),
        None,
    )

    # If that particular optimization failed, use the
    # trustworthy candidate nearest to previous solution.
    if tracked is None:

        tracked = min(
            good,
            key=lambda r:
                r[
                    "step_pi_from_previous"
                ],
        )

    # Separate audit: best D among all local perturbations.
    best = min(
        good,
        key=lambda r:
            r["D"],
    )

    branch_separation = pi_step(
        tracked,
        best,
    )

    relative_D_gap = (
        (
            tracked["D"]
            - best["D"]
        )
        / max(
            best["D"],
            1e-30,
        )
    )

    print()
    print(
        "TRACKED:",
        f"D={tracked['D']:.10e}",
        f"pi=({tracked['piEN']:+.8f},"
        f"{tracked['piEE']:+.8f})",
        f"dlocal={tracked['dist_local_pi']:.5e}",
        f"dglobal={tracked['dist_global_pi']:.5e}",
    )

    print(
        "LOCAL BEST:",
        f"D={best['D']:.10e}",
        f"pi=({best['piEN']:+.8f},"
        f"{best['piEE']:+.8f})",
        f"separation={branch_separation:.5e}",
        f"relative_D_gap={relative_D_gap:.5e}",
    )

    if branch_separation > 5e-3:
        print(
            "WARNING: local perturbations found "
            "a distinct competing minimum."
        )

    tracked = dict(
        tracked
    )

    tracked[
        "best_D"
    ] = float(
        best["D"]
    )

    tracked[
        "best_piEN"
    ] = float(
        best["piEN"]
    )

    tracked[
        "best_piEE"
    ] = float(
        best["piEE"]
    )

    tracked[
        "tracked_best_pi_separation"
    ] = float(
        branch_separation
    )

    tracked[
        "tracked_best_relative_D_gap"
    ] = float(
        relative_D_gap
    )

    return (
        tracked,
        results,
    )


# ============================================================
# Main
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


_, t0_ref = (
    base.roman.build_roman_times(
        tE_true=TE_TRUE
    )
)


anchor_initial = {
    "t0":
        t0_ref,

    "u0":
        ANCHOR_GUESS[
            "u0"
        ],

    "tE":
        ANCHOR_GUESS[
            "tE"
        ],

    "piEN":
        ANCHOR_GUESS[
            "piEN"
        ],

    "piEE":
        ANCHOR_GUESS[
            "piEE"
        ],
}


print()
print("=" * 116)
print("FINE CONTINUATION OF INTERIOR PARALLAX BRANCH")
print("=" * 116)

print(
    "truth:",
    f"u0={U0_TRUE}",
    f"tE={TE_TRUE}",
    f"pi=({PIEN_TRUE:+.8f},"
    f"{PIEE_TRUE:+.8f})",
)

print(
    "local C4 reference:",
    LOCAL_C4_ROOT,
)

print(
    "global +/-105 d reference:",
    GLOBAL_REFERENCE,
)


all_candidates = []
tracked_rows = []


# ------------------------------------------------------------
# 1. Refit anchor W=10 d
# ------------------------------------------------------------

anchor, candidates = run_window(
    window=ANCHOR_WINDOW,
    previous=anchor_initial,
    t0_ref=t0_ref,
)

tracked_rows.append(
    anchor
)

all_candidates.extend(
    candidates
)


print()
print(
    "ANCHOR REFINED:",
    f"t0={anchor['t0']:.9f}",
    f"u0={anchor['u0']:.9f}",
    f"tE={anchor['tE']:.9f}",
    f"piEN={anchor['piEN']:+.9f}",
    f"piEE={anchor['piEE']:+.9f}",
)


# ------------------------------------------------------------
# 2. One backward point W=8 d from same W=10 anchor
# ------------------------------------------------------------

backward, candidates = run_window(
    window=WINDOW_BACKWARD,
    previous=anchor,
    t0_ref=t0_ref,
)

tracked_rows.append(
    backward
)

all_candidates.extend(
    candidates
)


# ------------------------------------------------------------
# 3. Forward continuation, reset explicitly to W=10 anchor
# ------------------------------------------------------------

previous = dict(
    anchor
)


for window in WINDOWS_FORWARD:

    tracked, candidates = run_window(
        window=window,
        previous=previous,
        t0_ref=t0_ref,
    )

    tracked_rows.append(
        tracked
    )

    all_candidates.extend(
        candidates
    )

    previous = {
        "t0":
            tracked[
                "t0"
            ],

        "u0":
            tracked[
                "u0"
            ],

        "tE":
            tracked[
                "tE"
            ],

        "piEN":
            tracked[
                "piEN"
            ],

        "piEE":
            tracked[
                "piEE"
            ],
    }


# ============================================================
# Save
# ============================================================

tracked_df = pd.DataFrame(
    tracked_rows
)

tracked_df = (
    tracked_df
    .sort_values(
        "window_days"
    )
    .reset_index(
        drop=True
    )
)

candidate_df = pd.DataFrame(
    all_candidates
)


tracked_path = (
    OUTDIR
    / "tracked_branch.csv"
)

candidate_path = (
    OUTDIR
    / "all_local_candidates.csv"
)


tracked_df.to_csv(
    tracked_path,
    index=False,
)

candidate_df.to_csv(
    candidate_path,
    index=False,
)


# ============================================================
# Final compact report
# ============================================================

print()
print("=" * 116)
print("FINAL CONTINUOUS BRANCH")
print("=" * 116)

cols = [
    "window_days",
    "D",
    "u0",
    "tE",
    "piEN",
    "piEE",
    "optimality",
    "step_pi_from_previous",
    "dist_local_pi",
    "dist_global_pi",
    "tracked_best_pi_separation",
    "tracked_best_relative_D_gap",
]

print(
    tracked_df[
        cols
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Monotonic/path diagnostics
# ------------------------------------------------------------

forward = (
    tracked_df[
        tracked_df[
            "window_days"
        ]
        >= ANCHOR_WINDOW
    ]
    .sort_values(
        "window_days"
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 116)
print("FORWARD STEP DIAGNOSTICS")
print("=" * 116)

for i in range(
    1,
    len(forward),
):

    a = forward.iloc[
        i - 1
    ]

    b = forward.iloc[
        i
    ]

    dpi = float(
        np.hypot(
            b["piEN"]
            - a["piEN"],
            b["piEE"]
            - a["piEE"],
        )
    )

    print(
        f"{a['window_days']:6.1f}"
        f" -> {b['window_days']:6.1f} d : "
        f"dpi={dpi:.6e}   "
        f"piEN {a['piEN']:+.6f}"
        f" -> {b['piEN']:+.6f}   "
        f"piEE {a['piEE']:+.6f}"
        f" -> {b['piEE']:+.6f}"
    )


print()
print(
    "saved:",
    tracked_path
)

print(
    "saved:",
    candidate_path
)
