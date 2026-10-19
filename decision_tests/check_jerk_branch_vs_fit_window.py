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
# Configuration
# ============================================================

TE_TRUE = 30.0
U0_TRUE = 0.01

PIEN_TRUE = -0.24
PIEE_TRUE = +0.02

WINDOWS_DAYS = [
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    105.0,
]

# Same resolution scale as the dense control.
MAX_WINDOW = max(WINDOWS_DAYS)
N_FULL = 20001

LOCAL_WINDOW = 0.40
N_LOCAL = 9001

OUTDIR = Path(
    "/tmp/jerk_branch_vs_fit_window"
)

# Previously recovered full-window secondary branch.
GLOBAL_REFERENCE = {
    "t0": 2461849.008719672,
    "u0": 0.00790642487084667,
    "tE": 29.990144606115795,
    "piEN": -0.049942335065522404,
    "piEE": +0.024190626569186493,
}


# ============================================================
# Utilities
# ============================================================

def trapezoid_sqrt_weights(t):
    """
    sqrt(weights) such that

        sum [sqrt(w_i) r_i]^2

    approximates

        integral r(t)^2 dt.

    Normalize by mean(w) only for numerical conditioning.
    This does not change the minimizer.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    if len(t) < 2:
        raise ValueError(
            "Need at least two time samples."
        )

    w = np.empty_like(t)

    w[0] = (
        t[1]
        - t[0]
    ) / 2.0

    w[-1] = (
        t[-1]
        - t[-2]
    ) / 2.0

    if len(t) > 2:
        w[1:-1] = (
            t[2:]
            - t[:-2]
        ) / 2.0

    if np.any(
        w <= 0.0
    ):
        raise RuntimeError(
            "Non-positive trapezoid weights."
        )

    w = (
        w
        / np.mean(w)
    )

    return np.sqrt(w)


def trapz(y, x):
    try:
        return float(
            np.trapezoid(
                y,
                x,
            )
        )
    except AttributeError:
        return float(
            np.trapz(
                y,
                x,
            )
        )


def intrinsic_metrics(
    t,
    A_truth,
    A_fit,
):
    residual = (
        np.asarray(A_fit)
        - np.asarray(A_truth)
    )

    numerator = trapz(
        residual**2,
        t,
    )

    denominator = trapz(
        (
            np.asarray(A_truth)
            - 1.0
        )**2,
        t,
    )

    D = np.sqrt(
        numerator
        / denominator
    )

    rms = float(
        np.sqrt(
            np.mean(
                residual**2
            )
        )
    )

    return D, rms


def make_parameters(
    evaluator,
    *,
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    values = {
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),
        "piEN": float(piEN),
        "piEE": float(piEE),
    }

    vector = [
        values.get(
            name,
            None,
        )
        for name
        in evaluator.model.model_dictionnary.keys()
    ]

    return (
        evaluator.model.compute_pyLIMA_parameters(
            vector
        )
    )


def trajectory_u2(
    evaluator,
    *,
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    p = make_parameters(
        evaluator,
        t0=t0,
        u0=u0,
        tE=tE,
        piEN=piEN,
        piEE=piEE,
    )

    tr = (
        evaluator.model.sources_trajectory(
            evaluator.telescope,
            p,
            data_type="photometry",
        )
    )

    x = np.asarray(
        tr[0],
        dtype=float,
    )

    y = np.asarray(
        tr[1],
        dtype=float,
    )

    return (
        x*x
        + y*y
    )


def local_coefficients(
    t,
    u2,
    t0,
    window,
):
    dt = (
        np.asarray(t)
        - float(t0)
    )

    use = (
        np.abs(dt)
        <= float(window)
    )

    s = (
        dt[use]
        / float(window)
    )

    a = (
        np.polynomial.polynomial.polyfit(
            s,
            np.asarray(u2)[use],
            deg=6,
        )
    )

    C = np.array(
        [
            a[k]
            * (
                TE_TRUE
                / float(window)
            )**k
            for k
            in range(
                len(a)
            )
        ],
        dtype=float,
    )

    return C


# ============================================================
# Exact local u0=0 jerk root using pyLIMA itself
# ============================================================

def exact_local_jerk_root(
    t0_ref,
):
    t = np.linspace(
        t0_ref - LOCAL_WINDOW,
        t0_ref + LOCAL_WINDOW,
        N_LOCAL,
    )

    evaluator = (
        par.PSPLParallaxEvaluator(
            t=t,
            t0_reference=t0_ref,
        )
    )

    delta = np.asarray(
        evaluator.telescope.deltas_positions[
            "photometry"
        ],
        dtype=float,
    )

    dt = (
        t
        - t0_ref
    )

    s = (
        dt
        / LOCAL_WINDOW
    )

    coeffs = []

    for component in range(2):

        a = (
            np.polynomial.polynomial.polyfit(
                s,
                delta[component],
                deg=6,
            )
        )

        c = np.array(
            [
                a[k]
                / LOCAL_WINDOW**k
                for k
                in range(
                    len(a)
                )
            ],
            dtype=float,
        )

        coeffs.append(c)

    coeffs = np.asarray(
        coeffs
    )

    alpha_vec = (
        2.0
        * coeffs[:, 2]
    )

    alpha = float(
        np.linalg.norm(
            alpha_vec
        )
    )

    epar = (
        alpha_vec
        / alpha
    )

    eperp = np.array(
        [
            -epar[1],
            +epar[0],
        ]
    )

    truth_pi = np.array(
        [
            PIEN_TRUE,
            PIEE_TRUE,
        ],
        dtype=float,
    )

    ppar = float(
        np.dot(
            truth_pi,
            epar,
        )
    )

    pperp_truth = float(
        np.dot(
            truth_pi,
            eperp,
        )
    )

    u2_truth = trajectory_u2(
        evaluator,
        t0=t0_ref,
        u0=0.0,
        tE=TE_TRUE,
        piEN=PIEN_TRUE,
        piEE=PIEE_TRUE,
    )

    C_truth = local_coefficients(
        t,
        u2_truth,
        t0_ref,
        LOCAL_WINDOW,
    )

    pperp_grid = np.linspace(
        -0.45,
        +0.20,
        81,
    )

    C4 = []

    for pp in pperp_grid:

        pi_NE = (
            ppar
            * epar
            +
            pp
            * eperp
        )

        u2 = trajectory_u2(
            evaluator,
            t0=t0_ref,
            u0=0.0,
            tE=TE_TRUE,
            piEN=pi_NE[0],
            piEE=pi_NE[1],
        )

        C = local_coefficients(
            t,
            u2,
            t0_ref,
            LOCAL_WINDOW,
        )

        C4.append(
            C[4]
        )

    C4 = np.asarray(
        C4
    )

    q = np.polyfit(
        pperp_grid,
        C4 - C_truth[4],
        deg=2,
    )

    roots = np.roots(
        q
    )

    roots = [
        float(r.real)
        for r in roots
        if abs(r.imag) < 1e-8
    ]

    if len(roots) != 2:
        raise RuntimeError(
            f"Expected 2 real C4 roots: {roots}"
        )

    second = max(
        roots,
        key=lambda x:
            abs(
                x
                - pperp_truth
            ),
    )

    second_NE = (
        ppar
        * epar
        +
        second
        * eperp
    )

    return {
        "piEN":
            float(
                second_NE[0]
            ),

        "piEE":
            float(
                second_NE[1]
            ),

        "pperp":
            second,

        "ppar":
            ppar,

        "epar":
            epar,

        "eperp":
            eperp,
    }


# ============================================================
# Fit parametrization
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
            float(t0)
            - float(t0_ref),

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


# ============================================================
# One fit
# ============================================================

def fit_one(
    *,
    evaluator,
    t,
    A_truth,
    t0_ref,
    start,
    start_name,
):
    sqrt_w = (
        trapezoid_sqrt_weights(
            t
        )
    )

    # Diagnostic branch domain.
    #
    # The injected truth has piEN=-0.24 and is deliberately
    # outside this interval. We are following only the
    # secondary same-sign branch.
    lower = np.array(
        [
            -1.0,
            1e-6,
            np.log(
                25.0
                / TE_TRUE
            ),
            -0.15,
            -0.10,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            +1.0,
            0.05,
            np.log(
                35.0
                / TE_TRUE
            ),
            +0.05,
            +0.10,
        ],
        dtype=float,
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
                A
                - A_truth
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
        max_nfev=3000,
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

    active = np.asarray(
        opt.active_mask,
        dtype=int,
    )

    # u0 approaching zero is acceptable.
    bad_other_bound = bool(
        np.any(
            active[
                [
                    0,
                    2,
                    3,
                    4,
                ]
            ]
            != 0
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
        < 1e-2
        and not bad_other_bound
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

        "active_mask":
            ",".join(
                str(int(x))
                for x
                in active
            ),

        "D":
            D,

        "RMS":
            rms,

        **p,
    }


def parameter_distance(
    a,
    b,
):
    """
    Dimensionless distance used only to follow a continuous
    branch from one fit window to the next.
    """

    return float(
        np.sqrt(
            (
                (
                    a["t0"]
                    - b["t0"]
                )
                / 0.10
            )**2
            +
            (
                (
                    a["u0"]
                    - b["u0"]
                )
                / 0.01
            )**2
            +
            (
                (
                    a["tE"]
                    - b["tE"]
                )
                / 1.0
            )**2
            +
            (
                (
                    a["piEN"]
                    - b["piEN"]
                )
                / 0.05
            )**2
            +
            (
                (
                    a["piEE"]
                    - b["piEE"]
                )
                / 0.02
            )**2
        )
    )


# ============================================================
# Setup
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


local_root = (
    exact_local_jerk_root(
        t0_ref
    )
)


LOCAL_START = {
    "t0":
        t0_ref,

    "u0":
        U0_TRUE,

    "tE":
        TE_TRUE,

    "piEN":
        local_root[
            "piEN"
        ],

    "piEE":
        local_root[
            "piEE"
        ],
}


MID_START = {
    "t0":
        t0_ref,

    "u0":
        0.009,

    "tE":
        TE_TRUE,

    "piEN":
        -0.025,

    "piEE":
        +0.025,
}


print()
print("=" * 110)
print("JERK BRANCH VS FIT WINDOW")
print("=" * 110)

print(
    "truth:",
    f"u0={U0_TRUE}",
    f"tE={TE_TRUE}",
    f"pi=({PIEN_TRUE:+.9f},{PIEE_TRUE:+.9f})",
)

print(
    "exact local C4 root:",
    f"pi=({local_root['piEN']:+.9f},"
    f"{local_root['piEE']:+.9f})",
)

print(
    "known full-window branch:",
    f"pi=({GLOBAL_REFERENCE['piEN']:+.9f},"
    f"{GLOBAL_REFERENCE['piEE']:+.9f})",
)


# ============================================================
# Window sweep
# ============================================================

candidate_rows = []
summary_rows = []

previous_tracked = dict(
    LOCAL_START
)


for window in WINDOWS_DAYS:

    # Keep approximately constant dense sampling over the
    # full +/-105 d interval, but instantiate only points
    # needed by this fit.
    dt_full = (
        2.0
        * MAX_WINDOW
        / (
            N_FULL - 1
        )
    )

    n_half = max(
        50,
        int(
            np.ceil(
                window
                / dt_full
            )
        ),
    )

    # Odd number -> t0 is exactly in the grid.
    t = np.linspace(
        t0_ref - window,
        t0_ref + window,
        2*n_half + 1,
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


    # --------------------------------------------------------
    # Fixed-parameter diagnostics
    # --------------------------------------------------------

    A_local_fixed = evaluator.magnification(
        t0=LOCAL_START["t0"],
        u0=LOCAL_START["u0"],
        tE=LOCAL_START["tE"],
        piEN=LOCAL_START["piEN"],
        piEE=LOCAL_START["piEE"],
    )

    D_local_fixed, _ = intrinsic_metrics(
        t,
        A_truth,
        A_local_fixed,
    )


    A_global_fixed = evaluator.magnification(
        t0=GLOBAL_REFERENCE["t0"],
        u0=GLOBAL_REFERENCE["u0"],
        tE=GLOBAL_REFERENCE["tE"],
        piEN=GLOBAL_REFERENCE["piEN"],
        piEE=GLOBAL_REFERENCE["piEE"],
    )

    D_global_fixed, _ = intrinsic_metrics(
        t,
        A_truth,
        A_global_fixed,
    )


    # --------------------------------------------------------
    # Multiple starts
    # --------------------------------------------------------

    starts = [
        (
            "previous",
            dict(
                previous_tracked
            ),
        ),
        (
            "local_C4",
            dict(
                LOCAL_START
            ),
        ),
        (
            "midpoint",
            dict(
                MID_START
            ),
        ),
        (
            "global_reference",
            dict(
                GLOBAL_REFERENCE
            ),
        ),
    ]


    candidates = []

    print()
    print("=" * 110)
    print(
        f"WINDOW +/- {window:g} d "
        f"(N={len(t)})"
    )
    print("=" * 110)

    print(
        "fixed D:",
        f"local={D_local_fixed:.9e}",
        f"global={D_global_fixed:.9e}",
        flush=True,
    )


    for (
        start_name,
        start,
    ) in starts:

        result = fit_one(
            evaluator=evaluator,
            t=t,
            A_truth=A_truth,
            t0_ref=t0_ref,
            start=start,
            start_name=start_name,
        )

        result[
            "window_days"
        ] = window

        result[
            "N_time"
        ] = len(t)

        result[
            "dist_to_local"
        ] = parameter_distance(
            result,
            LOCAL_START,
        )

        result[
            "dist_to_global"
        ] = parameter_distance(
            result,
            GLOBAL_REFERENCE,
        )

        candidates.append(
            result
        )

        candidate_rows.append(
            result
        )

        print(
            f"{start_name:16s} "
            f"good={str(result['strict_good']):5s} "
            f"D={result['D']:.9e} "
            f"u0={result['u0']:+.8f} "
            f"tE={result['tE']:.8f} "
            f"piEN={result['piEN']:+.8f} "
            f"piEE={result['piEE']:+.8f} "
            f"nfev={result['nfev']:4d} "
            f"opt={result['optimality']:.3e}"
        )


    good = [
        r
        for r in candidates
        if r[
            "strict_good"
        ]
    ]


    if not good:

        raise RuntimeError(
            f"No trustworthy solution "
            f"for window={window} d"
        )


    # Lowest objective in the allowed secondary-branch domain.
    best = min(
        good,
        key=lambda r:
            r["D"],
    )


    # Follow the solution nearest to the previous member.
    tracked = min(
        good,
        key=lambda r:
            parameter_distance(
                r,
                previous_tracked,
            ),
    )


    previous_tracked = {
        key:
            tracked[key]
        for key in [
            "t0",
            "u0",
            "tE",
            "piEN",
            "piEE",
        ]
    }


    tracked_pi_distance_local = float(
        np.hypot(
            tracked["piEN"]
            - local_root[
                "piEN"
            ],
            tracked["piEE"]
            - local_root[
                "piEE"
            ],
        )
    )

    tracked_pi_distance_global = float(
        np.hypot(
            tracked["piEN"]
            - GLOBAL_REFERENCE[
                "piEN"
            ],
            tracked["piEE"]
            - GLOBAL_REFERENCE[
                "piEE"
            ],
        )
    )


    best_pi_distance_local = float(
        np.hypot(
            best["piEN"]
            - local_root[
                "piEN"
            ],
            best["piEE"]
            - local_root[
                "piEE"
            ],
        )
    )

    best_pi_distance_global = float(
        np.hypot(
            best["piEN"]
            - GLOBAL_REFERENCE[
                "piEN"
            ],
            best["piEE"]
            - GLOBAL_REFERENCE[
                "piEE"
            ],
        )
    )


    summary_rows.append(
        {
            "window_days":
                window,

            "N_time":
                len(t),

            "D_local_fixed":
                D_local_fixed,

            "D_global_fixed":
                D_global_fixed,

            "tracked_start":
                tracked[
                    "start_name"
                ],

            "tracked_D":
                tracked[
                    "D"
                ],

            "tracked_t0":
                tracked[
                    "t0"
                ],

            "tracked_u0":
                tracked[
                    "u0"
                ],

            "tracked_tE":
                tracked[
                    "tE"
                ],

            "tracked_piEN":
                tracked[
                    "piEN"
                ],

            "tracked_piEE":
                tracked[
                    "piEE"
                ],

            "tracked_optimality":
                tracked[
                    "optimality"
                ],

            "tracked_dist_local_pi":
                tracked_pi_distance_local,

            "tracked_dist_global_pi":
                tracked_pi_distance_global,

            "best_start":
                best[
                    "start_name"
                ],

            "best_D":
                best[
                    "D"
                ],

            "best_t0":
                best[
                    "t0"
                ],

            "best_u0":
                best[
                    "u0"
                ],

            "best_tE":
                best[
                    "tE"
                ],

            "best_piEN":
                best[
                    "piEN"
                ],

            "best_piEE":
                best[
                    "piEE"
                ],

            "best_optimality":
                best[
                    "optimality"
                ],

            "best_dist_local_pi":
                best_pi_distance_local,

            "best_dist_global_pi":
                best_pi_distance_global,

            "tracked_is_best":
                (
                    tracked
                    is best
                ),
        }
    )


    print()
    print(
        "TRACKED:",
        f"D={tracked['D']:.9e}",
        f"pi=({tracked['piEN']:+.8f},"
        f"{tracked['piEE']:+.8f})",
        f"d_local={tracked_pi_distance_local:.6e}",
        f"d_global={tracked_pi_distance_global:.6e}",
    )

    print(
        "BEST:   ",
        f"D={best['D']:.9e}",
        f"pi=({best['piEN']:+.8f},"
        f"{best['piEE']:+.8f})",
        f"d_local={best_pi_distance_local:.6e}",
        f"d_global={best_pi_distance_global:.6e}",
    )


# ============================================================
# Save/report
# ============================================================

candidates_df = pd.DataFrame(
    candidate_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


candidates_path = (
    OUTDIR
    / "all_candidates.csv"
)

summary_path = (
    OUTDIR
    / "window_summary.csv"
)


candidates_df.to_csv(
    candidates_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 110)
print("FINAL WINDOW EVOLUTION")
print("=" * 110)

print(
    summary_df[
        [
            "window_days",
            "tracked_D",
            "tracked_u0",
            "tracked_tE",
            "tracked_piEN",
            "tracked_piEE",
            "tracked_optimality",
            "tracked_dist_local_pi",
            "tracked_dist_global_pi",
            "best_D",
            "best_piEN",
            "best_piEE",
            "tracked_is_best",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "local C4 root:",
    local_root[
        "piEN"
    ],
    local_root[
        "piEE"
    ],
)

print(
    "global reference:",
    GLOBAL_REFERENCE[
        "piEN"
    ],
    GLOBAL_REFERENCE[
        "piEE"
    ],
)

print()
print(
    "saved:",
    summary_path
)

print(
    "saved:",
    candidates_path
)
