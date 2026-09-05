#!/usr/bin/env python3

"""
Validation of the intrinsic BSPL -> PSPL projection.

Two independent questions are tested:

1. MULTI-START ROBUSTNESS
   Is the Nelder-Mead solution obtained from the true barycentric
   (t0, u0, tE) stable against substantially different initial guesses?

2. TIME-RESOLUTION CONVERGENCE
   Is D_BSPL-PSPL stable when the uniform temporal grid is refined?

The close-approach case is also compared against a non-uniform
high-resolution reference grid because narrow point-source excursions
can be much shorter than u0*tE.

This script is diagnostic only. It does not overwrite production data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize


# ============================================================
# Repository root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from binary_source.analysis import roman_bspl_pspl_asimov as roman


# ============================================================
# Constants matching the intrinsic production setup
# ============================================================

T0 = 50.0
TE = 150.0

M_TOTAL = 3.0
REHAT_AU = 5.0

WINDOW_TE = 3.5

N_TIME_PRODUCTION = 10_000

THETA = 0.0
PHI = 0.0
INCLINATION = np.pi / 2.0


# ============================================================
# Validation configuration
# ============================================================

TIME_LEVELS_DEFAULT = [
    5_000,
    10_000,
    20_000,
    40_000,
    80_000,
]

TIME_LEVELS_FULL = [
    5_000,
    10_000,
    20_000,
    40_000,
    80_000,
    160_000,
]


# Adaptive close-approach grid.
#
# This location comes from the targeted finite-source sanity check:
#
#     (t_close - t0) / tE ~= 0.11984243
#
# We do NOT assume the whole map has such a feature. This is only
# the targeted extreme close-approach diagnostic.
CLOSE_X_CENTER = 0.11984243

ADAPTIVE_HALF_WIDTH_TE = 0.01
ADAPTIVE_N_LOCAL = 40_000
ADAPTIVE_N_BASE = 10_000


# ============================================================
# Data classes
# ============================================================

@dataclass
class Case:
    key: str
    description: str

    u0: float
    P_days: float

    q_mass: float
    qflux: float

    # Optional production reference.
    D_stored: float | None = None

    # Whether to run the time-resolution sequence.
    time_resolution: bool = False

    # Whether to construct the special adaptive grid.
    adaptive_reference: bool = False


# ============================================================
# Numerical utilities
# ============================================================

def trapz(y, x):
    """
    Compatibility wrapper.
    """

    if hasattr(np, "trapezoid"):
        return np.trapezoid(
            y,
            x,
        )

    return np.trapz(
        y,
        x,
    )


def pspl_magnification(
    t,
    t0,
    u0,
    tE,
):
    """
    Standard point-source point-lens magnification.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    t0 = float(t0)
    u0 = float(u0)
    tE = float(tE)

    if (
        not np.isfinite(t0)
        or not np.isfinite(u0)
        or not np.isfinite(tE)
        or tE <= 0.0
    ):
        return np.full_like(
            t,
            np.nan,
            dtype=float,
        )

    tau = (
        (t - t0)
        / tE
    )

    u = np.sqrt(
        u0**2
        + tau**2
    )

    # Avoid a numerical singularity at exactly u=0.
    u = np.maximum(
        u,
        1.0e-15,
    )

    return (
        (u**2 + 2.0)
        /
        (
            u
            * np.sqrt(
                u**2 + 4.0
            )
        )
    )


def intrinsic_objective(
    params,
    t,
    A_truth,
):
    """
    Exact numerator of the intrinsic metric:

        J = integral [A_BSPL - A_PSPL]^2 dt.

    This is the quantity minimized in the production intrinsic fit.
    """

    t0_fit, u0_fit, tE_fit = np.asarray(
        params,
        dtype=float,
    )

    # Broad sanity guard. Nelder-Mead itself is unconstrained.
    if (
        not np.isfinite(t0_fit)
        or not np.isfinite(u0_fit)
        or not np.isfinite(tE_fit)
        or tE_fit <= 0.0
        or tE_fit > 100.0 * TE
        or abs(u0_fit) > 100.0
        or abs(t0_fit - T0) > 100.0 * TE
    ):
        return 1.0e300

    A_fit = pspl_magnification(
        t=t,
        t0=t0_fit,
        u0=u0_fit,
        tE=tE_fit,
    )

    if np.any(
        ~np.isfinite(A_fit)
    ):
        return 1.0e300

    residual = (
        A_truth
        - A_fit
    )

    return float(
        trapz(
            residual**2,
            t,
        )
    )


def compute_D(
    J,
    t,
    A_truth,
):
    """
    Normalized intrinsic mismatch.
    """

    denominator = float(
        trapz(
            (
                A_truth
                - 1.0
            ) ** 2,
            t,
        )
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return np.nan

    return float(
        np.sqrt(
            max(
                float(J),
                0.0,
            )
            / denominator
        )
    )


# ============================================================
# Truth generation
# ============================================================

def make_uniform_time_grid(
    n_time,
):
    return np.linspace(
        T0 - WINDOW_TE * TE,
        T0 + WINDOW_TE * TE,
        int(n_time),
    )


def make_adaptive_close_grid():
    """
    Uniform production grid + dense local refinement around the
    known extreme close approach.

    Trapezoidal integration naturally handles the non-uniform spacing.
    """

    base = make_uniform_time_grid(
        ADAPTIVE_N_BASE
    )

    t_center = (
        T0
        + CLOSE_X_CENTER * TE
    )

    half_width = (
        ADAPTIVE_HALF_WIDTH_TE
        * TE
    )

    local = np.linspace(
        t_center - half_width,
        t_center + half_width,
        ADAPTIVE_N_LOCAL,
    )

    t = np.unique(
        np.concatenate(
            [
                base,
                local,
            ]
        )
    )

    t.sort()

    return t


def bspl_truth(
    case,
    t,
):
    """
    Use the same pyLIMA BSPL construction used by the Roman
    Asimov analysis.
    """

    truth = (
        roman.bspl_truth_magnification(
            t=t,
            t0_true=T0,
            u0_true=case.u0,
            tE_true=TE,
            P_days=case.P_days,
            q_mass=case.q_mass,
            qflux=case.qflux,
            Mtot_Msun=M_TOTAL,
            rEhat_AU=REHAT_AU,
            theta=THETA,
            phi=PHI,
            inclination=INCLINATION,
        )
    )

    A = np.asarray(
        truth["A_bspl"],
        dtype=float,
    ).reshape(-1)

    if len(A) != len(t):
        raise RuntimeError(
            f"{case.key}: truth length mismatch."
        )

    if np.any(
        ~np.isfinite(A)
    ):
        raise RuntimeError(
            f"{case.key}: non-finite truth magnification."
        )

    return A


# ============================================================
# Production-grid reference
# ============================================================

def load_one_luminous_reference(
    u0_target,
    P_target,
):
    """
    Find the nearest exact production node in:

        results/scan_many_tE_200x200/scan_u0_tE150

    Used only to verify that this diagnostic reproduces the
    stored production D values.
    """

    directory = (
        ROOT
        / "results"
        / "scan_many_tE_200x200"
        / "scan_u0_tE150"
    )

    files = sorted(
        directory.glob(
            "scan_kepler_u0_*.npz"
        )
    )

    if not files:
        return None

    best_record = None

    for fn in files:

        with np.load(
            fn,
            allow_pickle=False,
        ) as d:

            truth = np.asarray(
                d["truth"],
                dtype=float,
            )

            this_u0 = float(
                truth[1]
            )

            distance_u0 = abs(
                np.log10(this_u0)
                - np.log10(u0_target)
            )

            if (
                best_record is None
                or distance_u0
                < best_record[
                    "distance_u0"
                ]
            ):

                P_grid = np.asarray(
                    d["P_grid"],
                    dtype=float,
                )

                D_grid = np.asarray(
                    d["D"],
                    dtype=float,
                )

                success = np.asarray(
                    d["SUCCESS"],
                    dtype=bool,
                )

                ip = int(
                    np.argmin(
                        np.abs(
                            np.log10(P_grid)
                            - np.log10(P_target)
                        )
                    )
                )

                best_record = {
                    "file": str(fn),
                    "distance_u0": (
                        distance_u0
                    ),
                    "u0": this_u0,
                    "P_days": float(
                        P_grid[ip]
                    ),
                    "D": float(
                        D_grid[ip]
                    ),
                    "success": bool(
                        success[ip]
                    ),
                    "iP": ip,
                }

    return best_record


# ============================================================
# Case construction
# ============================================================

def build_cases():
    """
    Representative cases chosen to stress different parts of
    the intrinsic analysis.
    """

    # --------------------------------------------------------
    # Exact one-luminous production nodes
    # --------------------------------------------------------

    hidden_ref = (
        load_one_luminous_reference(
            u0_target=0.01,
            P_target=100_000.0,
        )
    )

    peak_ref = (
        load_one_luminous_reference(
            u0_target=0.01,
            P_target=TE,
        )
    )

    if hidden_ref is None:
        raise RuntimeError(
            "Could not load one-luminous production grid."
        )

    if peak_ref is None:
        raise RuntimeError(
            "Could not load one-luminous production grid."
        )


    cases = [

        # ====================================================
        # Very strongly degenerate, long-period case
        # ====================================================

        Case(
            key="one_luminous_hidden_long",
            description=(
                "One luminous source; very long period; "
                "strong intrinsic degeneracy"
            ),
            u0=hidden_ref["u0"],
            P_days=hidden_ref["P_days"],
            q_mass=0.5,
            qflux=0.0,
            D_stored=hidden_ref["D"],
            time_resolution=False,
        ),


        # ====================================================
        # Small-u0 case with P ~ tE
        #
        # Useful for testing resolution of the narrow ordinary
        # high-magnification peak.
        # ====================================================

        Case(
            key="one_luminous_small_u0",
            description=(
                "One luminous source; u0=0.01; P approximately tE"
            ),
            u0=peak_ref["u0"],
            P_days=peak_ref["P_days"],
            q_mass=0.5,
            qflux=0.0,
            D_stored=peak_ref["D"],
            time_resolution=True,
        ),


        # ====================================================
        # Photocenter-cancellation configuration
        # ====================================================

        Case(
            key="photocenter_cancel",
            description=(
                "Two luminous sources on qf=qM cancellation locus"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.5,
            qflux=0.5,
            D_stored=None,
            time_resolution=True,
        ),


        # ====================================================
        # Finite-source control geometry
        # ====================================================

        Case(
            key="close_control",
            description=(
                "Two luminous sources; ordinary control trajectory"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.30,
            qflux=0.01,
            D_stored=None,
            time_resolution=False,
        ),


        # ====================================================
        # Extreme close approach from the finite-source check
        # ====================================================

        Case(
            key="close_extreme",
            description=(
                "Extreme point-source close approach"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.09,
            qflux=0.01,
            D_stored=None,
            time_resolution=True,
            adaptive_reference=True,
        ),
    ]

    return cases


# ============================================================
# Multi-start initial conditions
# ============================================================

def build_multistarts(
    case,
    n_random,
    seed,
):
    """
    Truth start + deliberately broad deterministic starts +
    reproducible random perturbations.

    The production start is always the first entry.
    """

    truth = np.array(
        [
            T0,
            case.u0,
            TE,
        ],
        dtype=float,
    )

    starts = [
        (
            "truth",
            truth,
        ),

        (
            "dt0_-0.25tE",
            np.array(
                [
                    T0 - 0.25 * TE,
                    case.u0,
                    TE,
                ]
            ),
        ),

        (
            "dt0_+0.25tE",
            np.array(
                [
                    T0 + 0.25 * TE,
                    case.u0,
                    TE,
                ]
            ),
        ),

        (
            "dt0_-0.75tE",
            np.array(
                [
                    T0 - 0.75 * TE,
                    case.u0,
                    TE,
                ]
            ),
        ),

        (
            "dt0_+0.75tE",
            np.array(
                [
                    T0 + 0.75 * TE,
                    case.u0,
                    TE,
                ]
            ),
        ),

        (
            "u0_x0.25",
            np.array(
                [
                    T0,
                    max(
                        case.u0 * 0.25,
                        1.0e-5,
                    ),
                    TE,
                ]
            ),
        ),

        (
            "u0_x4",
            np.array(
                [
                    T0,
                    case.u0 * 4.0,
                    TE,
                ]
            ),
        ),

        (
            "tE_x0.5",
            np.array(
                [
                    T0,
                    case.u0,
                    0.5 * TE,
                ]
            ),
        ),

        (
            "tE_x2",
            np.array(
                [
                    T0,
                    case.u0,
                    2.0 * TE,
                ]
            ),
        ),

        (
            "combined_1",
            np.array(
                [
                    T0 - 0.35 * TE,
                    max(
                        case.u0 * 0.5,
                        1.0e-5,
                    ),
                    1.5 * TE,
                ]
            ),
        ),

        (
            "combined_2",
            np.array(
                [
                    T0 + 0.35 * TE,
                    case.u0 * 2.0,
                    0.7 * TE,
                ]
            ),
        ),
    ]


    # --------------------------------------------------------
    # Random starts
    # --------------------------------------------------------

    rng = np.random.default_rng(
        int(seed)
    )

    for i in range(
        int(n_random)
    ):

        dt0 = (
            rng.uniform(
                -1.0,
                1.0,
            )
            * TE
        )

        u_factor = 10.0 ** (
            rng.uniform(
                -1.0,
                1.0,
            )
        )

        tE_factor = 10.0 ** (
            rng.uniform(
                -0.5,
                0.5,
            )
        )

        start = np.array(
            [
                T0 + dt0,
                max(
                    case.u0 * u_factor,
                    1.0e-6,
                ),
                TE * tE_factor,
            ],
            dtype=float,
        )

        starts.append(
            (
                f"random_{i:02d}",
                start,
            )
        )


    return starts


# ============================================================
# Fit one start
# ============================================================

def fit_one_start(
    t,
    A_truth,
    x0,
    maxiter,
):
    result = minimize(
        intrinsic_objective,
        x0=np.asarray(
            x0,
            dtype=float,
        ),
        args=(
            t,
            A_truth,
        ),
        method="Nelder-Mead",
        options={
            "maxiter": int(
                maxiter
            ),
            "xatol": 1.0e-10,
            "fatol": 1.0e-12,
        },
    )

    J = float(
        intrinsic_objective(
            result.x,
            t,
            A_truth,
        )
    )

    D = compute_D(
        J=J,
        t=t,
        A_truth=A_truth,
    )

    return {
        "success": bool(
            result.success
        ),
        "message": str(
            result.message
        ),
        "nfev": int(
            result.nfev
        ),
        "nit": int(
            result.nit
        ),
        "J": J,
        "D": D,
        "best_t0": float(
            result.x[0]
        ),
        "best_u0": float(
            result.x[1]
        ),
        "best_tE": float(
            result.x[2]
        ),
    }


# ============================================================
# Multi-start validation
# ============================================================

def run_multistart_case(
    case,
    n_random,
    seed,
    maxiter,
):
    print()
    print("=" * 100)
    print("MULTI-START")
    print(case.key)
    print(case.description)
    print("=" * 100)

    t = make_uniform_time_grid(
        N_TIME_PRODUCTION
    )

    A_truth = bspl_truth(
        case,
        t,
    )

    starts = build_multistarts(
        case=case,
        n_random=n_random,
        seed=seed,
    )

    rows = []

    for i, (
        start_name,
        x0,
    ) in enumerate(starts):

        fit = fit_one_start(
            t=t,
            A_truth=A_truth,
            x0=x0,
            maxiter=maxiter,
        )

        row = {
            "case": case.key,
            "description": (
                case.description
            ),
            "start_index": i,
            "start_name": (
                start_name
            ),
            "start_t0": float(
                x0[0]
            ),
            "start_u0": float(
                x0[1]
            ),
            "start_tE": float(
                x0[2]
            ),
            **fit,
        }

        rows.append(
            row
        )

        print(
            f"{i:02d} "
            f"{start_name:18s} "
            f"D={fit['D']:.12e} "
            f"J={fit['J']:.12e} "
            f"success={fit['success']} "
            f"best="
            f"({fit['best_t0']:.8g}, "
            f"{fit['best_u0']:.8g}, "
            f"{fit['best_tE']:.8g})"
        )


    df = pd.DataFrame(
        rows
    )

    finite = (
        np.isfinite(
            df["D"].to_numpy()
        )
    )

    if not np.any(finite):
        raise RuntimeError(
            f"No finite fits for {case.key}"
        )

    best_index = int(
        df.loc[
            finite,
            "D",
        ].idxmin()
    )

    truth_row = df[
        df["start_name"]
        == "truth"
    ].iloc[0]

    best_row = df.loc[
        best_index
    ]

    D_truth = float(
        truth_row["D"]
    )

    D_best = float(
        best_row["D"]
    )

    absolute_improvement = (
        D_truth
        - D_best
    )

    fractional_improvement = (
        absolute_improvement
        / max(
            abs(D_truth),
            1.0e-300,
        )
    )


    print()
    print("-" * 100)
    print("MULTI-START SUMMARY")
    print("-" * 100)

    print(
        f"D truth-start = "
        f"{D_truth:.12e}"
    )

    print(
        f"D best        = "
        f"{D_best:.12e}"
    )

    print(
        f"best start    = "
        f"{best_row['start_name']}"
    )

    print(
        f"absolute improvement = "
        f"{absolute_improvement:.6e}"
    )

    print(
        f"fractional improvement = "
        f"{fractional_improvement:.6e}"
    )

    if case.D_stored is not None:

        production_relative_difference = (
            abs(
                D_truth
                - case.D_stored
            )
            / max(
                abs(case.D_stored),
                1.0e-300,
            )
        )

        print()
        print(
            f"D stored production = "
            f"{case.D_stored:.12e}"
        )

        print(
            f"truth-start vs stored "
            f"relative difference = "
            f"{production_relative_difference:.6e}"
        )

    else:

        production_relative_difference = (
            np.nan
        )


    # --------------------------------------------------------
    # Diagnostic classification
    # --------------------------------------------------------

    # We avoid a hard pass/fail based only on relative changes
    # when D itself is extremely small.
    stable = (
        abs(
            absolute_improvement
        )
        <= 1.0e-8
        or fractional_improvement
        <= 1.0e-3
    )

    print()
    print(
        "MULTISTART_STABLE =",
        bool(stable),
    )

    print("-" * 100)


    summary = {
        "case": case.key,
        "D_truth_start": (
            D_truth
        ),
        "D_best_multistart": (
            D_best
        ),
        "best_start_name": str(
            best_row[
                "start_name"
            ]
        ),
        "absolute_improvement": float(
            absolute_improvement
        ),
        "fractional_improvement": float(
            fractional_improvement
        ),
        "D_stored": (
            np.nan
            if case.D_stored is None
            else float(
                case.D_stored
            )
        ),
        "production_relative_difference": float(
            production_relative_difference
        ),
        "stable": bool(
            stable
        ),
    }

    return (
        df,
        summary,
        np.array(
            [
                best_row[
                    "best_t0"
                ],
                best_row[
                    "best_u0"
                ],
                best_row[
                    "best_tE"
                ],
            ],
            dtype=float,
        ),
    )


# ============================================================
# Small start set for the time-resolution study
# ============================================================

def resolution_starts(
    case,
    previous_best=None,
):
    starts = [
        (
            "truth",
            np.array(
                [
                    T0,
                    case.u0,
                    TE,
                ],
                dtype=float,
            ),
        ),

        (
            "perturbed_t0",
            np.array(
                [
                    T0 + 0.25 * TE,
                    case.u0,
                    TE,
                ],
                dtype=float,
            ),
        ),

        (
            "perturbed_u0",
            np.array(
                [
                    T0,
                    max(
                        0.5 * case.u0,
                        1.0e-6,
                    ),
                    TE,
                ],
                dtype=float,
            ),
        ),

        (
            "perturbed_tE",
            np.array(
                [
                    T0,
                    case.u0,
                    1.5 * TE,
                ],
                dtype=float,
            ),
        ),
    ]

    if previous_best is not None:

        starts.insert(
            0,
            (
                "previous_best",
                np.asarray(
                    previous_best,
                    dtype=float,
                ),
            ),
        )

    return starts


# ============================================================
# Fit best of a small start set
# ============================================================

def fit_best_small_multistart(
    case,
    t,
    A_truth,
    previous_best,
    maxiter,
):
    rows = []

    for name, x0 in resolution_starts(
        case,
        previous_best=previous_best,
    ):

        fit = fit_one_start(
            t=t,
            A_truth=A_truth,
            x0=x0,
            maxiter=maxiter,
        )

        rows.append(
            {
                "start_name": name,
                **fit,
            }
        )

    finite_rows = [
        row
        for row in rows
        if np.isfinite(
            row["D"]
        )
    ]

    if not finite_rows:
        raise RuntimeError(
            f"No finite resolution fits "
            f"for {case.key}"
        )

    best = min(
        finite_rows,
        key=lambda row: row["D"],
    )

    best_params = np.array(
        [
            best["best_t0"],
            best["best_u0"],
            best["best_tE"],
        ],
        dtype=float,
    )

    return (
        best,
        best_params,
    )


# ============================================================
# Time-resolution convergence
# ============================================================

def run_time_resolution_case(
    case,
    time_levels,
    maxiter,
):
    print()
    print("=" * 100)
    print("TIME RESOLUTION")
    print(case.key)
    print(case.description)
    print("=" * 100)

    rows = []

    previous_best = None

    for n_time in time_levels:

        t = make_uniform_time_grid(
            n_time
        )

        A_truth = bspl_truth(
            case,
            t,
        )

        fit, previous_best = (
            fit_best_small_multistart(
                case=case,
                t=t,
                A_truth=A_truth,
                previous_best=(
                    previous_best
                ),
                maxiter=maxiter,
            )
        )

        dt = float(
            np.median(
                np.diff(t)
            )
        )

        row = {
            "case": case.key,
            "grid_type": "uniform",
            "n_time": int(
                len(t)
            ),
            "dt_median_days": dt,
            "D": float(
                fit["D"]
            ),
            "J": float(
                fit["J"]
            ),
            "best_t0": float(
                fit["best_t0"]
            ),
            "best_u0": float(
                fit["best_u0"]
            ),
            "best_tE": float(
                fit["best_tE"]
            ),
            "best_start": str(
                fit["start_name"]
            ),
        }

        rows.append(
            row
        )

        print(
            f"N={len(t):7d} "
            f"dt={dt:.8f} d "
            f"D={fit['D']:.12e} "
            f"best="
            f"({fit['best_t0']:.8g}, "
            f"{fit['best_u0']:.8g}, "
            f"{fit['best_tE']:.8g})"
        )


    # ========================================================
    # Adaptive reference for the extreme close approach
    # ========================================================

    if case.adaptive_reference:

        t = make_adaptive_close_grid()

        A_truth = bspl_truth(
            case,
            t,
        )

        fit, best_params = (
            fit_best_small_multistart(
                case=case,
                t=t,
                A_truth=A_truth,
                previous_best=(
                    previous_best
                ),
                maxiter=maxiter,
            )
        )

        dt_all = np.diff(t)

        t_center = (
            T0
            + CLOSE_X_CENTER * TE
        )

        local_mask = (
            np.abs(
                t - t_center
            )
            <= (
                ADAPTIVE_HALF_WIDTH_TE
                * TE
            )
        )

        t_local = t[
            local_mask
        ]

        dt_local = float(
            np.median(
                np.diff(
                    t_local
                )
            )
        )

        row = {
            "case": case.key,
            "grid_type": "adaptive",
            "n_time": int(
                len(t)
            ),
            "dt_median_days": float(
                np.median(
                    dt_all
                )
            ),
            "dt_local_days": (
                dt_local
            ),
            "D": float(
                fit["D"]
            ),
            "J": float(
                fit["J"]
            ),
            "best_t0": float(
                fit["best_t0"]
            ),
            "best_u0": float(
                fit["best_u0"]
            ),
            "best_tE": float(
                fit["best_tE"]
            ),
            "best_start": str(
                fit["start_name"]
            ),
        }

        rows.append(
            row
        )

        print()
        print(
            "ADAPTIVE:"
        )

        print(
            f"N={len(t):7d} "
            f"local dt={dt_local:.10f} d "
            f"D={fit['D']:.12e}"
        )


    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # Compare uniform grids with the finest available reference
    # ========================================================

    adaptive_rows = df[
        df["grid_type"]
        == "adaptive"
    ]

    if len(adaptive_rows) > 0:

        D_reference = float(
            adaptive_rows.iloc[
                -1
            ]["D"]
        )

        reference_name = (
            "adaptive"
        )

    else:

        uniform = df[
            df["grid_type"]
            == "uniform"
        ]

        finest_index = (
            uniform[
                "n_time"
            ]
            .astype(int)
            .idxmax()
        )

        D_reference = float(
            uniform.loc[
                finest_index,
                "D",
            ]
        )

        reference_name = (
            f"uniform_N"
            f"{int(uniform.loc[finest_index, 'n_time'])}"
        )


    df[
        "D_reference"
    ] = D_reference

    df[
        "reference_name"
    ] = reference_name

    df[
        "relative_D_error"
    ] = (
        np.abs(
            df["D"]
            - D_reference
        )
        / max(
            abs(D_reference),
            1.0e-300,
        )
    )


    print()
    print("-" * 100)

    print(
        "REFERENCE =",
        reference_name,
    )

    print(
        "D_reference =",
        f"{D_reference:.12e}",
    )

    print()

    for _, row in df.iterrows():

        print(
            f"{row['grid_type']:8s} "
            f"N={int(row['n_time']):7d} "
            f"D={row['D']:.12e} "
            f"rel.err="
            f"{row['relative_D_error']:.6e}"
        )

    print("-" * 100)

    return df



# ============================================================
# Adaptive convergence for the extreme close approach
# ============================================================

def adaptive_convergence_close_extreme(
    case,
    maxiter=50_000,
):
    """
    Check convergence of the locally refined time grid used for
    the extreme point-source close-approach configuration.

    Two quantities are varied independently:

    1. local temporal resolution:
         N_local = 10k, 20k, 40k, 80k

    2. half-width of the refined region:
         0.005, 0.010, 0.020 tE

    The global base grid remains fixed at ADAPTIVE_N_BASE points
    over t0 +/- WINDOW_TE*tE.

    Each configuration is independently re-fitted with the same
    small multi-start strategy used in the uniform-resolution test.
    """

    local_ns = [
        10_000,
        20_000,
        40_000,
        80_000,
    ]

    half_widths_te = [
        0.005,
        0.010,
        0.020,
    ]

    rows = []

    t_center = (
        T0
        + CLOSE_X_CENTER * TE
    )

    print()
    print("=" * 100)
    print("ADAPTIVE CLOSE-APPROACH CONVERGENCE")
    print(case.key)
    print(case.description)
    print("=" * 100)

    print(
        f"close-approach center: "
        f"(t_close-t0)/tE = {CLOSE_X_CENTER:.8f}"
    )

    print(
        f"t_close = {t_center:.10f} d"
    )

    print()


    # --------------------------------------------------------
    # Each half-width is treated independently.
    #
    # Within one half-width, the solution from the previous
    # resolution is included as an additional start for the
    # next resolution.
    # --------------------------------------------------------

    for half_width_te in half_widths_te:

        previous_best = None

        half_width_days = (
            half_width_te
            * TE
        )

        print("-" * 100)

        print(
            f"half-width = "
            f"{half_width_te:.5f} tE "
            f"= {half_width_days:.8f} d"
        )

        print("-" * 100)

        for n_local in local_ns:

            base = make_uniform_time_grid(
                ADAPTIVE_N_BASE
            )

            local = np.linspace(
                t_center - half_width_days,
                t_center + half_width_days,
                int(n_local),
            )

            t = np.unique(
                np.concatenate(
                    [
                        base,
                        local,
                    ]
                )
            )

            t.sort()


            # ------------------------------------------------
            # Construct BSPL truth on this exact grid
            # ------------------------------------------------

            A_truth = bspl_truth(
                case,
                t,
            )


            # ------------------------------------------------
            # Re-optimize PSPL
            # ------------------------------------------------

            fit, previous_best = (
                fit_best_small_multistart(
                    case=case,
                    t=t,
                    A_truth=A_truth,
                    previous_best=previous_best,
                    maxiter=maxiter,
                )
            )


            # ------------------------------------------------
            # Actual local spacing
            # ------------------------------------------------

            local_mask = (
                np.abs(
                    t - t_center
                )
                <= half_width_days
            )

            t_local = t[
                local_mask
            ]

            dt_local = float(
                np.median(
                    np.diff(
                        t_local
                    )
                )
            )


            row = {
                "case": case.key,
                "half_width_te": float(
                    half_width_te
                ),
                "half_width_days": float(
                    half_width_days
                ),
                "n_local": int(
                    n_local
                ),
                "n_total": int(
                    len(t)
                ),
                "dt_local_days": float(
                    dt_local
                ),
                "D": float(
                    fit["D"]
                ),
                "J": float(
                    fit["J"]
                ),
                "best_t0": float(
                    fit["best_t0"]
                ),
                "best_u0": float(
                    fit["best_u0"]
                ),
                "best_tE": float(
                    fit["best_tE"]
                ),
                "best_start": str(
                    fit["start_name"]
                ),
            }

            rows.append(
                row
            )


            print(
                f"Nlocal={n_local:7d} "
                f"Ntotal={len(t):7d} "
                f"dtlocal={dt_local:.10e} d "
                f"D={fit['D']:.12e} "
                f"best="
                f"({fit['best_t0']:.8g}, "
                f"{fit['best_u0']:.8g}, "
                f"{fit['best_tE']:.8g})"
            )


    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # Convergence within each refinement width
    #
    # For each width, use N_local=80000 as its own reference.
    # ========================================================

    df[
        "D_reference_same_width"
    ] = np.nan

    df[
        "relative_error_same_width"
    ] = np.nan


    for half_width_te in half_widths_te:

        mask = (
            df["half_width_te"]
            == half_width_te
        )

        this = df[
            mask
        ]

        ref_index = (
            this[
                "n_local"
            ]
            .astype(int)
            .idxmax()
        )

        D_ref = float(
            df.loc[
                ref_index,
                "D",
            ]
        )

        df.loc[
            mask,
            "D_reference_same_width",
        ] = D_ref

        df.loc[
            mask,
            "relative_error_same_width",
        ] = (
            np.abs(
                df.loc[
                    mask,
                    "D",
                ]
                - D_ref
            )
            / max(
                abs(D_ref),
                1.0e-300,
            )
        )


    # ========================================================
    # Global reference
    #
    # Use the widest and finest locally refined grid:
    #
    #   half-width = 0.02 tE
    #   N_local    = 80000
    #
    # This checks both local sampling and sufficient refined
    # temporal coverage.
    # ========================================================

    global_reference_mask = (
        (
            df["half_width_te"]
            == max(
                half_widths_te
            )
        )
        & (
            df["n_local"]
            == max(
                local_ns
            )
        )
    )

    if (
        global_reference_mask.sum()
        != 1
    ):
        raise RuntimeError(
            "Could not identify unique adaptive global reference."
        )

    D_global_reference = float(
        df.loc[
            global_reference_mask,
            "D",
        ].iloc[0]
    )

    df[
        "D_global_reference"
    ] = D_global_reference

    df[
        "relative_error_global"
    ] = (
        np.abs(
            df["D"]
            - D_global_reference
        )
        / max(
            abs(D_global_reference),
            1.0e-300,
        )
    )


    # ========================================================
    # Printed summary
    # ========================================================

    print()
    print("=" * 100)
    print("ADAPTIVE CONVERGENCE SUMMARY")
    print("=" * 100)

    print(
        "Global reference:"
    )

    print(
        f"  half-width = "
        f"{max(half_widths_te):.5f} tE"
    )

    print(
        f"  N_local    = "
        f"{max(local_ns)}"
    )

    print(
        f"  D_ref      = "
        f"{D_global_reference:.12e}"
    )

    print()


    for half_width_te in half_widths_te:

        this = (
            df[
                df["half_width_te"]
                == half_width_te
            ]
            .sort_values(
                "n_local"
            )
        )

        print(
            f"half-width = "
            f"{half_width_te:.5f} tE"
        )

        for _, row in this.iterrows():

            print(
                f"  "
                f"Nlocal={int(row['n_local']):7d} "
                f"D={row['D']:.12e} "
                f"err(same width)="
                f"{row['relative_error_same_width']:.6e} "
                f"err(global)="
                f"{row['relative_error_global']:.6e}"
            )

        print()


    # ========================================================
    # Specific comparison of the old adaptive setup
    #
    # Old setup:
    #   half-width = 0.01 tE
    #   N_local    = 40000
    # ========================================================

    old_mask = (
        np.isclose(
            df["half_width_te"],
            0.010,
        )
        & (
            df["n_local"]
            == 40_000
        )
    )

    if old_mask.sum() == 1:

        old_row = df[
            old_mask
        ].iloc[0]

        print("-" * 100)

        print(
            "Previous adaptive setup "
            "(half-width=0.01 tE, Nlocal=40000):"
        )

        print(
            f"  D = "
            f"{old_row['D']:.12e}"
        )

        print(
            f"  relative difference to "
            f"global reference = "
            f"{old_row['relative_error_global']:.6e}"
        )

        print("-" * 100)


    return df



# ============================================================
# Overall report
# ============================================================

def print_overall_summary(
    multistart_summary,
    time_df,
):
    print()
    print()
    print("#" * 100)
    print("OVERALL VALIDATION SUMMARY")
    print("#" * 100)

    if multistart_summary:

        ms = pd.DataFrame(
            multistart_summary
        )

        print()
        print("MULTI-START")
        print("-" * 100)

        for _, row in ms.iterrows():

            print(
                f"{row['case']:30s} "
                f"D_truth="
                f"{row['D_truth_start']:.6e} "
                f"D_best="
                f"{row['D_best_multistart']:.6e} "
                f"improvement="
                f"{row['fractional_improvement']:.3e} "
                f"stable="
                f"{bool(row['stable'])}"
            )


    if time_df is not None and len(
        time_df
    ) > 0:

        print()
        print("TIME RESOLUTION AT PRODUCTION N=10000")
        print("-" * 100)

        for case_name in sorted(
            time_df["case"].unique()
        ):

            this = time_df[
                time_df["case"]
                == case_name
            ]

            production = this[
                (
                    this["grid_type"]
                    == "uniform"
                )
                & (
                    this["n_time"]
                    == N_TIME_PRODUCTION
                )
            ]

            if len(production) == 0:
                continue

            row = production.iloc[
                0
            ]

            print(
                f"{case_name:30s} "
                f"D_10k="
                f"{row['D']:.6e} "
                f"rel.err.to.reference="
                f"{row['relative_D_error']:.3e} "
                f"reference="
                f"{row['reference_name']}"
            )

    print()
    print("#" * 100)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "multistart",
            "time",
            "adaptive",
        ],
        default="all",
    )

    parser.add_argument(
        "--n-random",
        type=int,
        default=12,
        help=(
            "Number of reproducible random multi-start "
            "initial conditions per case."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--full-time",
        action="store_true",
        help=(
            "Extend the uniform convergence test to "
            "N_time=160000."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "validation_intrinsic_multistart"
        ),
    )

    args = parser.parse_args()


    outdir = (
        ROOT
        / args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    cases = build_cases()

    print("=" * 100)
    print("INTRINSIC MULTI-START + TIME-RESOLUTION VALIDATION")
    print("=" * 100)

    print(
        "production time window:",
        f"[t0-{WINDOW_TE}tE, "
        f"t0+{WINDOW_TE}tE]",
    )

    print(
        "production N_time:",
        N_TIME_PRODUCTION,
    )

    print(
        "n cases:",
        len(cases),
    )

    print()

    for case in cases:

        print(
            f"{case.key:30s} "
            f"u0={case.u0:.8g} "
            f"P={case.P_days:.8g} d "
            f"P/tE={case.P_days/TE:.8g} "
            f"qM={case.q_mass:.6g} "
            f"qf={case.qflux:.6g} "
            f"Dstored={case.D_stored}"
        )


    # ========================================================
    # Results containers
    # ========================================================

    df_adaptive = pd.DataFrame()


    # ========================================================
    # Multi-start
    # ========================================================

    all_multistart_rows = []
    multistart_summary = []
    best_multistart_params = {}

    if args.mode in (
        "all",
        "multistart",
    ):

        for icase, case in enumerate(
            cases
        ):

            df, summary, best_params = (
                run_multistart_case(
                    case=case,
                    n_random=args.n_random,
                    seed=(
                        args.seed
                        + 1000 * icase
                    ),
                    maxiter=args.maxiter,
                )
            )

            all_multistart_rows.append(
                df
            )

            multistart_summary.append(
                summary
            )

            best_multistart_params[
                case.key
            ] = best_params


        df_multistart = pd.concat(
            all_multistart_rows,
            ignore_index=True,
        )

        df_multistart.to_csv(
            outdir
            / "multistart_all_starts.csv",
            index=False,
        )

        pd.DataFrame(
            multistart_summary
        ).to_csv(
            outdir
            / "multistart_summary.csv",
            index=False,
        )


    # ========================================================
    # Time convergence
    # ========================================================

    all_time_rows = []

    if args.mode in (
        "all",
        "time",
    ):

        time_levels = (
            TIME_LEVELS_FULL
            if args.full_time
            else TIME_LEVELS_DEFAULT
        )

        for case in cases:

            if not case.time_resolution:
                continue

            df = (
                run_time_resolution_case(
                    case=case,
                    time_levels=time_levels,
                    maxiter=args.maxiter,
                )
            )

            all_time_rows.append(
                df
            )


        if all_time_rows:

            df_time = pd.concat(
                all_time_rows,
                ignore_index=True,
            )

            df_time.to_csv(
                outdir
                / "time_resolution_summary.csv",
                index=False,
            )

        else:

            df_time = pd.DataFrame()


    else:

        df_time = pd.DataFrame()



    # ========================================================
    # Adaptive close-approach convergence
    # ========================================================

    if args.mode == "adaptive":

        close_cases = [
            case
            for case in cases
            if case.key
            == "close_extreme"
        ]

        if len(close_cases) != 1:
            raise RuntimeError(
                "Could not identify unique close_extreme case."
            )

        close_case = close_cases[
            0
        ]

        df_adaptive = (
            adaptive_convergence_close_extreme(
                case=close_case,
                maxiter=args.maxiter,
            )
        )

        df_adaptive.to_csv(
            outdir
            / "adaptive_close_convergence.csv",
            index=False,
        )


    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "t0": T0,
        "tE": TE,
        "window_tE": WINDOW_TE,
        "production_n_time": (
            N_TIME_PRODUCTION
        ),
        "M_total_Msun": M_TOTAL,
        "rEhat_AU": REHAT_AU,
        "theta": THETA,
        "phi": PHI,
        "inclination": (
            INCLINATION
        ),
        "n_random_multistarts": (
            args.n_random
        ),
        "random_seed": (
            args.seed
        ),
        "maxiter": (
            args.maxiter
        ),
        "time_levels": (
            TIME_LEVELS_FULL
            if args.full_time
            else TIME_LEVELS_DEFAULT
        ),
        "adaptive_close_x_center": (
            CLOSE_X_CENTER
        ),
        "adaptive_half_width_tE": (
            ADAPTIVE_HALF_WIDTH_TE
        ),
        "adaptive_n_base": (
            ADAPTIVE_N_BASE
        ),
        "adaptive_n_local": (
            ADAPTIVE_N_LOCAL
        ),
        "cases": [
            asdict(case)
            for case in cases
        ],
    }

    (
        outdir
        / "validation_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


    # ========================================================
    # Final report
    # ========================================================

    print_overall_summary(
        multistart_summary=(
            multistart_summary
        ),
        time_df=df_time,
    )


    print()
    print("Saved:")
    print(
        " ",
        outdir
        / "validation_metadata.json",
    )

    if args.mode in (
        "all",
        "multistart",
    ):

        print(
            " ",
            outdir
            / "multistart_all_starts.csv",
        )

        print(
            " ",
            outdir
            / "multistart_summary.csv",
        )

    if (
        args.mode
        in (
            "all",
            "time",
        )
        and len(df_time) > 0
    ):

        print(
            " ",
            outdir
            / "time_resolution_summary.csv",
        )


    if (
        args.mode
        == "adaptive"
        and len(df_adaptive) > 0
    ):

        print(
            " ",
            outdir
            / "adaptive_close_convergence.csv",
        )


if __name__ == "__main__":
    main()
