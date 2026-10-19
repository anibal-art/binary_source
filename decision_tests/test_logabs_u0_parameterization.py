#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from binary_source.analysis import (
    intrinsic_fit_tools as base,
)


# ============================================================
# Same problematic Roman+xallarap configuration
# ============================================================

U0_TRUE = 0.03
TE_TRUE = 100.0
SOURCE_MAG = 19.0
P_OVER_TE = 1.0

PI_BOUND = 40.0

# Lower numerical floor for |u0| in the log parametrization.
# This is deliberately far below the current ~1e-3 solution.
U0_ABS_MIN = 1.0e-8
U0_ABS_MAX = 20.0


# ============================================================
# Load helpers from the validation script
# ============================================================

SCRIPT = Path(
    "decision_tests/"
    "validate_roman_two_sign_policy.py"
).resolve()

spec = importlib.util.spec_from_file_location(
    "validation",
    SCRIPT,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# ============================================================
# Build exact same Asimov dataset
# ============================================================

prep = m.prepare_configuration(
    u0=U0_TRUE,
    tE=TE_TRUE,
    source_mag=SOURCE_MAG,
    period_ratio=P_OVER_TE,
    need_windows=False,
)

truth = prep["truth"]
data = m.asimov_dataset(truth)

flux = np.asarray(
    data["flux"],
    dtype=float,
)

err_flux = np.asarray(
    data["err_flux"],
    dtype=float,
)

t0_ref = float(
    prep["t0"]
)

evaluator = prep[
    "full_evaluator"
]


# ============================================================
# Best complete physical point already obtained previously
#
# From:
# PSPL prefit -> PSPL+parallax
# negative-u0 start
#
# It had:
#   chi2 ~ 588124.1143
#   success=False
#   nfev=5000
#
# ============================================================

START_T0 = 2461849.046691147
START_U0 = -0.0009641175039411484
START_TE = 284.7597534679637
START_PIEN = 0.25722115213002894
START_PIEE = 0.1800551615651842

if START_U0 >= 0:
    raise RuntimeError(
        "This test is specifically for the negative-u0 branch."
    )


# ============================================================
# Common residual / chi2
# ============================================================

def residual_physical(
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    A = evaluator.magnification(
        t0=float(t0),
        u0=float(u0),
        tE=float(tE),
        piEN=float(piEN),
        piEE=float(piEE),
    )

    return base.profile_flux(
        A,
        flux,
        err_flux,
    )["residual"]


def chi2_physical(
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    r = residual_physical(
        t0,
        u0,
        tE,
        piEN,
        piEE,
    )

    return float(
        np.dot(r, r)
    )


# ============================================================
# Verify that we reproduce the known starting point
# ============================================================

chi2_start = chi2_physical(
    START_T0,
    START_U0,
    START_TE,
    START_PIEN,
    START_PIEE,
)

print()
print("=" * 110)
print("FIXED-SIGN log|u0| PARAMETERIZATION TEST")
print("=" * 110)

print()
print("DATASET")
print(
    "  t0_ref      =",
    t0_ref,
)
print(
    "  true u0     =",
    U0_TRUE,
)
print(
    "  true tE     =",
    TE_TRUE,
)
print(
    "  P/tE        =",
    P_OVER_TE,
)
print(
    "  epochs      =",
    len(flux),
)

print()
print("KNOWN STARTING POINT")
print(
    "  chi2        =",
    chi2_start,
)
print(
    "  t0          =",
    START_T0,
)
print(
    "  u0          =",
    START_U0,
)
print(
    "  |u0|        =",
    abs(START_U0),
)
print(
    "  tE          =",
    START_TE,
)
print(
    "  piEN        =",
    START_PIEN,
)
print(
    "  piEE        =",
    START_PIEE,
)
print(
    "  |piE|       =",
    np.hypot(
        START_PIEN,
        START_PIEE,
    ),
)


# ============================================================
# New internal coordinates
#
# x[0] = (t0 - t0_ref) / TE_TRUE
# x[1] = log(|u0|)
# x[2] = log(tE / TE_TRUE)
# x[3] = piEN
# x[4] = piEE
#
# Negative branch is enforced:
#
#       u0 = -exp(x[1])
#
# ============================================================

x0 = np.array(
    [
        (
            START_T0
            - t0_ref
        )
        / TE_TRUE,

        np.log(
            abs(
                START_U0
            )
        ),

        np.log(
            START_TE
            / TE_TRUE
        ),

        START_PIEN,
        START_PIEE,
    ],
    dtype=float,
)


# Keep exactly the old t0 and tE physical bounds.
shape_lower, shape_upper = (
    base.shape_bounds()
)

lower = np.array(
    [
        shape_lower[0],
        np.log(
            U0_ABS_MIN
        ),
        shape_lower[2],
        -PI_BOUND,
        -PI_BOUND,
    ],
    dtype=float,
)

upper = np.array(
    [
        shape_upper[0],
        np.log(
            U0_ABS_MAX
        ),
        shape_upper[2],
        +PI_BOUND,
        +PI_BOUND,
    ],
    dtype=float,
)


def decode(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    t0 = (
        t0_ref
        + x[0]
        * TE_TRUE
    )

    u0 = -np.exp(
        x[1]
    )

    tE = (
        TE_TRUE
        * np.exp(
            x[2]
        )
    )

    return (
        float(t0),
        float(u0),
        float(tE),
        float(x[3]),
        float(x[4]),
    )


def residual_logu(x):
    return residual_physical(
        *decode(x)
    )


# Check exact physical equivalence before optimizing.
decoded_start = decode(
    x0
)

chi2_transformed_start = float(
    np.dot(
        residual_logu(x0),
        residual_logu(x0),
    )
)

print()
print("TRANSFORMED START")
print(
    "  x0          =",
    x0,
)
print(
    "  decoded     =",
    decoded_start,
)
print(
    "  chi2        =",
    chi2_transformed_start,
)
print(
    "  delta chi2  =",
    (
        chi2_transformed_start
        - chi2_start
    ),
)

if not np.isclose(
    chi2_transformed_start,
    chi2_start,
    rtol=0.0,
    atol=1.0e-6,
):
    raise RuntimeError(
        "The transformed coordinates do not reproduce "
        "the same physical starting point."
    )


# ============================================================
# Optimize with fixed negative sign and log|u0|
# ============================================================

print()
print("-" * 110)
print("RUNNING TRF WITH u0 = -exp(eta)")
print("-" * 110)

fit = least_squares(
    residual_logu,
    x0,
    bounds=(
        lower,
        upper,
    ),
    method="trf",
    x_scale="jac",
    xtol=1.0e-10,
    ftol=1.0e-10,
    gtol=1.0e-10,
    max_nfev=5000,
    verbose=1,
)


# ============================================================
# Decode result
# ============================================================

t0_fit, u0_fit, tE_fit, piEN_fit, piEE_fit = (
    decode(
        fit.x
    )
)

r_final = residual_logu(
    fit.x
)

chi2_final = float(
    np.dot(
        r_final,
        r_final,
    )
)

piE_fit = float(
    np.hypot(
        piEN_fit,
        piEE_fit,
    )
)

abs_u0 = abs(
    u0_fit
)

near_u0_floor = bool(
    abs_u0
    < 10.0
    * U0_ABS_MIN
)

near_pi_bound = bool(
    max(
        abs(piEN_fit),
        abs(piEE_fit),
    )
    > 0.95
    * PI_BOUND
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 110)
print("RESULT")
print("=" * 110)

print(
    "success       =",
    fit.success,
)
print(
    "status        =",
    fit.status,
)
print(
    "message       =",
    fit.message,
)
print(
    "nfev          =",
    fit.nfev,
)
print(
    "njev          =",
    fit.njev,
)
print(
    "cost          =",
    fit.cost,
)
print(
    "optimality    =",
    fit.optimality,
)

print()
print(
    "chi2 start    =",
    chi2_start,
)
print(
    "chi2 final    =",
    chi2_final,
)
print(
    "delta chi2    =",
    (
        chi2_start
        - chi2_final
    ),
)

print()
print(
    "t0            =",
    t0_fit,
)
print(
    "u0            =",
    u0_fit,
)
print(
    "|u0|          =",
    abs_u0,
)
print(
    "log|u0|       =",
    fit.x[1],
)
print(
    "tE            =",
    tE_fit,
)
print(
    "piEN          =",
    piEN_fit,
)
print(
    "piEE          =",
    piEE_fit,
)
print(
    "|piE|         =",
    piE_fit,
)

print()
print(
    "near u0 floor =",
    near_u0_floor,
)
print(
    "near pi bound =",
    near_pi_bound,
)

print()
print(
    "internal x    =",
    fit.x,
)

print()
print("=" * 110)
print("INTERPRETATION")
print("=" * 110)

print("""
Compare primarily:

    nfev
    success / termination message
    chi2
    |u0|
    whether |u0| hits the artificial lower floor

The raw numerical value of `optimality` should NOT be compared
one-to-one with the old parametrization, because the coordinates
have changed.

Strong evidence that the old u0 coordinate was the problem would be:

    - chi2 <= the previous ~5.88e5 value,
    - convergence in far fewer than 5000 evaluations,
    - no hit on the |u0| floor,
    - stable physical parameters.

If |u0| runs all the way to 1e-8, that is also informative:
the preferred PSPL+parallax solution may be approaching the
u0 -> 0 boundary rather than a regular finite-u0 minimum.
""")
