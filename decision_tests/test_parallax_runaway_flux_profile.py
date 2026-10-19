#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from binary_source.analysis import (
    intrinsic_fit_tools as base,
)


# ============================================================
# Exact problematic configuration
# ============================================================

U0_TRUE = 0.03
TE_TRUE = 100.0
SOURCE_MAG = 19.0
P_OVER_TE = 1.0

PI_BOUND = 40.0

U0_ABS_MIN = 1.0e-8
U0_ABS_MAX = 20.0

# Diagnostic only:
# old shape bound was tE <= 20 * TE_TRUE = 2000 d.
TE_MAX_DIAGNOSTIC = 10000.0


# ============================================================
# Load validation helpers
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
# Build the same Asimov dataset
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
# Helpers
# ============================================================

def magnification(
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    return evaluator.magnification(
        t0=float(t0),
        u0=float(u0),
        tE=float(tE),
        piEN=float(piEN),
        piEE=float(piEE),
    )


def profile_at_point(
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    A = np.asarray(
        magnification(
            t0,
            u0,
            tE,
            piEN,
            piEE,
        ),
        dtype=float,
    )

    prof = base.profile_flux(
        A,
        flux,
        err_flux,
    )

    residual = np.asarray(
        prof["residual"],
        dtype=float,
    )

    chi2 = float(
        np.dot(
            residual,
            residual,
        )
    )

    # --------------------------------------------------------
    # Recover the actual profiled model flux robustly.
    #
    # Depending on residual sign convention:
    #
    #   r = (data - model) / sigma
    #
    # or
    #
    #   r = (model - data) / sigma
    #
    # Test both.  The correct reconstructed model must lie
    # essentially exactly in:
    #
    #   model = Fs * (A - 1) + Ft
    #
    # --------------------------------------------------------

    X = np.column_stack(
        [
            A - 1.0,
            np.ones_like(A),
        ]
    )

    candidates = []

    for residual_sign in (-1.0, +1.0):

        model_flux = (
            flux
            + residual_sign
            * residual
            * err_flux
        )

        coeff, *_ = np.linalg.lstsq(
            X,
            model_flux,
            rcond=None,
        )

        reconstructed = (
            X @ coeff
        )

        rms = float(
            np.sqrt(
                np.mean(
                    (
                        model_flux
                        - reconstructed
                    ) ** 2
                )
            )
        )

        candidates.append(
            (
                rms,
                residual_sign,
                model_flux,
                coeff,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    (
        reconstruction_rms,
        residual_sign,
        model_flux,
        coeff,
    ) = candidates[0]

    Fs = float(
        coeff[0]
    )

    Ft = float(
        coeff[1]
    )

    Fb = float(
        Ft - Fs
    )

    source_fraction = (
        Fs / Ft
        if Ft != 0.0
        else np.nan
    )

    blend_ratio = (
        Fb / Fs
        if Fs != 0.0
        else np.nan
    )

    piE = float(
        np.hypot(
            piEN,
            piEE,
        )
    )

    return {
        "chi2": chi2,
        "A": A,
        "profile": prof,
        "Fs": Fs,
        "Ft": Ft,
        "Fb": Fb,
        "source_fraction": source_fraction,
        "blend_ratio": blend_ratio,
        "reconstruction_rms": reconstruction_rms,
        "residual_sign": residual_sign,
        "piE": piE,
        "u0_tE": abs(u0) * tE,
        "tE_piE": tE * piE,
    }


def print_point(
    label,
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    out = profile_at_point(
        t0,
        u0,
        tE,
        piEN,
        piEE,
    )

    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    print(
        "chi2             =",
        out["chi2"],
    )

    print(
        "t0               =",
        t0,
    )

    print(
        "u0               =",
        u0,
    )

    print(
        "|u0|             =",
        abs(u0),
    )

    print(
        "tE               =",
        tE,
    )

    print(
        "piEN             =",
        piEN,
    )

    print(
        "piEE             =",
        piEE,
    )

    print(
        "|piE|            =",
        out["piE"],
    )

    print()
    print("PROFILED FLUX")

    print(
        "Fs               =",
        out["Fs"],
    )

    print(
        "Ft               =",
        out["Ft"],
    )

    print(
        "Fb = Ft - Fs     =",
        out["Fb"],
    )

    print(
        "Fs / Ft          =",
        out["source_fraction"],
    )

    print(
        "Fb / Fs          =",
        out["blend_ratio"],
    )

    print()
    print("DEGENERACY COMBINATIONS")

    print(
        "|u0| * tE        =",
        out["u0_tE"],
    )

    print(
        "tE * |piE|       =",
        out["tE_piE"],
    )

    print()
    print("PROFILE RECONSTRUCTION CHECK")

    print(
        "residual sign    =",
        out["residual_sign"],
    )

    print(
        "reconstruction RMS =",
        out["reconstruction_rms"],
    )

    # Also show scalar values directly returned by profile_flux,
    # in case it already exposes the fitted flux coefficients.
    print()
    print("SCALAR profile_flux OUTPUTS:")

    for key, value in out["profile"].items():

        arr = np.asarray(
            value
        )

        if arr.ndim == 0:
            print(
                f"  {key:24s} = {arr.item()}"
            )

    return out


# ============================================================
# Point A:
# old direct fit after 5000 evaluations
# ============================================================

POINT_A = {
    "t0": 2461849.046691147,
    "u0": -0.0009641175039411484,
    "tE": 284.7597534679637,
    "piEN": 0.25722115213002894,
    "piEE": 0.1800551615651842,
}


# ============================================================
# Point B:
# final result of fixed-sign log|u0| test
# ============================================================

POINT_B = {
    "t0": 2461849.054040197,
    "u0": -0.0002547990806783519,
    "tE": 1440.2152278019255,
    "piEN": 0.0634314318331091,
    "piEE": 0.03863483403899368,
}


print()
print("#" * 110)
print("TEST: IS THE PSPL+PARALLAX FIT RUNNING DOWN A HIGH-MAGNIFICATION / BLENDING VALLEY?")
print("#" * 110)

print_point(
    "POINT A — original 5D fit",
    **POINT_A,
)

print_point(
    "POINT B — after log|u0| continuation",
    **POINT_B,
)


# ============================================================
# Diagnostic continuation from Point B
#
# Same negative-u0 log parameterization, but allow
# tE to reach 10000 d instead of 2000 d.
# ============================================================

def encode(
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    if u0 >= 0.0:
        raise ValueError(
            "This diagnostic follows the negative-u0 branch."
        )

    return np.array(
        [
            (
                t0
                - t0_ref
            )
            / TE_TRUE,

            np.log(
                abs(u0)
            ),

            np.log(
                tE
                / TE_TRUE
            ),

            piEN,
            piEE,
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

    (
        t0,
        u0,
        tE,
        piEN,
        piEE,
    ) = decode(
        x
    )

    A = magnification(
        t0,
        u0,
        tE,
        piEN,
        piEE,
    )

    return base.profile_flux(
        A,
        flux,
        err_flux,
    )["residual"]


x0 = encode(
    **POINT_B,
)

shape_lower, shape_upper = (
    base.shape_bounds()
)

# Preserve the same t0 lower/upper bounds and original tE lower bound.
#
# Only diagnostic change:
#
#     tE upper: 2000 d -> 10000 d
#
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
        np.log(
            TE_MAX_DIAGNOSTIC
            / TE_TRUE
        ),
        +PI_BOUND,
        +PI_BOUND,
    ],
    dtype=float,
)


# ============================================================
# Verify same starting point
# ============================================================

r0 = np.asarray(
    residual_logu(
        x0
    ),
    dtype=float,
)

chi2_before = float(
    np.dot(
        r0,
        r0,
    )
)

print()
print("=" * 110)
print("EXTENDED-tE CONTINUATION")
print("=" * 110)

print(
    "old tE maximum       =",
    TE_TRUE * np.exp(
        shape_upper[2]
    ),
)

print(
    "diagnostic tE maximum=",
    TE_MAX_DIAGNOSTIC,
)

print(
    "start chi2           =",
    chi2_before,
)

print(
    "start x              =",
    x0,
)

print()
print("Running TRF...")
print()


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


(
    t0_final,
    u0_final,
    tE_final,
    piEN_final,
    piEE_final,
) = decode(
    fit.x
)

point_c = {
    "t0": t0_final,
    "u0": u0_final,
    "tE": tE_final,
    "piEN": piEN_final,
    "piEE": piEE_final,
}

out_c = print_point(
    "POINT C — continuation with tE <= 10000 d",
    **point_c,
)


print()
print("=" * 110)
print("OPTIMIZER RESULT")
print("=" * 110)

print(
    "success          =",
    fit.success,
)

print(
    "status           =",
    fit.status,
)

print(
    "message          =",
    fit.message,
)

print(
    "nfev             =",
    fit.nfev,
)

print(
    "njev             =",
    fit.njev,
)

print(
    "optimality       =",
    fit.optimality,
)

print(
    "internal x       =",
    fit.x,
)

print()
print(
    "near tE=10000 bound =",
    bool(
        tE_final
        > 0.95
        * TE_MAX_DIAGNOSTIC
    ),
)

print(
    "near |u0| floor     =",
    bool(
        abs(u0_final)
        < 10.0
        * U0_ABS_MIN
    ),
)

print(
    "near piE bound      =",
    bool(
        max(
            abs(piEN_final),
            abs(piEE_final),
        )
        > 0.95
        * PI_BOUND
    ),
)


# ============================================================
# Compact comparison
# ============================================================

out_a = profile_at_point(
    **POINT_A,
)

out_b = profile_at_point(
    **POINT_B,
)

print()
print("#" * 110)
print("COMPACT COMPARISON")
print("#" * 110)

print()
print(
    f"{'':12s}"
    f"{'POINT A':>18s}"
    f"{'POINT B':>18s}"
    f"{'POINT C':>18s}"
)

for label, a, b, c in [
    (
        "chi2",
        out_a["chi2"],
        out_b["chi2"],
        out_c["chi2"],
    ),
    (
        "|u0|",
        abs(POINT_A["u0"]),
        abs(POINT_B["u0"]),
        abs(point_c["u0"]),
    ),
    (
        "tE",
        POINT_A["tE"],
        POINT_B["tE"],
        point_c["tE"],
    ),
    (
        "|piE|",
        out_a["piE"],
        out_b["piE"],
        out_c["piE"],
    ),
    (
        "Fs",
        out_a["Fs"],
        out_b["Fs"],
        out_c["Fs"],
    ),
    (
        "Ft",
        out_a["Ft"],
        out_b["Ft"],
        out_c["Ft"],
    ),
    (
        "Fs/Ft",
        out_a["source_fraction"],
        out_b["source_fraction"],
        out_c["source_fraction"],
    ),
    (
        "Fb/Fs",
        out_a["blend_ratio"],
        out_b["blend_ratio"],
        out_c["blend_ratio"],
    ),
    (
        "|u0|tE",
        out_a["u0_tE"],
        out_b["u0_tE"],
        out_c["u0_tE"],
    ),
    (
        "tE|piE|",
        out_a["tE_piE"],
        out_b["tE_piE"],
        out_c["tE_piE"],
    ),
]:
    print(
        f"{label:12s}"
        f"{a:18.8g}"
        f"{b:18.8g}"
        f"{c:18.8g}"
    )


print()
print("#" * 110)
print("WHAT WOULD SUPPORT A RUNAWAY / DEGENERATE DIRECTION?")
print("#" * 110)

print("""
The strongest pattern would be:

    A -> B -> C

    chi2       keeps decreasing
    |u0|       keeps decreasing
    tE         keeps increasing
    Fs/Ft      keeps decreasing
    Fb/Fs      keeps increasing

with no regular finite-parameter minimum.

If Point C runs close to tE = 10000 d, that is particularly
strong evidence that the former tE <= 2000 d bound was merely
cutting off a descending direction.

That would NOT mean that 10000 d is a better scientific bound.
This is only a diagnostic test of the geometry of the profiled
PSPL+parallax likelihood.
""")
