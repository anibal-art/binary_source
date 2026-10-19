#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)


# ============================================================
# Exact same xallarap truth / Roman dataset
# ============================================================

U0_TRUE = 0.03
TE_TRUE = 100.0
SOURCE_MAG = 19.0
P_OVER_TE = 1.0

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

evaluator = prep["full_evaluator"]


# ============================================================
# Point C from the previous diagnostic
# ============================================================

T0_C = 2461849.054209083
U0_C = -0.00020922581202554943
TE_C = 1757.5335773996624
PIEN_C = 0.052129012612818416
PIEE_C = 0.03169739095588463


# ============================================================
# Scaling sequence
#
# Include lambda < 1 to see both sides of Point C,
# then follow the candidate runaway direction very far.
# ============================================================

LAMBDAS = np.array(
    [
        0.25,
        0.40,
        0.60,
        0.80,
        1.00,
        1.25,
        1.50,
        2.00,
        3.00,
        5.00,
        10.00,
        20.00,
        50.00,
        100.00,
    ],
    dtype=float,
)


def evaluate_point(
    t0,
    u0,
    tE,
    piEN,
    piEE,
):
    A = np.asarray(
        evaluator.magnification(
            t0=float(t0),
            u0=float(u0),
            tE=float(tE),
            piEN=float(piEN),
            piEE=float(piEE),
        ),
        dtype=float,
    )

    prof = base.profile_flux(
        A,
        flux,
        err_flux,
    )

    fs = float(
        prof["fs"]
    )

    ft = float(
        prof["ft"]
    )

    chi2 = float(
        prof["chi2"]
    )

    fb = (
        ft
        - fs
    )

    piE = float(
        np.hypot(
            piEN,
            piEE,
        )
    )

    model_flux = (
        fs
        * (
            A
            - 1.0
        )
        + ft
    )

    return {
        "A": A,
        "model_flux": model_flux,
        "chi2": chi2,
        "fs": fs,
        "ft": ft,
        "fb": fb,
        "fs_over_ft": fs / ft,
        "fb_over_fs": fb / fs,
        "piE": piE,
    }


# ============================================================
# First evaluate lambda = 1 reference
# ============================================================

ref = evaluate_point(
    T0_C,
    U0_C,
    TE_C,
    PIEN_C,
    PIEE_C,
)

model_ref = ref[
    "model_flux"
]


# ============================================================
# Scan
# ============================================================

rows = []

for lam in LAMBDAS:

    t0 = T0_C

    u0 = (
        U0_C
        / lam
    )

    tE = (
        TE_C
        * lam
    )

    piEN = (
        PIEN_C
        / lam
    )

    piEE = (
        PIEE_C
        / lam
    )

    out = evaluate_point(
        t0,
        u0,
        tE,
        piEN,
        piEE,
    )

    delta_model_sigma = (
        (
            out["model_flux"]
            - model_ref
        )
        / err_flux
    )

    rows.append(
        {
            "lambda": lam,
            "chi2": out["chi2"],

            "t0": t0,
            "u0": u0,
            "abs_u0": abs(u0),
            "tE": tE,

            "piEN": piEN,
            "piEE": piEE,
            "piE": out["piE"],

            "fs": out["fs"],
            "ft": out["ft"],
            "fb": out["fb"],

            "fs_over_ft": (
                out["fs_over_ft"]
            ),

            "fb_over_fs": (
                out["fb_over_fs"]
            ),

            # Candidate invariants
            "abs_u0_tE": (
                abs(u0)
                * tE
            ),

            "tE_piEN": (
                tE
                * piEN
            ),

            "tE_piEE": (
                tE
                * piEE
            ),

            "tE_piE": (
                tE
                * out["piE"]
            ),

            "fs_tE": (
                out["fs"]
                * tE
            ),

            # Compare the actual profiled light curve
            # against lambda=1 in units of sigma.
            "model_delta_sigma_rms": float(
                np.sqrt(
                    np.mean(
                        delta_model_sigma**2
                    )
                )
            ),

            "model_delta_sigma_max": float(
                np.max(
                    np.abs(
                        delta_model_sigma
                    )
                )
            ),
        }
    )


df = pd.DataFrame(
    rows
)


# ============================================================
# Diagnostics relative to largest-lambda point
# ============================================================

chi2_last = float(
    df.iloc[-1]["chi2"]
)

df["chi2_minus_last"] = (
    df["chi2"]
    - chi2_last
)


# ============================================================
# Print compact table
# ============================================================

print()
print("#" * 132)
print("DIRECT SCAN OF THE CANDIDATE PSPL+PARALLAX SCALING DIRECTION")
print("#" * 132)

print()
print(
    "Point C:"
)
print(
    f"  t0    = {T0_C:.12f}"
)
print(
    f"  u0    = {U0_C:.12g}"
)
print(
    f"  tE    = {TE_C:.12g}"
)
print(
    f"  piEN  = {PIEN_C:.12g}"
)
print(
    f"  piEE  = {PIEE_C:.12g}"
)
print(
    f"  chi2  = {ref['chi2']:.12f}"
)

print()
print(
    f"{'lam':>7s}"
    f"{'chi2':>16s}"
    f"{'chi2-last':>14s}"
    f"{'|u0|':>13s}"
    f"{'tE':>13s}"
    f"{'|piE|':>12s}"
    f"{'Fs':>13s}"
    f"{'Fs/Ft':>12s}"
    f"{'Fb/Fs':>12s}"
    f"{'Fs*tE':>14s}"
)

for _, r in df.iterrows():

    print(
        f"{r['lambda']:7.2f}"
        f"{r['chi2']:16.5f}"
        f"{r['chi2_minus_last']:14.5f}"
        f"{r['abs_u0']:13.5e}"
        f"{r['tE']:13.3f}"
        f"{r['piE']:12.5e}"
        f"{r['fs']:13.5f}"
        f"{r['fs_over_ft']:12.5e}"
        f"{r['fb_over_fs']:12.5f}"
        f"{r['fs_tE']:14.5f}"
    )


# ============================================================
# Explicit invariant check
# ============================================================

print()
print("#" * 132)
print("SCALING INVARIANTS")
print("#" * 132)

for col in [
    "abs_u0_tE",
    "tE_piEN",
    "tE_piEE",
    "tE_piE",
]:
    vals = np.asarray(
        df[col],
        dtype=float,
    )

    rel_span = (
        (
            vals.max()
            - vals.min()
        )
        / np.mean(
            np.abs(vals)
        )
    )

    print(
        f"{col:14s}"
        f" min={vals.min():.12g}"
        f" max={vals.max():.12g}"
        f" relative_span={rel_span:.3e}"
    )


# ============================================================
# Large-lambda behavior
# ============================================================

tail = df[
    df["lambda"] >= 5.0
].copy()

print()
print("#" * 132)
print("LARGE-LAMBDA TAIL")
print("#" * 132)

for _, r in tail.iterrows():

    print(
        f"lambda={r['lambda']:7.2f}"
        f"  chi2={r['chi2']:.9f}"
        f"  chi2-last={r['chi2_minus_last']:+.9f}"
        f"  Fs*tE={r['fs_tE']:.9f}"
        f"  model RMS vs lambda=1="
        f"{r['model_delta_sigma_rms']:.6g} sigma"
    )


# ============================================================
# Check monotonicity from lambda >= 1
# ============================================================

forward = df[
    df["lambda"] >= 1.0
].copy()

chi = np.asarray(
    forward["chi2"],
    dtype=float,
)

dchi = np.diff(
    chi
)

print()
print("#" * 132)
print("MONOTONICITY CHECK FOR lambda >= 1")
print("#" * 132)

print(
    "all successive delta-chi2 <= 0 ?",
    bool(
        np.all(
            dchi <= 0.0
        )
    )
)

print(
    "successive delta-chi2 =",
    dchi,
)


# ============================================================
# Crude asymptotic fit
#
# If chi2(lambda) = chi2_inf + a/lambda + b/lambda^2 + ...
# then the large-lambda points should be well described by
# a low-order polynomial in 1/lambda.
# ============================================================

tail_fit = df[
    df["lambda"] >= 5.0
].copy()

z = (
    1.0
    / np.asarray(
        tail_fit["lambda"],
        dtype=float,
    )
)

y = np.asarray(
    tail_fit["chi2"],
    dtype=float,
)

coef = np.polyfit(
    z,
    y,
    deg=2,
)

chi2_inf_est = float(
    coef[-1]
)

pred = np.polyval(
    coef,
    z,
)

rms_asymptotic_fit = float(
    np.sqrt(
        np.mean(
            (
                pred
                - y
            ) ** 2
        )
    )
)

print()
print("#" * 132)
print("CRUDE ASYMPTOTIC EXTRAPOLATION")
print("#" * 132)

print(
    "fit form:"
)
print(
    "  chi2(lambda) = c0 + c1/lambda + c2/lambda^2"
)

print(
    "estimated chi2(lambda -> infinity) =",
    chi2_inf_est,
)

print(
    "RMS of tail fit =",
    rms_asymptotic_fit,
)


# ============================================================
# Save result
# ============================================================

OUT = Path(
    "/tmp/parallax_scaling_asymptote.csv"
)

df.to_csv(
    OUT,
    index=False,
)

print()
print(
    "saved:",
    OUT,
)

print()
print("#" * 132)
print("INTERPRETATION GUIDE")
print("#" * 132)

print("""
Strong evidence for an asymptotic degeneracy would be:

  1. for lambda >= 1, chi2 continues to decrease or rapidly
     approaches a plateau;

  2. the prescribed combinations
         |u0| * tE,
         tE * piEN,
         tE * piEE
     remain exactly constant by construction;

  3. the profiled solution approximately develops
         Fs ~ 1/lambda
     so that Fs * tE approaches a constant;

  4. there is no distinguished finite lambda where chi2 has
     an isolated minimum;

  5. the large-lambda values are well described by a smooth
     expansion in 1/lambda with a finite chi2 limit.

If instead chi2 reaches a clear minimum and rises again at large
lambda, then Point C lies near a regular finite solution and the
runaway interpretation is wrong or incomplete.
""")
