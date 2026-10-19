#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

import intrinsic_bspl_pspl_parallax_scan as seq

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_parallax as par,
)


TE_TRUE = 30.0

MINIMA_PATH = Path(
    "/tmp/parallax_branch_minima.csv"
)

OUTDIR = Path(
    "/tmp/taylor_parallax_branch_check"
)

# Local windows used to check stability of the Taylor coefficients.
WINDOWS_DAYS = [
    2.0,
    5.0,
    10.0,
    15.0,
    20.0,
]

# One common fine grid is enough for all windows.
MAX_WINDOW_DAYS = max(WINDOWS_DAYS)
N_TIME = 8001

# Fit higher order than the coefficients we care about so that
# C2-C4 do not absorb C5,C6,... over the finite window.
POLY_DEGREE = 8


def u2_from_pspl_magnification(A):
    """
    Exact inversion of

        A = (u^2 + 2) / [u sqrt(u^2 + 4)]

    giving

        u^2 = 2 [ A/sqrt(A^2 - 1) - 1 ].
    """

    A = np.asarray(
        A,
        dtype=float,
    )

    if np.any(~np.isfinite(A)):
        raise RuntimeError(
            "Non-finite magnification."
        )

    if np.any(A <= 1.0):
        raise RuntimeError(
            "PSPL magnification must be > 1."
        )

    denom = np.sqrt(
        np.maximum(
            A * A - 1.0,
            np.finfo(float).tiny,
        )
    )

    u2 = 2.0 * (
        A / denom - 1.0
    )

    if np.any(u2 < -1e-12):
        raise RuntimeError(
            "Recovered negative u^2."
        )

    return np.maximum(
        u2,
        0.0,
    )


def evaluate_solution(
    evaluator,
    row,
):
    A = evaluator.magnification(
        t0=float(row["t0"]),
        u0=float(row["u0"]),
        tE=float(row["tE"]),
        piEN=float(row["piEN"]),
        piEE=float(row["piEE"]),
    )

    u2 = u2_from_pspl_magnification(
        A
    )

    return (
        np.asarray(A, dtype=float),
        np.asarray(u2, dtype=float),
    )


def taylor_coefficients(
    t,
    u2,
    t0_ref,
    window_days,
):
    """
    Expand u^2 in the COMMON dimensionless coordinate

        z = (t - t0_ref) / TE_TRUE

    as

        u^2 = C0 + C1 z + C2 z^2 + ...

    Equality of coefficients between two branches is unaffected
    by this common rescaling of time.
    """

    dt = (
        np.asarray(t, dtype=float)
        - float(t0_ref)
    )

    use = (
        np.abs(dt)
        <= float(window_days)
    )

    if np.count_nonzero(use) < 50:
        raise RuntimeError(
            "Too few points in Taylor window."
        )

    z = (
        dt[use]
        / TE_TRUE
    )

    coeff = (
        np.polynomial.polynomial.polyfit(
            z,
            np.asarray(u2)[use],
            deg=POLY_DEGREE,
        )
    )

    fitted = (
        np.polynomial.polynomial.polyval(
            z,
            coeff,
        )
    )

    residual = (
        np.asarray(u2)[use]
        - fitted
    )

    return {
        "coeff":
            coeff,

        "rms_poly":
            float(
                np.sqrt(
                    np.mean(
                        residual**2
                    )
                )
            ),

        "max_poly":
            float(
                np.max(
                    np.abs(
                        residual
                    )
                )
            ),
    }


def relative_difference(
    a,
    b,
    floor=1e-14,
):
    scale = max(
        abs(float(a)),
        abs(float(b)),
        floor,
    )

    return abs(
        float(b) - float(a)
    ) / scale


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Common Roman/L2 geometry
# ============================================================

_, t0_ref = (
    base.roman.build_roman_times(
        tE_true=TE_TRUE
    )
)

t = np.linspace(
    t0_ref - MAX_WINDOW_DAYS,
    t0_ref + MAX_WINDOW_DAYS,
    N_TIME,
)

evaluator = (
    par.PSPLParallaxEvaluator(
        t=t,
        t0_reference=t0_ref,
    )
)

print(
    "t0 reference =",
    t0_ref,
)

print(
    "time interval =",
    t[0],
    t[-1],
)

print(
    "N time =",
    len(t),
)


# ============================================================
# Load empirical minima
# ============================================================

df = pd.read_csv(
    MINIMA_PATH
)

df["u0_sign"] = np.where(
    df["u0"] >= 0.0,
    +1,
    -1,
)


# ============================================================
# Compare two best minima at fixed P/tE and fixed sign(u0)
# ============================================================

summary_rows = []
coefficient_rows = []


for (
    P_over_tE,
    u0_sign,
), group in df.groupby(
    [
        "P_over_tE",
        "u0_sign",
    ],
    sort=True,
):

    group = (
        group
        .sort_values("D")
        .reset_index(drop=True)
    )

    if len(group) < 2:
        continue

    a = group.iloc[0]
    b = group.iloc[1]

    A_a, u2_a = (
        evaluate_solution(
            evaluator,
            a,
        )
    )

    A_b, u2_b = (
        evaluate_solution(
            evaluator,
            b,
        )
    )

    print()
    print("=" * 120)

    print(
        f"P/tE={P_over_tE:.5g} "
        f"sign(u0)={int(u0_sign):+d}"
    )

    print("=" * 120)

    print(
        "A:",
        f"D={a['D']:.9e}",
        f"u0={a['u0']:+.8f}",
        f"tE={a['tE']:.8f}",
        f"pi=({a['piEN']:+.8f},"
        f"{a['piEE']:+.8f})",
    )

    print(
        "B:",
        f"D={b['D']:.9e}",
        f"u0={b['u0']:+.8f}",
        f"tE={b['tE']:.8f}",
        f"pi=({b['piEN']:+.8f},"
        f"{b['piEE']:+.8f})",
    )

    for window_days in WINDOWS_DAYS:

        fit_a = (
            taylor_coefficients(
                t=t,
                u2=u2_a,
                t0_ref=t0_ref,
                window_days=window_days,
            )
        )

        fit_b = (
            taylor_coefficients(
                t=t,
                u2=u2_b,
                t0_ref=t0_ref,
                window_days=window_days,
            )
        )

        ca = fit_a["coeff"]
        cb = fit_b["coeff"]

        # Difference of the actual trajectories inside the window.
        use = (
            np.abs(
                t - t0_ref
            )
            <= window_days
        )

        du2 = (
            u2_b[use]
            - u2_a[use]
        )

        dA = (
            A_b[use]
            - A_a[use]
        )

        row = {
            "P_over_tE":
                float(P_over_tE),

            "u0_sign":
                int(u0_sign),

            "window_days":
                float(window_days),

            "D_a":
                float(a["D"]),

            "D_b":
                float(b["D"]),

            "u0_a":
                float(a["u0"]),

            "u0_b":
                float(b["u0"]),

            "tE_a":
                float(a["tE"]),

            "tE_b":
                float(b["tE"]),

            "piEN_a":
                float(a["piEN"]),

            "piEE_a":
                float(a["piEE"]),

            "piEN_b":
                float(b["piEN"]),

            "piEE_b":
                float(b["piEE"]),

            "u2_pair_rms":
                float(
                    np.sqrt(
                        np.mean(
                            du2**2
                        )
                    )
                ),

            "u2_pair_max":
                float(
                    np.max(
                        np.abs(
                            du2
                        )
                    )
                ),

            "A_pair_rms":
                float(
                    np.sqrt(
                        np.mean(
                            dA**2
                        )
                    )
                ),

            "A_pair_max":
                float(
                    np.max(
                        np.abs(
                            dA
                        )
                    )
                ),

            "poly_rms_a":
                fit_a["rms_poly"],

            "poly_rms_b":
                fit_b["rms_poly"],
        }

        for k in range(
            POLY_DEGREE + 1
        ):

            row[
                f"C{k}_a"
            ] = float(
                ca[k]
            )

            row[
                f"C{k}_b"
            ] = float(
                cb[k]
            )

            row[
                f"dC{k}"
            ] = float(
                cb[k]
                - ca[k]
            )

            row[
                f"rel_dC{k}"
            ] = relative_difference(
                ca[k],
                cb[k],
            )

        # Gould-relevant coefficient blocks.
        d234 = (
            cb[2:5]
            - ca[2:5]
        )

        scale234 = max(
            np.linalg.norm(
                ca[2:5]
            ),
            np.linalg.norm(
                cb[2:5]
            ),
            1e-14,
        )

        row[
            "norm_dC234"
        ] = float(
            np.linalg.norm(
                d234
            )
        )

        row[
            "rel_norm_dC234"
        ] = float(
            np.linalg.norm(
                d234
            )
            / scale234
        )

        d01234 = (
            cb[:5]
            - ca[:5]
        )

        scale01234 = max(
            np.linalg.norm(
                ca[:5]
            ),
            np.linalg.norm(
                cb[:5]
            ),
            1e-14,
        )

        row[
            "rel_norm_dC0to4"
        ] = float(
            np.linalg.norm(
                d01234
            )
            / scale01234
        )

        coefficient_rows.append(
            row
        )

        print()
        print(
            f"window={window_days:5.1f} d"
        )

        for k in range(5):

            print(
                f"  C{k}: "
                f"A={ca[k]:+.10e} "
                f"B={cb[k]:+.10e} "
                f"d={cb[k]-ca[k]:+.3e} "
                f"rel={relative_difference(ca[k],cb[k]):.3e}"
            )

        print(
            "  rel ||dC2..C4|| =",
            f"{row['rel_norm_dC234']:.6e}",
        )

        print(
            "  RMS delta u^2   =",
            f"{row['u2_pair_rms']:.6e}",
        )

        print(
            "  RMS delta A     =",
            f"{row['A_pair_rms']:.6e}",
        )

    # Use the 5-day window as one compact diagnostic.
    # Small enough to be local, wide enough to be numerically stable.
    five = [
        row
        for row in coefficient_rows
        if (
            row["P_over_tE"]
            == float(P_over_tE)
            and row["u0_sign"]
            == int(u0_sign)
            and row["window_days"]
            == 5.0
        )
    ][-1]

    summary_rows.append(
        {
            "P_over_tE":
                float(P_over_tE),

            "u0_sign":
                int(u0_sign),

            "D_a":
                float(a["D"]),

            "D_b":
                float(b["D"]),

            "rel_dC0":
                five["rel_dC0"],

            "rel_dC1":
                five["rel_dC1"],

            "rel_dC2":
                five["rel_dC2"],

            "rel_dC3":
                five["rel_dC3"],

            "rel_dC4":
                five["rel_dC4"],

            "rel_norm_dC234":
                five[
                    "rel_norm_dC234"
                ],

            "u2_pair_rms":
                five[
                    "u2_pair_rms"
                ],

            "A_pair_rms":
                five[
                    "A_pair_rms"
                ],
        }
    )


coef_df = pd.DataFrame(
    coefficient_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


coef_path = (
    OUTDIR
    / "taylor_coefficients_all_windows.csv"
)

summary_path = (
    OUTDIR
    / "taylor_pair_summary_5d.csv"
)

coef_df.to_csv(
    coef_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 120)
print("SUMMARY — 5 DAY LOCAL EXPANSION")
print("=" * 120)

if len(summary_df):

    print(
        summary_df[
            [
                "P_over_tE",
                "u0_sign",
                "rel_dC0",
                "rel_dC1",
                "rel_dC2",
                "rel_dC3",
                "rel_dC4",
                "rel_norm_dC234",
                "u2_pair_rms",
                "A_pair_rms",
            ]
        ].to_string(
            index=False
        )
    )

print()
print(
    "saved:",
    coef_path,
)

print(
    "saved:",
    summary_path,
)
