#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

import intrinsic_bspl_pspl_parallax_scan as seq

from binary_source.analysis import (
    intrinsic_fit_tools as base,
)
from binary_source.analysis import (
    intrinsic_parallax as par,
)


TE_TRUE = 30.0

MINIMA_PATH = Path(
    "/tmp/parallax_branch_minima.csv"
)

OUTDIR = Path(
    "/tmp/final_jerk_parallax_check"
)

# Stay strictly inside the Roman ephemeris interval
# [t0-0.5, t0+0.5].
MAX_WINDOW = 0.45
N_TIME = 9001

LOCAL_WINDOWS = [
    0.20,
    0.30,
    0.40,
    0.45,
]

POLY_DEGREE = 6


def pylima_parameters_from_row(
    evaluator,
    row,
):
    values = {
        "t0": float(row["t0"]),
        "u0": float(row["u0"]),
        "tE": float(row["tE"]),
        "piEN": float(row["piEN"]),
        "piEE": float(row["piEE"]),
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
    row,
):
    params = pylima_parameters_from_row(
        evaluator,
        row,
    )

    out = evaluator.model.sources_trajectory(
        evaluator.telescope,
        params,
        data_type="photometry",
    )

    x = np.asarray(
        out[0],
        dtype=float,
    )

    y = np.asarray(
        out[1],
        dtype=float,
    )

    if x.shape != y.shape:
        raise RuntimeError(
            f"trajectory shape mismatch: "
            f"{x.shape} {y.shape}"
        )

    return x * x + y * y


def geometry_derivatives(
    t,
    t0,
    delta,
    window=0.45,
):
    """
    Local polynomial expansion of pyLIMA deltas_positions.

    With

        delta = c0 + c1 dt + c2 dt^2 + ...

    alpha = 2 c2.

    We only need alpha for Gould Eq. 22.
    """

    dt = (
        np.asarray(t)
        - float(t0)
    )

    use = (
        np.abs(dt)
        <= float(window)
    )

    # Scale to [-1,1] for conditioning.
    s = (
        dt[use]
        / float(window)
    )

    coeffs = []

    for component in range(2):

        c_scaled = (
            np.polynomial.polynomial.polyfit(
                s,
                delta[
                    component,
                    use,
                ],
                deg=6,
            )
        )

        # delta =
        # c_scaled[k] * (dt/window)^k
        #
        # physical coefficient of dt^k:
        c = np.array(
            [
                c_scaled[k]
                / window**k
                for k
                in range(
                    len(c_scaled)
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

    e_parallel = (
        alpha_vec
        / alpha
    )

    # Same convention used in previous check.
    e_perp = np.array(
        [
            -e_parallel[1],
            +e_parallel[0],
        ]
    )

    return (
        alpha,
        e_parallel,
        e_perp,
    )


def project_perp(
    row,
    e_perp,
):
    pi = np.array(
        [
            float(row["piEN"]),
            float(row["piEE"]),
        ]
    )

    return float(
        np.dot(
            pi,
            e_perp,
        )
    )


def fit_local_coefficients(
    t,
    u2,
    t0_ref,
    window,
):
    """
    Fit

        u^2 = sum a_k s^k,

    where

        s = (t-t0_ref)/window.

    Convert to the same dimensionless coordinate used previously,

        z = (t-t0_ref)/TE_TRUE,

    so that

        C_k = a_k (TE_TRUE/window)^k.

    The fit itself is done in s for numerical conditioning.
    """

    dt = (
        np.asarray(t)
        - float(t0_ref)
    )

    use = (
        np.abs(dt)
        <= float(window)
    )

    s = (
        dt[use]
        / float(window)
    )

    y = np.asarray(
        u2[use],
        dtype=float,
    )

    a = (
        np.polynomial.polynomial.polyfit(
            s,
            y,
            deg=POLY_DEGREE,
        )
    )

    fitted = (
        np.polynomial.polynomial.polyval(
            s,
            a,
        )
    )

    C = np.array(
        [
            a[k]
            * (
                TE_TRUE
                / float(window)
            )**k
            for k in range(
                len(a)
            )
        ],
        dtype=float,
    )

    return {
        "C": C,
        "rms": float(
            np.sqrt(
                np.mean(
                    (
                        y - fitted
                    )**2
                )
            )
        ),
    }


def relative_difference(
    a,
    b,
    floor=1e-14,
):
    return (
        abs(
            float(b)
            - float(a)
        )
        /
        max(
            abs(float(a)),
            abs(float(b)),
            floor,
        )
    )


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Exact Roman/L2 geometry
# ============================================================

_, t0_ref = (
    base.roman.build_roman_times(
        tE_true=TE_TRUE
    )
)

t = np.linspace(
    t0_ref - MAX_WINDOW,
    t0_ref + MAX_WINDOW,
    N_TIME,
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

if delta.shape != (
    2,
    N_TIME,
):
    raise RuntimeError(
        f"unexpected deltas_positions shape: "
        f"{delta.shape}"
    )


alpha, epar, eperp = (
    geometry_derivatives(
        t=t,
        t0=t0_ref,
        delta=delta,
        window=MAX_WINDOW,
    )
)


print("=" * 110)
print("ROMAN/L2 LOCAL GEOMETRY")
print("=" * 110)

print(
    "t0_ref        =",
    t0_ref,
)

print(
    "interval      =",
    t[0],
    t[-1],
)

print(
    "alpha         =",
    alpha,
    "day^-2",
)

print(
    "e_parallel NE =",
    epar,
)

print(
    "e_perp NE     =",
    eperp,
)


# ============================================================
# Load recovered branches
# ============================================================

df = pd.read_csv(
    MINIMA_PATH
)

df["u0_sign"] = np.where(
    df["u0"] >= 0.0,
    +1,
    -1,
)


eq22_rows = []
taylor_rows = []


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

    A = group.iloc[0]
    B = group.iloc[1]

    # ========================================================
    # Gould Eq. 22
    #
    # Delta tE ~
    #
    # -(alpha tE^2 / 2)
    #  (pi_perp Delta u0
    #   + u0 Delta pi_perp)
    #  tE
    #
    # Evaluate A -> B.
    # ========================================================

    pi_perp_A = project_perp(
        A,
        eperp,
    )

    pi_perp_B = project_perp(
        B,
        eperp,
    )

    du0 = (
        float(B["u0"])
        - float(A["u0"])
    )

    dpi_perp = (
        pi_perp_B
        - pi_perp_A
    )

    tE_A = float(
        A["tE"]
    )

    tE_B = float(
        B["tE"]
    )

    observed_dtE = (
        tE_B
        - tE_A
    )

    predicted_dtE = -(
        alpha
        * tE_A**3
        / 2.0
    ) * (
        pi_perp_A
        * du0
        +
        float(A["u0"])
        * dpi_perp
    )

    abs_error = (
        predicted_dtE
        - observed_dtE
    )

    rel_error = (
        abs_error
        / observed_dtE
        if observed_dtE != 0.0
        else np.nan
    )

    eq22_rows.append(
        {
            "P_over_tE":
                float(P_over_tE),

            "u0_sign":
                int(u0_sign),

            "u0_A":
                float(A["u0"]),

            "u0_B":
                float(B["u0"]),

            "tE_A":
                tE_A,

            "tE_B":
                tE_B,

            "pi_perp_A":
                pi_perp_A,

            "pi_perp_B":
                pi_perp_B,

            "delta_u0":
                du0,

            "delta_pi_perp":
                dpi_perp,

            "observed_delta_tE":
                observed_dtE,

            "predicted_delta_tE":
                predicted_dtE,

            "delta_tE_error":
                abs_error,

            "relative_error":
                rel_error,
        }
    )


    # ========================================================
    # Exact pyLIMA trajectories
    # ========================================================

    u2_A = trajectory_u2(
        evaluator,
        A,
    )

    u2_B = trajectory_u2(
        evaluator,
        B,
    )

    du2 = (
        u2_B
        - u2_A
    )


    # ========================================================
    # Ultra-local Taylor coefficients
    # ========================================================

    for window in LOCAL_WINDOWS:

        fa = fit_local_coefficients(
            t=t,
            u2=u2_A,
            t0_ref=t0_ref,
            window=window,
        )

        fb = fit_local_coefficients(
            t=t,
            u2=u2_B,
            t0_ref=t0_ref,
            window=window,
        )

        CA = fa["C"]
        CB = fb["C"]

        use = (
            np.abs(
                t - t0_ref
            )
            <= window
        )

        row = {
            "P_over_tE":
                float(P_over_tE),

            "u0_sign":
                int(u0_sign),

            "window_days":
                float(window),

            "poly_rms_A":
                fa["rms"],

            "poly_rms_B":
                fb["rms"],

            "pair_u2_rms":
                float(
                    np.sqrt(
                        np.mean(
                            du2[use]**2
                        )
                    )
                ),

            "pair_u2_max":
                float(
                    np.max(
                        np.abs(
                            du2[use]
                        )
                    )
                ),
        }

        for k in range(5):

            row[
                f"C{k}_A"
            ] = float(
                CA[k]
            )

            row[
                f"C{k}_B"
            ] = float(
                CB[k]
            )

            row[
                f"dC{k}"
            ] = float(
                CB[k]
                - CA[k]
            )

            row[
                f"rel_dC{k}"
            ] = relative_difference(
                CA[k],
                CB[k],
            )

        taylor_rows.append(
            row
        )


eq22 = pd.DataFrame(
    eq22_rows
)

taylor = pd.DataFrame(
    taylor_rows
)


eq22.to_csv(
    OUTDIR
    / "gould_eq22_test.csv",
    index=False,
)

taylor.to_csv(
    OUTDIR
    / "ultralocal_taylor_test.csv",
    index=False,
)


# ============================================================
# Reports
# ============================================================

print()
print("=" * 110)
print("GOULD EQ. 22 — FINITE-u0 TIMESCALE CORRECTION")
print("=" * 110)

print(
    eq22[
        [
            "P_over_tE",
            "u0_sign",
            "pi_perp_A",
            "pi_perp_B",
            "observed_delta_tE",
            "predicted_delta_tE",
            "delta_tE_error",
            "relative_error",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "median |relative error| =",
    np.median(
        np.abs(
            eq22[
                "relative_error"
            ]
        )
    ),
)

print(
    "max |relative error|    =",
    np.max(
        np.abs(
            eq22[
                "relative_error"
            ]
        )
    ),
)


print()
print("=" * 110)
print("ULTRALOCAL TAYLOR TEST")
print("=" * 110)

print(
    taylor[
        [
            "P_over_tE",
            "u0_sign",
            "window_days",
            "rel_dC0",
            "rel_dC1",
            "rel_dC2",
            "rel_dC3",
            "rel_dC4",
            "pair_u2_rms",
            "poly_rms_A",
            "poly_rms_B",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 110)
print("C4 STABILITY BY WINDOW")
print("=" * 110)

for (
    P_over_tE,
    u0_sign,
), g in taylor.groupby(
    [
        "P_over_tE",
        "u0_sign",
    ]
):

    print()
    print(
        f"P/tE={P_over_tE:.5g} "
        f"sign={int(u0_sign):+d}"
    )

    print(
        g[
            [
                "window_days",
                "C4_A",
                "C4_B",
                "dC4",
                "rel_dC4",
            ]
        ].to_string(
            index=False
        )
    )


print()
print(
    "saved:",
    OUTDIR
    / "gould_eq22_test.csv"
)

print(
    "saved:",
    OUTDIR
    / "ultralocal_taylor_test.csv"
)
