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
N_TIME = 5000

MINIMA_PATH = Path(
    "/tmp/parallax_branch_minima.csv"
)

OUTDIR = Path(
    "/tmp/jerk_relation_check"
)

WINDOWS_DAYS = [
    5.0,
    10.0,
    20.0,
    30.0,
]

MAIN_WINDOW = 20.0


def get_telescope(evaluator):

    if hasattr(
        evaluator,
        "telescope",
    ):
        return evaluator.telescope

    if hasattr(
        evaluator,
        "event",
    ):
        return evaluator.event.telescopes[0]

    if hasattr(
        evaluator,
        "model",
    ):
        return evaluator.model.event.telescopes[0]

    raise RuntimeError(
        "Could not locate telescope "
        "inside PSPLParallaxEvaluator."
    )


def observer_kinematics(
    *,
    t,
    t0,
    delta,
    window_days,
):
    """
    Local Taylor expansion of pyLIMA's exact projected
    parallax displacement.

        Delta s(dt)
          = c0 + c1 dt + c2 dt^2 + c3 dt^3 + ...

        alpha = 2 c2
        jerk  = 6 c3

    Coordinates are pyLIMA projected (North, East).
    Time units are days.
    """

    dt = (
        np.asarray(t)
        - float(t0)
    )

    use = (
        np.abs(dt)
        <= float(window_days)
    )

    if np.count_nonzero(use) < 20:
        raise RuntimeError(
            "Too few points in local window."
        )

    coeffs = []

    for component in range(2):

        c = (
            np.polynomial.polynomial.polyfit(
                dt[use],
                delta[
                    component,
                    use,
                ],
                deg=5,
            )
        )

        coeffs.append(c)

    coeffs = np.asarray(
        coeffs
    )

    alpha = (
        2.0
        * coeffs[:, 2]
    )

    jerk = (
        6.0
        * coeffs[:, 3]
    )

    alpha_norm = float(
        np.linalg.norm(alpha)
    )

    e_parallel = (
        alpha
        / alpha_norm
    )

    # +90 degree rotation in (N,E).
    e_perp = np.array(
        [
            -e_parallel[1],
            +e_parallel[0],
        ]
    )

    return {
        "window_days":
            float(window_days),

        "alpha_N":
            float(alpha[0]),

        "alpha_E":
            float(alpha[1]),

        "alpha_norm":
            alpha_norm,

        "jerk_N":
            float(jerk[0]),

        "jerk_E":
            float(jerk[1]),

        "epar_N":
            float(e_parallel[0]),

        "epar_E":
            float(e_parallel[1]),

        "eperp_N":
            float(e_perp[0]),

        "eperp_E":
            float(e_perp[1]),

        "alpha":
            alpha,

        "jerk":
            jerk,

        "e_parallel":
            e_parallel,

        "e_perp":
            e_perp,

        # Geocentric sanity checks:
        "c0_N":
            float(coeffs[0, 0]),

        "c0_E":
            float(coeffs[1, 0]),

        "c1_N":
            float(coeffs[0, 1]),

        "c1_E":
            float(coeffs[1, 1]),
    }


def project_piE(
    piEN,
    piEE,
    e_parallel,
    e_perp,
):

    v = np.array(
        [
            float(piEN),
            float(piEE),
        ]
    )

    return (
        float(
            np.dot(
                v,
                e_parallel,
            )
        ),
        float(
            np.dot(
                v,
                e_perp,
            )
        ),
    )


def from_parallel_perp(
    ppar,
    pperp,
    e_parallel,
    e_perp,
):

    return (
        float(ppar)
        * e_parallel
        +
        float(pperp)
        * e_perp
    )


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Reconstruct exact dense Roman/L2 parallax geometry
# ============================================================

_, t0 = (
    base.roman.build_roman_times(
        tE_true=TE_TRUE
    )
)

t = np.linspace(
    t0 - seq.WINDOW_TE * TE_TRUE,
    t0 + seq.WINDOW_TE * TE_TRUE,
    N_TIME,
)

evaluator = (
    par.PSPLParallaxEvaluator(
        t=t,
        t0_reference=t0,
    )
)

tel = get_telescope(
    evaluator
)

delta = np.asarray(
    tel.deltas_positions[
        "photometry"
    ],
    dtype=float,
)

print(
    "deltas_positions shape =",
    delta.shape,
)

if delta.shape != (
    2,
    len(t),
):
    raise RuntimeError(
        f"Unexpected delta shape: {delta.shape}"
    )


# ============================================================
# Stability of alpha / jerk with fitting window
# ============================================================

kinematics = {}

kin_rows = []

for window in WINDOWS_DAYS:

    k = observer_kinematics(
        t=t,
        t0=t0,
        delta=delta,
        window_days=window,
    )

    kinematics[
        window
    ] = k

    pi_j_true_tE = (
        (4.0 / 3.0)
        * k["jerk"]
        /
        (
            k["alpha_norm"] ** 2
            * TE_TRUE
        )
    )

    pi_j_perp = float(
        np.dot(
            pi_j_true_tE,
            k["e_perp"],
        )
    )

    kin_rows.append(
        {
            "window_days":
                window,

            "alpha_N":
                k["alpha_N"],

            "alpha_E":
                k["alpha_E"],

            "alpha_norm":
                k["alpha_norm"],

            "jerk_N":
                k["jerk_N"],

            "jerk_E":
                k["jerk_E"],

            "epar_N":
                k["epar_N"],

            "epar_E":
                k["epar_E"],

            "eperp_N":
                k["eperp_N"],

            "eperp_E":
                k["eperp_E"],

            "pi_j_perp_tE30":
                pi_j_perp,

            "c0_norm":
                float(
                    np.hypot(
                        k["c0_N"],
                        k["c0_E"],
                    )
                ),

            "c1_norm":
                float(
                    np.hypot(
                        k["c1_N"],
                        k["c1_E"],
                    )
                ),
        }
    )


kin_df = pd.DataFrame(
    kin_rows
)

kin_df.to_csv(
    OUTDIR
    / "observer_kinematics.csv",
    index=False,
)

print()
print("=" * 110)
print("OBSERVER KINEMATICS")
print("=" * 110)

print(
    kin_df.to_string(
        index=False
    )
)


# ============================================================
# Load empirical branch minima
# ============================================================

df = pd.read_csv(
    MINIMA_PATH
)

df["u0_sign"] = np.where(
    df["u0"] >= 0.0,
    +1,
    -1,
)

k = kinematics[
    MAIN_WINDOW
]

epar = k[
    "e_parallel"
]

eperp = k[
    "e_perp"
]


# Project all recovered solutions.
parallel = []
perpendicular = []

for _, row in df.iterrows():

    ppar, pperp = project_piE(
        row["piEN"],
        row["piEE"],
        epar,
        eperp,
    )

    parallel.append(
        ppar
    )

    perpendicular.append(
        pperp
    )

df[
    "piE_parallel"
] = parallel

df[
    "piE_perp"
] = perpendicular


# ============================================================
# Compare two best minima for each P and each u0 sign
# ============================================================

pair_rows = []

for (
    P_over_tE,
    u0_sign,
), group in df.groupby(
    [
        "P_over_tE",
        "u0_sign",
    ]
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

    # Use mean fitted tE for the approximate Gould mapping.
    tE_pair = 0.5 * (
        float(a["tE"])
        + float(b["tE"])
    )

    pi_j = (
        (4.0 / 3.0)
        * k["jerk"]
        /
        (
            k["alpha_norm"] ** 2
            * tE_pair
        )
    )

    pi_j_perp = float(
        np.dot(
            pi_j,
            eperp,
        )
    )

    ppar_a = float(
        a["piE_parallel"]
    )

    pperp_a = float(
        a["piE_perp"]
    )

    ppar_b = float(
        b["piE_parallel"]
    )

    pperp_b = float(
        b["piE_perp"]
    )

    # --------------------------------------------------------
    # Gould sign convention:
    #
    # ppar'  = ppar
    # pperp' = -(pperp + pi_j_perp)
    # --------------------------------------------------------

    pred_raw_ppar = (
        ppar_a
    )

    pred_raw_pperp = -(
        pperp_a
        + pi_j_perp
    )

    pred_raw_NE = (
        from_parallel_perp(
            pred_raw_ppar,
            pred_raw_pperp,
            epar,
            eperp,
        )
    )

    raw_NE_error = float(
        np.hypot(
            pred_raw_NE[0]
            - float(
                b["piEN"]
            ),

            pred_raw_NE[1]
            - float(
                b["piEE"]
            ),
        )
    )

    raw_parallel_error = (
        ppar_b
        - ppar_a
    )

    raw_perp_relation = (
        pperp_a
        + pperp_b
        + pi_j_perp
    )

    # --------------------------------------------------------
    # Opposite delta-s sign convention.
    # --------------------------------------------------------

    pred_flip_pperp = -(
        pperp_a
        - pi_j_perp
    )

    pred_flip_NE = (
        from_parallel_perp(
            ppar_a,
            pred_flip_pperp,
            epar,
            eperp,
        )
    )

    flip_NE_error = float(
        np.hypot(
            pred_flip_NE[0]
            - float(
                b["piEN"]
            ),

            pred_flip_NE[1]
            - float(
                b["piEE"]
            ),
        )
    )

    flip_perp_relation = (
        pperp_a
        + pperp_b
        - pi_j_perp
    )

    pair_rows.append(
        {
            "P_over_tE":
                float(
                    P_over_tE
                ),

            "u0_sign":
                int(
                    u0_sign
                ),

            "D_a":
                float(
                    a["D"]
                ),

            "D_b":
                float(
                    b["D"]
                ),

            "piEN_a":
                float(
                    a["piEN"]
                ),

            "piEE_a":
                float(
                    a["piEE"]
                ),

            "piEN_b":
                float(
                    b["piEN"]
                ),

            "piEE_b":
                float(
                    b["piEE"]
                ),

            "parallel_a":
                ppar_a,

            "parallel_b":
                ppar_b,

            "perp_a":
                pperp_a,

            "perp_b":
                pperp_b,

            "pi_j_perp":
                pi_j_perp,

            "parallel_error":
                raw_parallel_error,

            "raw_perp_relation_error":
                raw_perp_relation,

            "raw_NE_error":
                raw_NE_error,

            "flip_perp_relation_error":
                flip_perp_relation,

            "flip_NE_error":
                flip_NE_error,
        }
    )


pairs = pd.DataFrame(
    pair_rows
)

pairs.to_csv(
    OUTDIR
    / "jerk_pair_test.csv",
    index=False,
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 110)
print("ROTATION BASIS")
print("=" * 110)

print(
    "parallel (N,E) =",
    epar,
)

print(
    "perp     (N,E) =",
    eperp,
)

print()

print("=" * 110)
print("JERK PAIR TEST")
print("=" * 110)

if len(pairs):

    cols = [
        "P_over_tE",
        "u0_sign",
        "pi_j_perp",
        "parallel_error",
        "raw_perp_relation_error",
        "raw_NE_error",
        "flip_perp_relation_error",
        "flip_NE_error",
    ]

    print(
        pairs[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print("MEDIAN ERRORS")
    print(
        "raw NE   =",
        np.median(
            pairs[
                "raw_NE_error"
            ]
        ),
    )

    print(
        "flip NE  =",
        np.median(
            pairs[
                "flip_NE_error"
            ]
        ),
    )

    print(
        "parallel =",
        np.median(
            np.abs(
                pairs[
                    "parallel_error"
                ]
            )
        ),
    )

else:

    print(
        "No groups contained "
        "two same-sign minima."
    )

print()
print(
    "saved:",
    OUTDIR
    / "observer_kinematics.csv"
)

print(
    "saved:",
    OUTDIR
    / "jerk_pair_test.csv"
)
