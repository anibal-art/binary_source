#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import intrinsic_bspl_pspl_parallax_scan as seq

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_parallax as par,
)


TE_TRUE = 30.0

PIEN_TRUE = -0.24
PIEE_TRUE = +0.02

U0_VALUES = [
    0.010,
    0.007,
    0.005,
    0.003,
    0.002,
    0.001,
]

N_TIME = 20001
PIE_BOUND = 40.0

OUTDIR = Path(
    "/tmp/jerk_branch_u0_continuation"
)

# Reliable secondary-branch solution already recovered at u0=0.01.
ANCHOR = {
    "truth_u0": 0.01,
    "t0": 2461849.008719672,
    "u0": 0.00790642487084667,
    "tE": 29.990144606115795,
    "piEN": -0.049942335065522404,
    "piEE": 0.024190626569186493,
}


def geometry(
    evaluator,
    t,
    t0,
    tE,
    window_days=20.0,
):
    delta = np.asarray(
        evaluator.telescope.deltas_positions[
            "photometry"
        ],
        dtype=float,
    )

    dt = (
        np.asarray(t)
        - float(t0)
    )

    use = (
        np.abs(dt)
        <= float(window_days)
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

    alpha_vec = (
        2.0
        * coeffs[:, 2]
    )

    jerk_vec = (
        6.0
        * coeffs[:, 3]
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

    pi_j = (
        (4.0 / 3.0)
        * jerk_vec
        /
        (
            alpha**2
            * float(tE)
        )
    )

    pi_j_perp = float(
        np.dot(
            pi_j,
            eperp,
        )
    )

    return {
        "alpha":
            alpha,

        "epar":
            epar,

        "eperp":
            eperp,

        "pi_j_perp":
            pi_j_perp,
    }


def project(
    piEN,
    piEE,
    basis,
):
    return float(
        np.dot(
            np.array(
                [
                    float(piEN),
                    float(piEE),
                ]
            ),
            basis,
        )
    )


def gould_partner(
    geom,
):
    truth = np.array(
        [
            PIEN_TRUE,
            PIEE_TRUE,
        ]
    )

    ppar = float(
        np.dot(
            truth,
            geom["epar"],
        )
    )

    pperp = float(
        np.dot(
            truth,
            geom["eperp"],
        )
    )

    partner_perp = -(
        pperp
        + geom[
            "pi_j_perp"
        ]
    )

    partner = (
        ppar
        * geom["epar"]
        +
        partner_perp
        * geom["eperp"]
    )

    return {
        "piEN":
            float(partner[0]),

        "piEE":
            float(partner[1]),
    }


def encode_start(
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
            (
                float(t0)
                - float(t0_ref)
            )
            / TE_TRUE,

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
    return (
        float(
            t0_ref
            + TE_TRUE
            * x[0]
        ),

        float(
            x[1]
        ),

        float(
            TE_TRUE
            * np.exp(
                x[2]
            )
        ),

        float(
            x[3]
        ),

        float(
            x[4]
        ),
    )


def fit_secondary_branch(
    *,
    evaluator,
    t,
    A_true,
    t0_ref,
    starts,
):
    """
    Targeted continuation fit.

    Important:
    piEN is deliberately restricted to [-0.15, +0.05].
    The injected truth at piEN=-0.24 is therefore excluded.

    This is NOT a global fit. It is a diagnostic continuation
    of the already-established secondary jerk branch.
    """

    weights = (
        seq.trapezoid_weights(
            len(t)
        )
    )

    lower = np.array(
        [
            -0.05,
            1e-8,
            np.log(0.95),
            -0.15,
            -0.10,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            +0.05,
            +0.05,
            np.log(1.05),
            +0.05,
            +0.10,
        ],
        dtype=float,
    )

    def residual(x):

        (
            t0,
            u0,
            tE,
            piEN,
            piEE,
        ) = decode(
            x,
            t0_ref,
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=piEN,
            piEE=piEE,
        )

        return (
            weights
            * (
                A
                - A_true
            )
        )

    candidates = []

    for start_id, x0 in enumerate(
        starts
    ):

        x0 = np.clip(
            np.asarray(
                x0,
                dtype=float,
            ),
            lower + 1e-10,
            upper - 1e-10,
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

        (
            t0,
            u0,
            tE,
            piEN,
            piEE,
        ) = decode(
            opt.x,
            t0_ref,
        )

        A_fit = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=piEN,
            piEE=piEE,
        )

        metric = seq.metrics(
            A_true,
            A_fit,
            t,
        )

        # u0 is allowed to approach its lower bound as u0_true -> 0.
        # Other active bounds make the candidate suspicious.
        active_mask = np.asarray(
            opt.active_mask,
            dtype=int,
        )

        bad_non_u0_bound = bool(
            np.any(
                active_mask[
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
            and np.isfinite(
                metric["D"]
            )
            and np.isfinite(
                opt.optimality
            )
            and float(
                opt.optimality
            )
            < 1e-2
            and not bad_non_u0_bound
        )

        row = {
            "start_id":
                start_id,

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

            "message":
                str(
                    opt.message
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
                    in active_mask
                ),

            "D":
                float(
                    metric["D"]
                ),

            "RMS":
                float(
                    metric["rms"]
                ),

            "t0":
                t0,

            "u0":
                u0,

            "tE":
                tE,

            "piEN":
                piEN,

            "piEE":
                piEE,
        }

        candidates.append(
            row
        )

    good = [
        row
        for row in candidates
        if row[
            "strict_good"
        ]
    ]

    if not good:

        print()
        print(
            "NO STRICTLY CONVERGED "
            "SECONDARY-BRANCH SOLUTION"
        )

        print(
            pd.DataFrame(
                candidates
            )[
                [
                    "start_id",
                    "success",
                    "strict_good",
                    "D",
                    "nfev",
                    "optimality",
                    "active_mask",
                    "u0",
                    "tE",
                    "piEN",
                    "piEE",
                ]
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Continuation stopped: "
            "no numerically trustworthy solution."
        )

    best = min(
        good,
        key=lambda row:
            row["D"],
    )

    return (
        best,
        candidates,
    )


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Dense grid — same high-resolution grid as failed broad sweep
# ============================================================

_, t0_ref = (
    base.roman.build_roman_times(
        tE_true=TE_TRUE
    )
)

t = np.linspace(
    t0_ref
    - seq.WINDOW_TE
    * TE_TRUE,

    t0_ref
    + seq.WINDOW_TE
    * TE_TRUE,

    N_TIME,
)


evaluator = (
    par.PSPLParallaxEvaluator(
        t=t,
        t0_reference=t0_ref,
    )
)


geom = geometry(
    evaluator,
    t,
    t0_ref,
    TE_TRUE,
)

prediction = gould_partner(
    geom
)


print()
print("=" * 110)
print("TARGETED JERK-BRANCH CONTINUATION")
print("=" * 110)

print(
    "analytic Eq21 partner =",
    prediction[
        "piEN"
    ],
    prediction[
        "piEE"
    ],
)

print(
    "pi_j_perp =",
    geom[
        "pi_j_perp"
    ],
)


previous = dict(
    ANCHOR
)

previous_truth_u0 = (
    ANCHOR[
        "truth_u0"
    ]
)


best_rows = []
candidate_rows = []


for truth_u0 in U0_VALUES:

    print()
    print("=" * 110)

    print(
        "TRUTH u0 =",
        truth_u0,
        flush=True,
    )

    print("=" * 110)


    A_true = evaluator.magnification(
        t0=t0_ref,
        u0=float(
            truth_u0
        ),
        tE=TE_TRUE,
        piEN=PIEN_TRUE,
        piEE=PIEE_TRUE,
    )


    ratio = (
        float(
            truth_u0
        )
        / float(
            previous_truth_u0
        )
    )


    # ========================================================
    # Continuation starts
    # ========================================================

    starts = []

    # 1. Smooth continuation of previous solution.
    starts.append(
        encode_start(
            t0=previous["t0"],
            u0=max(
                previous["u0"]
                * ratio,
                1e-7,
            ),
            tE=previous["tE"],
            piEN=previous["piEN"],
            piEE=previous["piEE"],
            t0_ref=t0_ref,
        )
    )

    # 2. Same previous piE but u0 close to current truth.
    starts.append(
        encode_start(
            t0=previous["t0"],
            u0=max(
                0.8
                * truth_u0,
                1e-7,
            ),
            tE=previous["tE"],
            piEN=previous["piEN"],
            piEE=previous["piEE"],
            t0_ref=t0_ref,
        )
    )

    # 3. Allow the partner impact parameter to collapse faster.
    starts.append(
        encode_start(
            t0=previous["t0"],
            u0=max(
                0.2
                * truth_u0,
                1e-7,
            ),
            tE=previous["tE"],
            piEN=previous["piEN"],
            piEE=previous["piEE"],
            t0_ref=t0_ref,
        )
    )

    # 4. Analytic Eq21 branch.
    starts.append(
        encode_start(
            t0=t0_ref,
            u0=max(
                0.8
                * truth_u0,
                1e-7,
            ),
            tE=TE_TRUE,
            piEN=prediction[
                "piEN"
            ],
            piEE=prediction[
                "piEE"
            ],
            t0_ref=t0_ref,
        )
    )

    # 5. Eq21 piE with nearly-zero positive u0.
    starts.append(
        encode_start(
            t0=t0_ref,
            u0=max(
                0.05
                * truth_u0,
                1e-7,
            ),
            tE=TE_TRUE,
            piEN=prediction[
                "piEN"
            ],
            piEE=prediction[
                "piEE"
            ],
            t0_ref=t0_ref,
        )
    )


    best, candidates = (
        fit_secondary_branch(
            evaluator=evaluator,
            t=t,
            A_true=A_true,
            t0_ref=t0_ref,
            starts=starts,
        )
    )


    for row in candidates:

        row[
            "truth_u0"
        ] = truth_u0

        candidate_rows.append(
            row
        )


    eq21_error = float(
        np.hypot(
            best[
                "piEN"
            ]
            - prediction[
                "piEN"
            ],

            best[
                "piEE"
            ]
            - prediction[
                "piEE"
            ],
        )
    )


    ppar_true = project(
        PIEN_TRUE,
        PIEE_TRUE,
        geom[
            "epar"
        ],
    )

    ppar_fit = project(
        best[
            "piEN"
        ],
        best[
            "piEE"
        ],
        geom[
            "epar"
        ],
    )

    pperp_true = project(
        PIEN_TRUE,
        PIEE_TRUE,
        geom[
            "eperp"
        ],
    )

    pperp_fit = project(
        best[
            "piEN"
        ],
        best[
            "piEE"
        ],
        geom[
            "eperp"
        ],
    )


    row = {
        **best,

        "truth_u0":
            truth_u0,

        "eq21_pred_piEN":
            prediction[
                "piEN"
            ],

        "eq21_pred_piEE":
            prediction[
                "piEE"
            ],

        "eq21_NE_error":
            eq21_error,

        "parallel_error":
            ppar_fit
            - ppar_true,

        "perp_relation_error":
            (
                pperp_fit
                + pperp_true
                + geom[
                    "pi_j_perp"
                ]
            ),
    }

    best_rows.append(
        row
    )


    print(
        "best:",
        f"D={best['D']:.9e}",
        f"u0={best['u0']:+.9e}",
        f"tE={best['tE']:.9f}",
        f"piEN={best['piEN']:+.9f}",
        f"piEE={best['piEE']:+.9f}",
        f"nfev={best['nfev']}",
        f"optimality={best['optimality']:.3e}",
    )

    print(
        "Eq21 |delta piE| =",
        f"{eq21_error:.9e}",
    )


    # Anchor sanity check.
    if np.isclose(
        truth_u0,
        0.01,
    ):

        anchor_delta = float(
            np.hypot(
                best[
                    "piEN"
                ]
                - ANCHOR[
                    "piEN"
                ],

                best[
                    "piEE"
                ]
                - ANCHOR[
                    "piEE"
                ],
            )
        )

        print(
            "anchor |delta piE| =",
            anchor_delta,
        )

        if anchor_delta > 3e-3:

            raise RuntimeError(
                "u0=0.01 continuation "
                "does not reproduce known branch."
            )


    previous = dict(
        best
    )

    previous_truth_u0 = (
        truth_u0
    )


best_df = pd.DataFrame(
    best_rows
)

candidate_df = pd.DataFrame(
    candidate_rows
)


best_path = (
    OUTDIR
    / "continuation.csv"
)

candidate_path = (
    OUTDIR
    / "candidates.csv"
)


best_df.to_csv(
    best_path,
    index=False,
)

candidate_df.to_csv(
    candidate_path,
    index=False,
)


print()
print("=" * 110)
print("CONTINUATION SUMMARY")
print("=" * 110)

print(
    best_df[
        [
            "truth_u0",
            "D",
            "u0",
            "tE",
            "piEN",
            "piEE",
            "nfev",
            "optimality",
            "eq21_NE_error",
            "parallel_error",
            "perp_relation_error",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Small-u0 trend
# ============================================================

x = best_df[
    "truth_u0"
].to_numpy(
    dtype=float
)

y = np.abs(
    best_df[
        "eq21_NE_error"
    ].to_numpy(
        dtype=float
    )
)


print()
print("=" * 110)
print("u0 -> 0 TREND")
print("=" * 110)


for i in range(
    len(x)
):

    print(
        f"u0={x[i]:.6g} "
        f"error={y[i]:.9e}"
    )


if len(x) >= 4:

    small = (
        x <= 0.005
    )

    xs = x[small]
    ys = y[small]

    if len(xs) >= 3:

        linear = np.polyfit(
            xs,
            ys,
            1,
        )

        print()
        print(
            "linear small-u0 intercept =",
            linear[1],
        )

        good = (
            (xs > 0.0)
            & (ys > 0.0)
        )

        if np.count_nonzero(
            good
        ) >= 3:

            p, logA = np.polyfit(
                np.log10(
                    xs[good]
                ),
                np.log10(
                    ys[good]
                ),
                1,
            )

            print(
                "power-law exponent p =",
                p,
            )


print()
print(
    "saved:",
    best_path
)

print(
    "saved:",
    candidate_path
)
