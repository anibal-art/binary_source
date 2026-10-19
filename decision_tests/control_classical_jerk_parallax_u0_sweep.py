#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import intrinsic_bspl_pspl_parallax_scan as seq

from binary_source.analysis import (
    intrinsic_fit_tools as base,
)
from binary_source.analysis import (
    intrinsic_parallax as par,
)


TE_TRUE = 30.0
PIEN_TRUE = -0.24
PIEE_TRUE = +0.02

U0_VALUES = [
    0.10,
    0.03,
    0.01,
    0.003,
    0.001,
]

# Odd N includes t0 exactly.
# At N=20001 the spacing over +/-3.5 tE is ~0.0105 d.
N_TIME = 20001
PIE_BOUND = 40.0

RADII = [
    0.0,
    0.03,
    0.10,
    0.30,
    1.0,
    2.0,
    3.0,
]

N_ANGLES = 8

OUTDIR = Path(
    "/tmp/classical_jerk_parallax_u0_sweep"
)


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

        "pi_j":
            pi_j,

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


def from_basis(
    ppar,
    pperp,
    epar,
    eperp,
):
    return (
        float(ppar)
        * epar
        +
        float(pperp)
        * eperp
    )


def gould_partner(
    piEN,
    piEE,
    geom,
):
    ppar = project(
        piEN,
        piEE,
        geom["epar"],
    )

    pperp = project(
        piEN,
        piEE,
        geom["eperp"],
    )

    # Gould u0 -> 0 jerk-parallax mapping.
    partner_perp = -(
        pperp
        + geom["pi_j_perp"]
    )

    partner = from_basis(
        ppar,
        partner_perp,
        geom["epar"],
        geom["eperp"],
    )

    return {
        "piEN":
            float(partner[0]),

        "piEE":
            float(partner[1]),

        "parallel":
            ppar,

        "perp":
            partner_perp,
    }


def make_starts(
    predicted_partner,
):
    starts = [
        (0.0, 0.0),
    ]

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        N_ANGLES,
        endpoint=False,
    )

    for radius in RADII:

        if radius == 0.0:
            continue

        for angle in angles:

            starts.append(
                (
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                )
            )

    # Explicitly include exact truth and analytic partner.
    starts.extend(
        [
            (
                PIEN_TRUE,
                PIEE_TRUE,
            ),
            (
                predicted_partner["piEN"],
                predicted_partner["piEE"],
            ),
        ]
    )

    unique = []

    for x in starts:

        if not any(
            np.allclose(
                x,
                y,
                atol=1e-12,
                rtol=0.0,
            )
            for y in unique
        ):
            unique.append(x)

    return unique


def fit_one(
    evaluator,
    t,
    A_true,
    t0_ref,
    u0_start,
    piEN_start,
    piEE_start,
):
    weights = seq.trapezoid_weights(
        len(t)
    )

    lower = np.array(
        [
            -seq.WINDOW_TE,
            -5.0,
            np.log(0.05),
            -PIE_BOUND,
            -PIE_BOUND,
        ]
    )

    upper = np.array(
        [
            +seq.WINDOW_TE,
            +5.0,
            np.log(20.0),
            +PIE_BOUND,
            +PIE_BOUND,
        ]
    )

    # Shape encoding used by the intrinsic scan:
    # x0 = dt0/tE_ref, x1=u0, x2=log(tE/tE_ref).
    x0 = np.array(
        [
            0.0,
            float(u0_start),
            0.0,
            float(piEN_start),
            float(piEE_start),
        ]
    )

    x0 = np.clip(
        x0,
        lower + 1e-10,
        upper - 1e-10,
    )

    def residual(x):

        t0, u0, tE = seq.decode_shape(
            x[:3],
            t0_ref,
            TE_TRUE,
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=float(x[3]),
            piEE=float(x[4]),
        )

        return (
            weights
            * (
                A
                - A_true
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
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=5000,
    )

    t0, u0, tE = seq.decode_shape(
        opt.x[:3],
        t0_ref,
        TE_TRUE,
    )

    A_fit = evaluator.magnification(
        t0=t0,
        u0=u0,
        tE=tE,
        piEN=float(opt.x[3]),
        piEE=float(opt.x[4]),
    )

    metric = seq.metrics(
        A_true,
        A_fit,
        t,
    )

    return {
        "success":
            bool(opt.success),

        "D":
            float(metric["D"]),

        "RMS":
            float(metric["rms"]),

        "t0":
            float(t0),

        "u0":
            float(u0),

        "tE":
            float(tE),

        "piEN":
            float(opt.x[3]),

        "piEE":
            float(opt.x[4]),

        "piE":
            float(
                np.hypot(
                    opt.x[3],
                    opt.x[4],
                )
            ),

        "nfev":
            int(opt.nfev),

        "optimality":
            float(opt.optimality),

        "start_u0":
            float(u0_start),

        "start_piEN":
            float(piEN_start),

        "start_piEE":
            float(piEE_start),
    }


def cluster(
    rows,
    tol=2e-3,
):
    valid = [
        r
        for r in rows
        if (
            r["success"]
            and np.isfinite(
                r["D"]
            )
        )
    ]

    valid.sort(
        key=lambda r:
            r["D"]
    )

    groups = []

    for row in valid:

        sign = (
            +1
            if row["u0"] >= 0.0
            else -1
        )

        matched = False

        for g in groups:

            if sign != g["sign"]:
                continue

            d = np.hypot(
                row["piEN"]
                - g["best"]["piEN"],

                row["piEE"]
                - g["best"]["piEE"],
            )

            if d <= tol:

                g["n"] += 1
                matched = True
                break

        if not matched:

            groups.append(
                {
                    "sign":
                        sign,

                    "best":
                        row,

                    "n":
                        1,
                }
            )

    out = []

    for rank, g in enumerate(
        groups,
        start=1,
    ):

        row = dict(
            g["best"]
        )

        row[
            "rank"
        ] = rank

        row[
            "cluster_size"
        ] = g["n"]

        row[
            "u0_sign"
        ] = g["sign"]

        out.append(row)

    return out


def pylima_parameters(
    evaluator,
    row,
):
    values = {
        "t0":
            float(row["t0"]),

        "u0":
            float(row["u0"]),

        "tE":
            float(row["tE"]),

        "piEN":
            float(row["piEN"]),

        "piEE":
            float(row["piEE"]),
    }

    vector = [
        values.get(
            name,
            None,
        )
        for name
        in evaluator.model.model_dictionnary.keys()
    ]

    return evaluator.model.compute_pyLIMA_parameters(
        vector
    )


def trajectory_u2(
    evaluator,
    row,
):
    params = pylima_parameters(
        evaluator,
        row,
    )

    trajectory = (
        evaluator.model.sources_trajectory(
            evaluator.telescope,
            params,
            data_type="photometry",
        )
    )

    x = np.asarray(
        trajectory[0],
        dtype=float,
    )

    y = np.asarray(
        trajectory[1],
        dtype=float,
    )

    return (
        x * x
        + y * y
    )


def local_coefficients(
    t,
    u2,
    t0_ref,
    window=0.4,
):
    dt = (
        np.asarray(t)
        - float(t0_ref)
    )

    use = (
        np.abs(dt)
        <= window
    )

    s = (
        dt[use]
        / window
    )

    a = (
        np.polynomial.polynomial.polyfit(
            s,
            u2[use],
            deg=6,
        )
    )

    # Convert to z=(t-t0)/TE_TRUE.
    C = np.array(
        [
            a[k]
            * (
                TE_TRUE
                / window
            )**k
            for k in range(
                len(a)
            )
        ]
    )

    return C


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Common geometry
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

# Sanity check of shape encoding.
tt, uu, ee = seq.decode_shape(
    np.array(
        [
            0.0,
            0.1,
            0.0,
        ]
    ),
    t0_ref,
    TE_TRUE,
)

assert np.isclose(
    tt,
    t0_ref,
)

assert np.isclose(
    uu,
    0.1,
)

assert np.isclose(
    ee,
    TE_TRUE,
)


geom = geometry(
    evaluator,
    t,
    t0_ref,
    TE_TRUE,
)

prediction = gould_partner(
    PIEN_TRUE,
    PIEE_TRUE,
    geom,
)

print()
print("=" * 110)
print("CLASSICAL JERK-PARALLAX CONTROL")
print("=" * 110)

print(
    "truth piE       =",
    PIEN_TRUE,
    PIEE_TRUE,
)

print(
    "alpha           =",
    geom["alpha"],
)

print(
    "e_parallel NE   =",
    geom["epar"],
)

print(
    "e_perp NE       =",
    geom["eperp"],
)

print(
    "pi_j_perp       =",
    geom["pi_j_perp"],
)

print(
    "predicted branch=",
    prediction["piEN"],
    prediction["piEE"],
)


all_rows = []
cluster_rows = []
summary_rows = []


# ============================================================
# Controls
# ============================================================

for u0_true in U0_VALUES:

    A_true = evaluator.magnification(
        t0=t0_ref,
        u0=u0_true,
        tE=TE_TRUE,
        piEN=PIEN_TRUE,
        piEE=PIEE_TRUE,
    )

    starts = make_starts(
        prediction
    )

    rows = []

    for sign in (
        +1.0,
        -1.0,
    ):

        for (
            start_id,
            (
                piEN0,
                piEE0,
            ),
        ) in enumerate(
            starts
        ):

            result = fit_one(
                evaluator=evaluator,
                t=t,
                A_true=A_true,
                t0_ref=t0_ref,
                u0_start=(
                    sign
                    * abs(
                        u0_true
                    )
                ),
                piEN_start=piEN0,
                piEE_start=piEE0,
            )

            result[
                "truth_u0"
            ] = u0_true

            result[
                "start_id"
            ] = start_id

            rows.append(
                result
            )

            all_rows.append(
                result
            )

    groups = cluster(
        rows
    )

    for g in groups:

        g[
            "truth_u0"
        ] = u0_true

        cluster_rows.append(
            g
        )


    print()
    print("=" * 110)

    print(
        f"TRUTH u0={u0_true}"
    )

    print("=" * 110)

    for g in groups[:8]:

        print(
            f"rank={g['rank']:2d} "
            f"n={g['cluster_size']:2d} "
            f"D={g['D']:.10e} "
            f"u0={g['u0']:+.8f} "
            f"tE={g['tE']:.8f} "
            f"piEN={g['piEN']:+.8f} "
            f"piEE={g['piEE']:+.8f}"
        )


    same_sign = [
        g
        for g in groups
        if g["u0_sign"] > 0
    ]

    if len(
        same_sign
    ) < 2:
        print(
            "WARNING: fewer than two "
            "positive-u0 minima."
        )
        continue


    # Branch closest to injected truth.
    truth_branch = min(
        same_sign,
        key=lambda g:
            np.hypot(
                g["piEN"]
                - PIEN_TRUE,

                g["piEE"]
                - PIEE_TRUE,
            ),
    )


    other = [
        g
        for g in same_sign
        if g is not truth_branch
    ]


    partner_branch = min(
        other,
        key=lambda g:
            np.hypot(
                g["piEN"]
                - prediction[
                    "piEN"
                ],

                g["piEE"]
                - prediction[
                    "piEE"
                ],
            ),
    )


    # --------------------------------------------------------
    # Eq. 21
    # --------------------------------------------------------

    truth_parallel = project(
        truth_branch[
            "piEN"
        ],
        truth_branch[
            "piEE"
        ],
        geom[
            "epar"
        ],
    )

    partner_parallel = project(
        partner_branch[
            "piEN"
        ],
        partner_branch[
            "piEE"
        ],
        geom[
            "epar"
        ],
    )

    truth_perp = project(
        truth_branch[
            "piEN"
        ],
        truth_branch[
            "piEE"
        ],
        geom[
            "eperp"
        ],
    )

    partner_perp = project(
        partner_branch[
            "piEN"
        ],
        partner_branch[
            "piEE"
        ],
        geom[
            "eperp"
        ],
    )

    eq21_parallel_error = (
        partner_parallel
        - truth_parallel
    )

    eq21_perp_error = (
        partner_perp
        + truth_perp
        + geom[
            "pi_j_perp"
        ]
    )

    eq21_NE_error = float(
        np.hypot(
            partner_branch[
                "piEN"
            ]
            - prediction[
                "piEN"
            ],

            partner_branch[
                "piEE"
            ]
            - prediction[
                "piEE"
            ],
        )
    )


    # --------------------------------------------------------
    # Eq. 22 finite-u0 tE correction
    # --------------------------------------------------------

    du0 = (
        partner_branch[
            "u0"
        ]
        - truth_branch[
            "u0"
        ]
    )

    dpi_perp = (
        partner_perp
        - truth_perp
    )

    observed_dtE = (
        partner_branch[
            "tE"
        ]
        - truth_branch[
            "tE"
        ]
    )

    predicted_dtE = -(
        geom["alpha"]
        * truth_branch[
            "tE"
        ]**3
        / 2.0
    ) * (
        truth_perp
        * du0
        +
        truth_branch[
            "u0"
        ]
        * dpi_perp
    )


    # --------------------------------------------------------
    # Same C2-C4 trajectory test used on BSPL branches
    # --------------------------------------------------------

    u2_truth = trajectory_u2(
        evaluator,
        truth_branch,
    )

    u2_partner = trajectory_u2(
        evaluator,
        partner_branch,
    )

    C_truth = local_coefficients(
        t,
        u2_truth,
        t0_ref,
        window=0.4,
    )

    C_partner = local_coefficients(
        t,
        u2_partner,
        t0_ref,
        window=0.4,
    )


    summary = {
        "truth_u0":
            u0_true,

        "truth_branch_D":
            truth_branch["D"],

        "partner_D":
            partner_branch["D"],

        "truth_piEN_fit":
            truth_branch[
                "piEN"
            ],

        "truth_piEE_fit":
            truth_branch[
                "piEE"
            ],

        "partner_piEN":
            partner_branch[
                "piEN"
            ],

        "partner_piEE":
            partner_branch[
                "piEE"
            ],

        "predicted_piEN":
            prediction[
                "piEN"
            ],

        "predicted_piEE":
            prediction[
                "piEE"
            ],

        "eq21_NE_error":
            eq21_NE_error,

        "eq21_parallel_error":
            eq21_parallel_error,

        "eq21_perp_error":
            eq21_perp_error,

        "observed_delta_tE":
            observed_dtE,

        "predicted_delta_tE":
            predicted_dtE,

        "eq22_relative_error":
            (
                (
                    predicted_dtE
                    - observed_dtE
                )
                / observed_dtE
                if observed_dtE != 0.0
                else np.nan
            ),

        "C2_truth":
            C_truth[2],

        "C2_partner":
            C_partner[2],

        "dC2":
            C_partner[2]
            - C_truth[2],

        "C3_truth":
            C_truth[3],

        "C3_partner":
            C_partner[3],

        "dC3":
            C_partner[3]
            - C_truth[3],

        "C4_truth":
            C_truth[4],

        "C4_partner":
            C_partner[4],

        "dC4":
            C_partner[4]
            - C_truth[4],
    }

    summary_rows.append(
        summary
    )


    print()
    print(
        "Injected/truth branch:"
    )

    print(
        truth_branch
    )

    print()
    print(
        "Recovered jerk partner:"
    )

    print(
        partner_branch
    )

    print()
    print(
        "Eq21 predicted piE =",
        prediction[
            "piEN"
        ],
        prediction[
            "piEE"
        ],
    )

    print(
        "Eq21 recovered-predicted |dpiE| =",
        eq21_NE_error,
    )

    print(
        "Eq21 parallel error =",
        eq21_parallel_error,
    )

    print(
        "Eq21 perp relation error =",
        eq21_perp_error,
    )

    print()
    print(
        "Eq22 observed dtE =",
        observed_dtE,
    )

    print(
        "Eq22 predicted dtE =",
        predicted_dtE,
    )

    print()
    print(
        "Taylor differences:"
    )

    print(
        "dC2 =",
        summary["dC2"],
    )

    print(
        "dC3 =",
        summary["dC3"],
    )

    print(
        "dC4 =",
        summary["dC4"],
    )


# ============================================================
# Save
# ============================================================

pd.DataFrame(
    all_rows
).to_csv(
    OUTDIR
    / "all_starts.csv",
    index=False,
)

pd.DataFrame(
    cluster_rows
).to_csv(
    OUTDIR
    / "distinct_minima.csv",
    index=False,
)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTDIR
    / "control_summary.csv",
    index=False,
)


print()
print("=" * 110)
print("CONTROL SUMMARY")
print("=" * 110)

if len(summary_df):

    print(
        summary_df[
            [
                "truth_u0",
                "truth_branch_D",
                "partner_D",
                "partner_piEN",
                "partner_piEE",
                "predicted_piEN",
                "predicted_piEE",
                "eq21_NE_error",
                "eq21_parallel_error",
                "eq21_perp_error",
                "observed_delta_tE",
                "predicted_delta_tE",
                "eq22_relative_error",
                "dC2",
                "dC3",
                "dC4",
            ]
        ].to_string(
            index=False
        )
    )

print()
print(
    "saved:",
    OUTDIR
)
