#!/usr/bin/env python3

import numpy as np

from binary_source.analysis import (
    intrinsic_fit_tools as base,
)
from binary_source.analysis import (
    intrinsic_parallax as par,
)


TE = 30.0

PIEN = -0.24
PIEE = +0.02

WINDOWS = [
    0.20,
    0.30,
    0.40,
    0.45,
]

N_TIME = 9001


def make_params(
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
        values.get(name, None)
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
    p = make_params(
        evaluator,
        t0=t0,
        u0=u0,
        tE=tE,
        piEN=piEN,
        piEE=piEE,
    )

    tr = evaluator.model.sources_trajectory(
        evaluator.telescope,
        p,
        data_type="photometry",
    )

    x = np.asarray(
        tr[0],
        dtype=float,
    )

    y = np.asarray(
        tr[1],
        dtype=float,
    )

    return x*x + y*y


def local_coeffs(
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

    # Fit in well-conditioned coordinate s.
    a = (
        np.polynomial.polynomial.polyfit(
            s,
            u2[use],
            deg=6,
        )
    )

    # Convert to z=(t-t0)/TE.
    C = np.array(
        [
            a[k]
            * (
                TE
                / float(window)
            )**k
            for k in range(
                len(a)
            )
        ],
        dtype=float,
    )

    return C


def observer_geometry(
    evaluator,
    t,
    t0,
    window=0.45,
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
        <= float(window)
    )

    s = (
        dt[use]
        / float(window)
    )

    coeffs = []

    for component in range(2):

        a = (
            np.polynomial.polynomial.polyfit(
                s,
                delta[
                    component,
                    use,
                ],
                deg=6,
            )
        )

        c = np.array(
            [
                a[k]
                / float(window)**k
                for k
                in range(
                    len(a)
                )
            ]
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
            * TE
        )
    )

    pi_j_perp = float(
        np.dot(
            pi_j,
            eperp,
        )
    )

    return (
        alpha,
        epar,
        eperp,
        pi_j_perp,
    )


def to_NE(
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


_, t0 = (
    base.roman.build_roman_times(
        tE_true=TE
    )
)


# Stay entirely inside one ephemeris interval.
t = np.linspace(
    t0 - 0.45,
    t0 + 0.45,
    N_TIME,
)


evaluator = (
    par.PSPLParallaxEvaluator(
        t=t,
        t0_reference=t0,
    )
)


(
    alpha,
    epar,
    eperp,
    pi_j_perp,
) = observer_geometry(
    evaluator,
    t,
    t0,
)


truth_pi = np.array(
    [
        PIEN,
        PIEE,
    ],
    dtype=float,
)


ppar = float(
    np.dot(
        truth_pi,
        epar,
    )
)

pperp = float(
    np.dot(
        truth_pi,
        eperp,
    )
)


eq21_pperp = -(
    pperp
    + pi_j_perp
)

eq21_NE = to_NE(
    ppar,
    eq21_pperp,
    epar,
    eperp,
)


print("=" * 100)
print("GEOMETRY")
print("=" * 100)

print("alpha       =", alpha)
print("e_parallel  =", epar)
print("e_perp      =", eperp)

print(
    "truth parallel =",
    ppar,
)

print(
    "truth perp     =",
    pperp,
)

print(
    "pi_j_perp     =",
    pi_j_perp,
)

print(
    "Eq21 perp     =",
    eq21_pperp,
)

print(
    "Eq21 NE       =",
    eq21_NE,
)


# ============================================================
# Truth trajectory at EXACT u0 = 0
# ============================================================

u2_truth = trajectory_u2(
    evaluator,
    t0=t0,
    u0=0.0,
    tE=TE,
    piEN=PIEN,
    piEE=PIEE,
)


# ============================================================
# Scan p_perp at fixed p_parallel.
#
# C4 is quadratic in p_perp in the u0 -> 0 Taylor problem.
# We fit that quadratic numerically using pyLIMA itself.
# ============================================================

pperp_grid = np.linspace(
    -0.45,
    +0.20,
    81,
)


print()
print("=" * 100)
print("NUMERICAL C4 SECOND ROOT")
print("=" * 100)


for window in WINDOWS:

    C_truth = local_coeffs(
        t,
        u2_truth,
        t0,
        window,
    )

    C4_values = []
    C3_values = []

    for pp in pperp_grid:

        pi_NE = to_NE(
            ppar,
            pp,
            epar,
            eperp,
        )

        u2 = trajectory_u2(
            evaluator,
            t0=t0,
            u0=0.0,
            tE=TE,
            piEN=pi_NE[0],
            piEE=pi_NE[1],
        )

        C = local_coeffs(
            t,
            u2,
            t0,
            window,
        )

        C3_values.append(
            C[3]
        )

        C4_values.append(
            C[4]
        )


    C4_values = np.asarray(
        C4_values
    )

    C3_values = np.asarray(
        C3_values
    )


    # Fit:
    #
    # C4(p_perp)-C4_truth =
    # q2 p^2 + q1 p + q0
    #
    q = np.polyfit(
        pperp_grid,
        C4_values
        - C_truth[4],
        deg=2,
    )

    roots = np.roots(q)

    real_roots = sorted(
        [
            float(r.real)
            for r in roots
            if abs(r.imag) < 1e-8
        ]
    )

    if len(real_roots) != 2:
        raise RuntimeError(
            f"Expected 2 real C4 roots, got {roots}"
        )


    # Root closest to injected truth is the trivial root.
    root_truth = min(
        real_roots,
        key=lambda x:
            abs(
                x
                - pperp
            ),
    )

    root_second = max(
        real_roots,
        key=lambda x:
            abs(
                x
                - pperp
            ),
    )


    second_NE = to_NE(
        ppar,
        root_second,
        epar,
        eperp,
    )


    effective_pi_j_perp = -(
        pperp
        + root_second
    )


    # Evaluate actual coefficients at numerical second root.
    u2_second = trajectory_u2(
        evaluator,
        t0=t0,
        u0=0.0,
        tE=TE,
        piEN=second_NE[0],
        piEE=second_NE[1],
    )

    C_second = local_coeffs(
        t,
        u2_second,
        t0,
        window,
    )


    # Evaluate coefficients of Eq21 analytic prediction.
    u2_eq21 = trajectory_u2(
        evaluator,
        t0=t0,
        u0=0.0,
        tE=TE,
        piEN=eq21_NE[0],
        piEE=eq21_NE[1],
    )

    C_eq21 = local_coeffs(
        t,
        u2_eq21,
        t0,
        window,
    )


    print()
    print(
        f"window = {window:.2f} d"
    )

    print(
        "C4 roots                  =",
        real_roots,
    )

    print(
        "trivial root              =",
        root_truth,
    )

    print(
        "numerical second p_perp   =",
        root_second,
    )

    print(
        "Eq21 predicted p_perp     =",
        eq21_pperp,
    )

    print(
        "delta p_perp              =",
        root_second
        - eq21_pperp,
    )

    print(
        "numerical second NE       =",
        second_NE,
    )

    print(
        "Eq21 predicted NE         =",
        eq21_NE,
    )

    print(
        "|delta piE|               =",
        np.linalg.norm(
            second_NE
            - eq21_NE
        ),
    )

    print(
        "pi_j_perp from derivatives=",
        pi_j_perp,
    )

    print(
        "pi_j_perp from C4 root    =",
        effective_pi_j_perp,
    )

    print(
        "delta pi_j_perp           =",
        effective_pi_j_perp
        - pi_j_perp,
    )

    print(
        "C3 truth / numerical / Eq21 =",
        C_truth[3],
        C_second[3],
        C_eq21[3],
    )

    print(
        "dC3 numerical             =",
        C_second[3]
        - C_truth[3],
    )

    print(
        "dC3 Eq21                 =",
        C_eq21[3]
        - C_truth[3],
    )

    print(
        "C4 truth / numerical / Eq21 =",
        C_truth[4],
        C_second[4],
        C_eq21[4],
    )

    print(
        "dC4 numerical             =",
        C_second[4]
        - C_truth[4],
    )

    print(
        "dC4 Eq21                 =",
        C_eq21[4]
        - C_truth[4],
    )
