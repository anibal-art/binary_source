#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import intrinsic_bspl_pspl_parallax_scan as seq

from binary_source.analysis import intrinsic_fit_tools as base
from binary_source.analysis import intrinsic_parallax as par


U0_TRUE = 0.1
TE_TRUE = 30.0
PIE_BOUND = 40.0
N_TIME = 5000

P_OVER_TE_VALUES = [
    12.67,
    13.69,
    14.79,
    20.16,
    32.07,
    34.65,
    37.44,
]

RADII = [
    0.0,
    0.03,
    0.1,
    0.3,
    1.0,
    2.0,
    3.0,
]

ANGLES = np.linspace(
    0.0,
    2.0 * np.pi,
    8,
    endpoint=False,
)


def make_starts():

    starts = [(0.0, 0.0)]

    for r in RADII:

        if r == 0.0:
            continue

        for a in ANGLES:
            starts.append(
                (
                    r * np.cos(a),
                    r * np.sin(a),
                )
            )

    return starts


def cluster(rows, tol=3e-3):

    rows = sorted(
        rows,
        key=lambda x: x["D"],
    )

    groups = []

    for row in rows:

        sign = np.sign(row["u0"])

        found = False

        for g in groups:

            if sign != g["sign"]:
                continue

            d = np.hypot(
                row["piEN"] - g["row"]["piEN"],
                row["piEE"] - g["row"]["piEE"],
            )

            if d < tol:
                g["n"] += 1
                found = True
                break

        if not found:

            groups.append(
                {
                    "sign": sign,
                    "row": row,
                    "n": 1,
                }
            )

    out = []

    for rank, g in enumerate(
        groups,
        start=1,
    ):

        r = dict(g["row"])

        r["rank"] = rank
        r["cluster_size"] = g["n"]

        out.append(r)

    return out


_, t0 = base.roman.build_roman_times(
    tE_true=TE_TRUE
)

t = np.linspace(
    t0 - seq.WINDOW_TE * TE_TRUE,
    t0 + seq.WINDOW_TE * TE_TRUE,
    N_TIME,
)

bspl = base.PhysicalCircularBSPL(
    t=t,
    t0_reference=t0,
)

pspl_pi = par.PSPLParallaxEvaluator(
    t=t,
    t0_reference=t0,
)

weights = seq.trapezoid_weights(
    len(t)
)

starts = make_starts()

all_rows = []
cluster_rows = []


for P_over_tE in P_OVER_TE_VALUES:

    P_days = P_over_tE * TE_TRUE
    omega = 2.0 * np.pi / P_days

    A_true = bspl.magnification(
        t0=t0,
        u0=U0_TRUE,
        tE=TE_TRUE,
        omega=omega,
    )

    fit0 = seq.fit_pspl_no_parallax(
        t=t,
        A_true=A_true,
        t0_true=t0,
        u0_true=U0_TRUE,
        tE_true=TE_TRUE,
    )

    base_shape = np.asarray(
        fit0["x"],
        dtype=float,
    )

    rows = []

    for u0_sign in (+1.0, -1.0):

        for start_id, (piEN0, piEE0) in enumerate(starts):

            x0 = np.concatenate(
                [
                    base_shape.copy(),
                    [piEN0, piEE0],
                ]
            )

            x0[1] = (
                u0_sign
                * abs(base_shape[1])
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

            x0 = np.clip(
                x0,
                lower + 1e-10,
                upper - 1e-10,
            )

            def residual(x):

                ft0, fu0, ftE = seq.decode_shape(
                    x[:3],
                    t0,
                    TE_TRUE,
                )

                A = pspl_pi.magnification(
                    t0=ft0,
                    u0=fu0,
                    tE=ftE,
                    piEN=x[3],
                    piEE=x[4],
                )

                return weights * (
                    A - A_true
                )

            opt = least_squares(
                residual,
                x0,
                bounds=(lower, upper),
                method="trf",
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=5000,
            )

            ft0, fu0, ftE = seq.decode_shape(
                opt.x[:3],
                t0,
                TE_TRUE,
            )

            A_fit = pspl_pi.magnification(
                t0=ft0,
                u0=fu0,
                tE=ftE,
                piEN=opt.x[3],
                piEE=opt.x[4],
            )

            m = seq.metrics(
                A_true,
                A_fit,
                t,
            )

            row = {
                "P_over_tE": P_over_tE,
                "P_days": P_days,
                "start_id": start_id,
                "start_u0_sign": int(u0_sign),
                "start_piEN": piEN0,
                "start_piEE": piEE0,
                "D": m["D"],
                "RMS": m["rms"],
                "t0": ft0,
                "u0": fu0,
                "tE": ftE,
                "piEN": opt.x[3],
                "piEE": opt.x[4],
                "piE": np.hypot(
                    opt.x[3],
                    opt.x[4],
                ),
                "nfev": opt.nfev,
                "success": opt.success,
            }

            rows.append(row)
            all_rows.append(row)

    groups = cluster(
        rows
    )

    for row in groups:
        cluster_rows.append(row)

    print()
    print("=" * 100)
    print("P/tE =", P_over_tE)
    print("=" * 100)

    for row in groups[:8]:

        print(
            f"rank={row['rank']:2d} "
            f"n={row['cluster_size']:2d} "
            f"D={row['D']:.8e} "
            f"u0={row['u0']:+.6f} "
            f"piEN={row['piEN']:+.6f} "
            f"piEE={row['piEE']:+.6f} "
            f"|piE|={row['piE']:.6f}"
        )


pd.DataFrame(
    all_rows
).to_csv(
    "/tmp/parallax_branch_all_starts.csv",
    index=False,
)

pd.DataFrame(
    cluster_rows
).to_csv(
    "/tmp/parallax_branch_minima.csv",
    index=False,
)

print()
print(
    "saved:",
    "/tmp/parallax_branch_all_starts.csv"
)

print(
    "saved:",
    "/tmp/parallax_branch_minima.csv"
)
