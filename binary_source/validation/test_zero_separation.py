#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity test for the intrinsic BSPL--PSPL degeneracy.

Physical requirement:

    xi_rel -> 0

must imply

    A_BSPL(t) -> A_PSPL(t)

for ANY q_M and q_f.

Therefore:

    D -> 0
    DT0 -> 0
    DU0 -> 0
    DTE -> 0

up to numerical precision.

This test is especially important for q_f > 0.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np


# ============================================================
# Project import
# ============================================================

SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from degeneracy_fit import run_grid_and_save_npz_kepler


# ============================================================
# Base event
# ============================================================

t0_true = 50.0
u0_true = 0.1
tE_true = 150.0

t = np.linspace(
    t0_true - 3.5 * tE_true,
    t0_true + 3.5 * tE_true,
    5000,
)


# ============================================================
# Geometry
# ============================================================

phi_true = 0.0
i_true = np.pi / 2.0
theta_true = 0.0


# ============================================================
# System
# ============================================================

Mtot = 3.0
rEhat_AU = 5.0

P_grid = np.array(
    [tE_true],
    dtype=float,
)


# ============================================================
# Cases
# ============================================================

cases = [
    (0.01, 0.0),
    (0.01, 0.01),
    (0.1, 0.0),
    (0.1, 0.1),
    (0.5, 0.0),
    (0.5, 0.5),
    (1.0, 0.0),
    (1.0, 1.0),
]


# ============================================================
# Numerical tolerances
# ============================================================

D_TOL = 1e-8
BIAS_TOL = 1e-7


# ============================================================
# Helper
# ============================================================

def masses_from_qM(qM):

    M1 = Mtot / (1.0 + qM)
    M2 = qM * M1

    return M1, M2


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    failures = []

    print()
    print("=" * 78)
    print("ZERO-SEPARATION SANITY TEST")
    print("=" * 78)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:

        tmpdir = Path(tmpdir)

        for k, (qM, qf) in enumerate(cases):

            M1, M2 = masses_from_qM(qM)

            outfile = tmpdir / f"case_{k:02d}.npz"

            run_grid_and_save_npz_kepler(
                out_npz_path=str(outfile),

                t=t,

                t0_true=t0_true,
                u0_true=u0_true,
                tE_true=tE_true,

                phi_true=phi_true,
                i_true=i_true,
                theta_true=theta_true,

                qflux_true=qf,

                M1_Msun=M1,
                M2_Msun=M2,

                rEhat_AU=rEhat_AU,

                P_grid=P_grid,

                msource_true=24.0,
                mtotal_true=24.0,

                # Exact zero relative source separation
                override_xiE=0.0,

                set_flux_from_truth_photometry=True,
                rms_on_magnification=True,

                store_curves=False,
            )

            with np.load(
                outfile,
                allow_pickle=False,
            ) as d:

                D = float(d["D"][0])
                DT0 = float(d["DT0"][0])
                DU0 = float(d["DU0"][0])
                DTE = float(d["DTE"][0])
                success = bool(d["SUCCESS"][0])

            passed = (
                success
                and abs(D) < D_TOL
                and abs(DT0) < BIAS_TOL
                and abs(DU0) < BIAS_TOL
                and abs(DTE) < BIAS_TOL
            )

            print(
                f"qM={qM:6.3f}  "
                f"qf={qf:6.3f}  "
                f"D={D:12.5e}  "
                f"DT0={DT0:12.5e}  "
                f"DU0={DU0:12.5e}  "
                f"DTE={DTE:12.5e}  "
                f"{'PASS' if passed else 'FAIL'}"
            )

            if not passed:

                failures.append(
                    {
                        "qM": qM,
                        "qf": qf,
                        "D": D,
                        "DT0": DT0,
                        "DU0": DU0,
                        "DTE": DTE,
                    }
                )

    print()
    print("=" * 78)

    if failures:

        print(
            f"FAILED: {len(failures)} "
            f"of {len(cases)} cases"
        )

        print()
        print(
            "This means the fitting pipeline "
            "does not recover the exact PSPL "
            "limit for xi_rel = 0."
        )

        raise SystemExit(1)

    else:

        print(
            f"PASSED: all {len(cases)} cases"
        )

        raise SystemExit(0)
