#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare old q_f=0 BSPL--PSPL results against the new
magnification-space fitting objective.

Purpose
-------
Check whether the large existing q_f=0 scans can be retained.

The old NPZ files contain:
    t
    P_grid
    D
    BEST_T0U0TE
    SUCCESS
    truth

We reconstruct a sparse representative subset using exactly the
same physical parameters and time sampling.

The comparison is intentionally sparse:
    - several u0 values spanning the scan
    - several P values spanning each P grid
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
# Configuration
# ============================================================

OLD_DIRECTORY = (
    Path.home()
    / "binary_source"
    / "results"
    / "scan_u0_tE150"
)

N_U0_FILES = 5
N_P_PER_FILE = 7


# ============================================================
# Indicative tolerances
#
# These are deliberately not machine-precision tolerances because
# the old objective used a discrete sum whereas the new one uses
# a trapezoidal integral.
# ============================================================

D_ABS_TOL = 1e-5
D_REL_TOL = 5e-3       # 0.5 percent

PARAM_T0_OVER_TE_TOL = 5e-4
PARAM_U0_REL_TOL = 1e-3
PARAM_TE_REL_TOL = 1e-4


# ============================================================
# Helpers
# ============================================================

def flux_to_mag_local(flux, zp=27.615):

    flux = float(flux)

    if not np.isfinite(flux) or flux <= 0.0:
        return 24.0

    return float(
        zp - 2.5 * np.log10(flux)
    )


def select_evenly(indices, n):

    indices = np.asarray(
        indices,
        dtype=int,
    )

    if len(indices) <= n:
        return indices

    positions = np.linspace(
        0,
        len(indices) - 1,
        n,
    )

    positions = np.unique(
        np.round(
            positions
        ).astype(int)
    )

    return indices[
        positions
    ]


def load_record(filename):

    with np.load(
        filename,
        allow_pickle=False,
    ) as d:

        required = [
            "t",
            "P_grid",
            "D",
            "SUCCESS",
            "BEST_T0U0TE",
            "truth",
        ]

        missing = [
            key
            for key in required
            if key not in d.files
        ]

        if missing:
            raise KeyError(
                f"{filename}: missing keys {missing}"
            )

        truth = np.asarray(
            d["truth"],
            dtype=float,
        )

        return {
            "filename": filename,
            "t": np.asarray(
                d["t"],
                dtype=float,
            ),
            "P_grid": np.asarray(
                d["P_grid"],
                dtype=float,
            ),
            "D": np.asarray(
                d["D"],
                dtype=float,
            ),
            "SUCCESS": np.asarray(
                d["SUCCESS"],
                dtype=bool,
            ),
            "BEST": np.asarray(
                d["BEST_T0U0TE"],
                dtype=float,
            ),
            "truth": truth,
            "u0": float(
                truth[1]
            ),
        }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if not OLD_DIRECTORY.exists():

        raise FileNotFoundError(
            OLD_DIRECTORY
        )

    filenames = sorted(
        OLD_DIRECTORY.glob(
            "scan_kepler_u0_*.npz"
        )
    )

    if len(filenames) == 0:

        raise FileNotFoundError(
            f"No NPZ files found in {OLD_DIRECTORY}"
        )


    # ========================================================
    # Load only files containing D and q_f = 0
    # ========================================================

    records = []

    for filename in filenames:

        try:

            rec = load_record(
                filename
            )

        except Exception as error:

            print(
                f"SKIP {filename.name}: {error}"
            )

            continue

        truth = rec[
            "truth"
        ]

        if len(truth) < 10:

            print(
                f"SKIP {filename.name}: "
                "truth vector too short"
            )

            continue

        qf = float(
            truth[8]
        )

        if not np.isclose(
            qf,
            0.0,
            atol=1e-14,
        ):

            print(
                f"SKIP {filename.name}: "
                f"qf={qf}"
            )

            continue

        records.append(
            rec
        )


    if len(records) == 0:

        raise RuntimeError(
            "No valid qf=0 files containing D were found."
        )


    # ========================================================
    # Sort by actual u0 rather than filename
    # ========================================================

    records = sorted(
        records,
        key=lambda rec:
            rec["u0"],
    )


    file_indices = select_evenly(
        np.arange(
            len(records)
        ),
        N_U0_FILES,
    )

    selected_records = [
        records[i]
        for i in file_indices
    ]


    # ========================================================
    # Global diagnostics
    # ========================================================

    rows = []

    print()
    print("=" * 100)
    print("OLD vs NEW q_f=0 VALIDATION")
    print("=" * 100)

    print(
        f"Directory       : {OLD_DIRECTORY}"
    )

    print(
        f"Available files : {len(records)}"
    )

    print(
        f"Selected files  : {len(selected_records)}"
    )

    print(
        f"P per file      : {N_P_PER_FILE}"
    )

    print("=" * 100)
    print()


    # ========================================================
    # Temporary directory for new results
    # ========================================================

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(
            tmp
        )


        # ====================================================
        # Loop over representative old files
        # ====================================================

        for i_file, old in enumerate(
            selected_records
        ):

            truth = old[
                "truth"
            ]

            t = old[
                "t"
            ]

            P_grid_old = old[
                "P_grid"
            ]

            D_old_all = old[
                "D"
            ]

            success_old = old[
                "SUCCESS"
            ]

            best_old_all = old[
                "BEST"
            ]


            # ================================================
            # Truth vector convention
            #
            #  0 t0
            #  1 u0
            #  2 tE
            #  3 phi
            #  4 inclination
            #  5 M1
            #  6 M2
            #  7 rEhat
            #  8 qflux
            #  9 theta
            # 10 fsource
            # 11 fblend
            # 12 old objective flag
            # 13 override xiE (-1 => None)
            # 14 set_flux_from_truth_photometry
            # 15 rms_on_magnification
            # ================================================

            t0_true = float(
                truth[0]
            )

            u0_true = float(
                truth[1]
            )

            tE_true = float(
                truth[2]
            )

            phi_true = float(
                truth[3]
            )

            i_true = float(
                truth[4]
            )

            M1 = float(
                truth[5]
            )

            M2 = float(
                truth[6]
            )

            rEhat = float(
                truth[7]
            )

            qf = float(
                truth[8]
            )

            theta_true = float(
                truth[9]
            )


            # ================================================
            # Recover photometric scale when possible.
            #
            # It does not affect the new intrinsic
            # magnification fit, but keeps the wrapper close
            # to the original simulation.
            # ================================================

            if len(truth) > 10:

                fsource_old = float(
                    truth[10]
                )

            else:

                fsource_old = np.nan


            if len(truth) > 11:

                fblend_old = float(
                    truth[11]
                )

            else:

                fblend_old = 0.0


            msource = flux_to_mag_local(
                fsource_old
            )

            ftotal_approx = (
                fsource_old
                + fblend_old
            )

            mtotal = flux_to_mag_local(
                ftotal_approx
            )


            if len(truth) > 12:

                use_magnification_fit = bool(
                    round(
                        truth[12]
                    )
                )

            else:

                use_magnification_fit = False


            if len(truth) > 13:

                override_raw = float(
                    truth[13]
                )

                override_xiE = (
                    None
                    if override_raw < 0.0
                    else override_raw
                )

            else:

                override_xiE = None


            if len(truth) > 14:

                set_flux_from_truth = bool(
                    round(
                        truth[14]
                    )
                )

            else:

                set_flux_from_truth = True


            if len(truth) > 15:

                rms_on_magnification = bool(
                    round(
                        truth[15]
                    )
                )

            else:

                rms_on_magnification = True


            # ================================================
            # Select only valid old P bins
            # ================================================

            valid_old = (
                success_old
                & np.isfinite(
                    D_old_all
                )
                & np.all(
                    np.isfinite(
                        best_old_all
                    ),
                    axis=1,
                )
            )

            valid_indices = np.flatnonzero(
                valid_old
            )

            if len(valid_indices) == 0:

                print(
                    f"SKIP {old['filename'].name}: "
                    "no valid P points"
                )

                continue


            P_indices = select_evenly(
                valid_indices,
                N_P_PER_FILE,
            )

            P_selected = (
                P_grid_old[
                    P_indices
                ]
            )


            # ================================================
            # Run new objective
            # ================================================

            outfile = (
                tmp
                / f"new_{i_file:03d}.npz"
            )

            run_grid_and_save_npz_kepler(

                out_npz_path=str(
                    outfile
                ),

                t=t,

                t0_true=t0_true,
                u0_true=u0_true,
                tE_true=tE_true,

                phi_true=phi_true,
                i_true=i_true,
                qflux_true=qf,
                theta_true=theta_true,

                M1_Msun=M1,
                M2_Msun=M2,
                rEhat_AU=rEhat,

                P_grid=P_selected,

                msource_true=msource,
                mtotal_true=mtotal,

                use_magnification_fit=(
                    use_magnification_fit
                ),

                override_xiE=override_xiE,

                set_flux_from_truth_photometry=(
                    set_flux_from_truth
                ),

                rms_on_magnification=(
                    rms_on_magnification
                ),

                store_curves=False,
            )


            # ================================================
            # Read new results
            # ================================================

            with np.load(
                outfile,
                allow_pickle=False,
            ) as new:

                D_new = np.asarray(
                    new["D"],
                    dtype=float,
                )

                success_new = np.asarray(
                    new["SUCCESS"],
                    dtype=bool,
                )

                best_new = np.asarray(
                    new["BEST_T0U0TE"],
                    dtype=float,
                )


            # ================================================
            # Compare each selected P
            # ================================================

            print()
            print("-" * 100)

            print(
                f"{old['filename'].name}"
            )

            print(
                f"u0 = {u0_true:.8g}, "
                f"tE = {tE_true:.8g}, "
                f"qf = {qf:.3g}"
            )

            print("-" * 100)

            print(
                " idxP"
                "     P/tE"
                "          D_old"
                "          D_new"
                "        abs_dD"
                "        rel_dD"
                "      |dt0|/tE"
                "       |du0|/u0"
                "       |dtE|/tE"
            )


            for j_new, j_old in enumerate(
                P_indices
            ):

                if not success_new[
                    j_new
                ]:

                    print(
                        f"{j_old:5d} "
                        "NEW FIT FAILED"
                    )

                    rows.append(
                        {
                            "success": False
                        }
                    )

                    continue


                D_old = float(
                    D_old_all[
                        j_old
                    ]
                )

                D_n = float(
                    D_new[
                        j_new
                    ]
                )

                best_old = (
                    best_old_all[
                        j_old
                    ]
                )

                best_n = (
                    best_new[
                        j_new
                    ]
                )


                abs_dD = abs(
                    D_n - D_old
                )


                if abs(D_old) > 1e-10:

                    rel_dD = (
                        abs_dD
                        / abs(D_old)
                    )

                else:

                    rel_dD = np.nan


                dt0_norm = (
                    abs(
                        best_n[0]
                        - best_old[0]
                    )
                    / tE_true
                )


                u0_scale = max(
                    abs(u0_true),
                    1e-6,
                )

                du0_norm = (
                    abs(
                        best_n[1]
                        - best_old[1]
                    )
                    / u0_scale
                )


                dtE_norm = (
                    abs(
                        best_n[2]
                        - best_old[2]
                    )
                    / tE_true
                )


                print(
                    f"{j_old:5d} "
                    f"{P_grid_old[j_old]/tE_true:10.4e} "
                    f"{D_old:14.6e} "
                    f"{D_n:14.6e} "
                    f"{abs_dD:12.4e} "
                    f"{rel_dD:12.4e} "
                    f"{dt0_norm:12.4e} "
                    f"{du0_norm:12.4e} "
                    f"{dtE_norm:12.4e}"
                )


                rows.append(
                    {
                        "success": True,
                        "D_old": D_old,
                        "D_new": D_n,
                        "abs_dD": abs_dD,
                        "rel_dD": rel_dD,
                        "dt0_norm": dt0_norm,
                        "du0_norm": du0_norm,
                        "dtE_norm": dtE_norm,
                    }
                )


    # ========================================================
    # Summary
    # ========================================================

    successful_rows = [
        row
        for row in rows
        if row.get(
            "success",
            False,
        )
    ]


    n_fail_fit = (
        len(rows)
        - len(successful_rows)
    )


    if len(successful_rows) == 0:

        raise RuntimeError(
            "No successful comparisons."
        )


    abs_dD = np.array(
        [
            row["abs_dD"]
            for row in successful_rows
        ],
        dtype=float,
    )

    rel_dD = np.array(
        [
            row["rel_dD"]
            for row in successful_rows
        ],
        dtype=float,
    )

    dt0_norm = np.array(
        [
            row["dt0_norm"]
            for row in successful_rows
        ],
        dtype=float,
    )

    du0_norm = np.array(
        [
            row["du0_norm"]
            for row in successful_rows
        ],
        dtype=float,
    )

    dtE_norm = np.array(
        [
            row["dtE_norm"]
            for row in successful_rows
        ],
        dtype=float,
    )


    finite_rel = rel_dD[
        np.isfinite(
            rel_dD
        )
    ]


    max_abs_dD = float(
        np.max(
            abs_dD
        )
    )

    max_rel_dD = (
        float(
            np.max(
                finite_rel
            )
        )
        if len(finite_rel)
        else 0.0
    )

    max_dt0 = float(
        np.max(
            dt0_norm
        )
    )

    max_du0 = float(
        np.max(
            du0_norm
        )
    )

    max_dtE = float(
        np.max(
            dtE_norm
        )
    )


    print()
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Comparisons              : {len(rows)}"
    )

    print(
        f"Successful new fits       : {len(successful_rows)}"
    )

    print(
        f"Failed new fits           : {n_fail_fit}"
    )

    print()

    print(
        f"max |D_new-D_old|         : "
        f"{max_abs_dD:.6e}"
    )

    print(
        f"max relative D difference : "
        f"{max_rel_dD:.6e}"
    )

    print()

    print(
        f"max |dt0_new-dt0_old|/tE : "
        f"{max_dt0:.6e}"
    )

    print(
        f"max |du0_new-du0_old|/u0 : "
        f"{max_du0:.6e}"
    )

    print(
        f"max |dtE_new-dtE_old|/tE : "
        f"{max_dtE:.6e}"
    )

    print("=" * 100)


    # ========================================================
    # PASS / FAIL
    # ========================================================

    D_ok = (
        max_abs_dD < D_ABS_TOL
        or max_rel_dD < D_REL_TOL
    )

    params_ok = (
        max_dt0
        < PARAM_T0_OVER_TE_TOL
        and max_du0
        < PARAM_U0_REL_TOL
        and max_dtE
        < PARAM_TE_REL_TOL
    )

    passed = (
        n_fail_fit == 0
        and D_ok
        and params_ok
    )


    print()

    if passed:

        print(
            "PASS: old q_f=0 results are numerically "
            "consistent with the new intrinsic objective."
        )

        print(
            "The existing q_f=0 scans can likely be retained."
        )

        raise SystemExit(0)

    else:

        print(
            "FAIL / REVIEW REQUIRED"
        )

        if not D_ok:

            print(
                "- D changed more than the provisional tolerance."
            )

        if not params_ok:

            print(
                "- Best-fit PSPL parameters changed more than "
                "the provisional tolerance."
            )

        if n_fail_fit > 0:

            print(
                "- At least one new fit failed."
            )

        print()
        print(
            "Do not discard the old scans yet; inspect the "
            "reported differences first."
        )

        raise SystemExit(1)
