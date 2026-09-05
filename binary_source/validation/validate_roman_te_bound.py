#!/usr/bin/env python3

"""
Targeted validation of the upper tE bound in the Roman PSPL fits.

The production Roman fit uses

    tE in [0.02, 20] * tE_true.

This script:

1. finds every physical grid cell for which ANY Roman magnitude
   has BEST_TE within 1% of the production upper bound;

2. regenerates exactly those BSPL Asimov light curves;

3. repeats the PSPL fit with upper bounds

       20, 50, 100, 200 * tE_true;

4. compares Delta chi2, D_Roman,w and the fitted PSPL parameters;

5. checks that the factor=20 rerun reproduces the stored production
   solution.

No production files are modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from binary_source.analysis import roman_bspl_pspl_asimov as roman


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ROMAN_DIR = (
    ROOT
    / "results"
    / "roman_asimov"
)

OUTDIR = (
    ROOT
    / "results"
    / "validation_roman_te_bound"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


MAGS = [
    19.0,
    21.0,
    23.0,
]

UPPER_FACTORS = [
    20.0,
    50.0,
    100.0,
    200.0,
]


TE_TRUE = 150.0

Q_MASS = 0.5
Q_FLUX = 0.0
MTOT_MSUN = 3.0
REHAT_AU = 5.0

THETA = 0.0
PHI = 0.0
INCLINATION = np.pi / 2.0


# ============================================================
# Helpers
# ============================================================

def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def scalar(x):

    arr = np.asarray(x)

    if arr.size != 1:

        raise ValueError(
            f"Expected scalar, got shape {arr.shape}"
        )

    return float(
        arr.reshape(-1)[0]
    )


def as_array(x):

    if hasattr(
        x,
        "value",
    ):

        return np.asarray(
            x.value,
            dtype=float,
        )

    return np.asarray(
        x,
        dtype=float,
    )


# ============================================================
# Roman times
# ============================================================

def resolve_roman_times():
    """
    Prefer the already-tested helper from the blending/window
    validation script.

    A fallback parser is provided so this script remains usable
    independently.
    """

    try:

        from binary_source.validation.validate_window_and_blending import (
            resolve_roman_times as resolver,
        )

        return resolver()

    except Exception:

        pass


    result = roman.build_roman_times(
        tE_true=TE_TRUE,
        anchor_season_index=2,
        fit_window_te=3.5,
        include_off_seasons=True,
    )


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        times = None
        t0 = None

        for key in [
            "t",
            "time",
            "times",
            "roman_times",
        ]:

            if key in result:

                arr = np.asarray(
                    result[key],
                    dtype=float,
                )

                if (
                    arr.ndim == 1
                    and arr.size > 10
                ):

                    times = arr
                    break


        for key in [
            "t0",
            "t0_true",
            "anchor_t0",
            "event_t0",
        ]:

            if key in result:

                try:

                    t0 = float(
                        result[key]
                    )

                    break

                except Exception:

                    pass


        if (
            times is not None
            and t0 is not None
        ):

            return (
                times,
                t0,
            )


    # --------------------------------------------------------
    # Tuple/list
    # --------------------------------------------------------

    if isinstance(
        result,
        (
            tuple,
            list,
        ),
    ):

        arrays = []
        scalars = []

        for item in result:

            try:

                arr = np.asarray(
                    item,
                    dtype=float,
                )

            except Exception:

                continue


            if (
                arr.ndim == 1
                and arr.size > 10
            ):

                arrays.append(
                    arr
                )

            elif arr.size == 1:

                scalars.append(
                    float(
                        arr.reshape(-1)[0]
                    )
                )


        if arrays:

            times = max(
                arrays,
                key=len,
            )

            plausible = [
                x
                for x in scalars
                if (
                    np.min(times)
                    - 10.0 * TE_TRUE
                    <= x
                    <= np.max(times)
                    + 10.0 * TE_TRUE
                )
            ]

            if plausible:

                t0 = min(
                    plausible,
                    key=lambda x: abs(
                        x
                        - np.median(
                            times
                        )
                    ),
                )

                return (
                    times,
                    t0,
                )


    raise RuntimeError(
        "Could not resolve Roman times/t0."
    )


# ============================================================
# Custom Roman PSPL fitter
# ============================================================

def fit_pspl_roman_custom_te_bound(
    ev,
    t0_true,
    u0_true,
    tE_true,
    upper_factor,
    initial_guess=None,
):
    """
    Exact copy of the production Roman PSPL logic except that
    the upper tE bound is configurable.

    Production corresponds to upper_factor=20.
    """

    model_pspl = (
        roman.PSPL_model.PSPLmodel(
            ev,
            blend_flux_parameter="ftotal",
        )
    )

    model_pspl.define_model_parameters()


    fit = (
        roman.TRF_fit.TRFfit(
            model_pspl
        )
    )


    # --------------------------------------------------------
    # Bounds: identical to production except tE upper factor
    # --------------------------------------------------------

    if "t0" in fit.fit_parameters:

        fit.fit_parameters[
            "t0"
        ][1] = [
            float(
                t0_true
                - 2.0
                * tE_true
            ),

            float(
                t0_true
                + 2.0
                * tE_true
            ),
        ]


    if "u0" in fit.fit_parameters:

        u_bound = max(
            20.0,
            2.5
            * abs(
                float(
                    u0_true
                )
            ),
        )

        fit.fit_parameters[
            "u0"
        ][1] = [
            -u_bound,
            u_bound,
        ]


    if "tE" in fit.fit_parameters:

        fit.fit_parameters[
            "tE"
        ][1] = [
            0.02
            * float(
                tE_true
            ),

            float(
                upper_factor
            )
            * float(
                tE_true
            ),
        ]


    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    if initial_guess is None:

        initial_guess = [
            float(
                t0_true
            ),

            float(
                u0_true
            ),

            float(
                tE_true
            ),
        ]


    fit.model_parameters_guess = [
        float(
            initial_guess[0]
        ),

        float(
            initial_guess[1]
        ),

        float(
            initial_guess[2]
        ),
    ]


    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fit.fit()


    results = (
        fit.fit_results
    )


    if (
        "best_model"
        not in results
    ):

        raise RuntimeError(
            "pyLIMA did not return best_model."
        )


    best_model = np.asarray(
        results[
            "best_model"
        ],
        dtype=float,
    ).reshape(-1)


    best_t0 = float(
        best_model[0]
    )

    best_u0 = float(
        best_model[1]
    )

    best_tE = float(
        best_model[2]
    )


    # --------------------------------------------------------
    # Recompute chi2 explicitly
    # --------------------------------------------------------

    py_best = (
        model_pspl.compute_pyLIMA_parameters(
            best_model
        )
    )


    telescope = (
        ev.telescopes[0]
    )


    F_model = as_array(
        model_pspl.compute_the_microlensing_model(
            telescope,
            py_best,
        )[
            "photometry"
        ]
    )


    F_data = as_array(
        telescope.lightcurve[
            "flux"
        ]
    )


    sigma_F = as_array(
        telescope.lightcurve[
            "err_flux"
        ]
    )


    valid = (
        np.isfinite(
            F_data
        )
        & np.isfinite(
            F_model
        )
        & np.isfinite(
            sigma_F
        )
        & (
            sigma_F > 0.0
        )
    )


    chi2 = float(
        np.sum(
            (
                (
                    F_data[
                        valid
                    ]
                    - F_model[
                        valid
                    ]
                )
                / sigma_F[
                    valid
                ]
            ) ** 2
        )
    )


    return {
        "fit": fit,
        "model": model_pspl,

        "best_model": (
            best_model
        ),

        "best_t0": (
            best_t0
        ),

        "best_u0": (
            best_u0
        ),

        "best_tE": (
            best_tE
        ),

        "chi2": (
            chi2
        ),
    }


# ============================================================
# Load production grids
# ============================================================

def production_path(
    mag,
):

    return (
        ROMAN_DIR
        / (
            "roman_intrinsic_grid_"
            f"W149_{int(mag)}.npz"
        )
    )


def load_products():

    products = {}


    for mag in MAGS:

        path = (
            production_path(
                mag
            )
        )


        if not path.exists():

            raise FileNotFoundError(
                path
            )


        products[
            mag
        ] = np.load(
            path,
            allow_pickle=False,
        )


    return products


# ============================================================
# Candidate cells
# ============================================================

def find_candidate_cells(
    products,
):
    """
    Union of physical cells that lie within 1% of the production
    upper tE bound for at least one magnitude.
    """

    candidates = set()


    upper = (
        20.0
        * TE_TRUE
    )


    for mag, z in (
        products.items()
    ):

        best_te = np.squeeze(
            np.asarray(
                z[
                    "BEST_TE"
                ],
                dtype=float,
            )
        )


        mask = (
            np.isfinite(
                best_te
            )
            & (
                best_te
                >= 0.99
                * upper
            )
        )


        for i, j in (
            np.argwhere(
                mask
            )
        ):

            candidates.add(
                (
                    int(i),
                    int(j),
                )
            )


    return sorted(
        candidates
    )


# ============================================================
# Run one candidate
# ============================================================

def run_candidate(
    products,
    i,
    j,
    t,
    t0_true,
):

    reference = (
        products[
            MAGS[0]
        ]
    )


    u0_grid = np.asarray(
        reference[
            "u0_grid"
        ],
        dtype=float,
    )


    P_grid = np.asarray(
        reference[
            "P_grid"
        ],
        dtype=float,
    )


    u0_true = float(
        u0_grid[i]
    )

    P_days = float(
        P_grid[j]
    )


    print()
    print("#" * 100)

    print(
        f"CELL {(i, j)}"
    )

    print(
        f"u0       = {u0_true:.12g}"
    )

    print(
        f"P        = {P_days:.12g} d"
    )

    print(
        f"P/tE     = {P_days / TE_TRUE:.12g}"
    )

    print("#" * 100)


    rows = []


    # Truth magnification is independent of source magnitude.
    truth = (
        roman.bspl_truth_magnification(
            t=t,

            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=TE_TRUE,

            P_days=P_days,

            q_mass=Q_MASS,
            qflux=Q_FLUX,

            Mtot_Msun=MTOT_MSUN,
            rEhat_AU=REHAT_AU,

            theta=THETA,
            phi=PHI,
            inclination=INCLINATION,
        )
    )


    A_bspl = np.asarray(
        truth[
            "A_bspl"
        ],
        dtype=float,
    )


    for mag in MAGS:

        z = (
            products[
                mag
            ]
        )


        stored_te = float(
            np.squeeze(
                z[
                    "BEST_TE"
                ]
            )[
                i,
                j,
            ]
        )


        stored_chi2 = float(
            np.squeeze(
                z[
                    "DELTA_CHI2"
                ]
            )[
                i,
                j,
            ]
        )


        stored_Droman = float(
            np.squeeze(
                z[
                    "D_ROMAN_EFF"
                ]
            )[
                i,
                j,
            ]
        )


        asimov = (
            roman.make_roman_asimov_event(
                t=t,
                A_bspl=A_bspl,
                source_mag=float(
                    mag
                ),
            )
        )


        snr = float(
            roman.event_snr(
                telescope=asimov[
                    "telescope"
                ],
                A_truth=A_bspl,
            )
        )


        previous_best = None


        for factor in (
            UPPER_FACTORS
        ):

            # =================================================
            # First run: production-like truth initialization
            # =================================================

            fit = (
                fit_pspl_roman_custom_te_bound(
                    ev=asimov[
                        "event"
                    ],

                    t0_true=t0_true,
                    u0_true=u0_true,
                    tE_true=TE_TRUE,

                    upper_factor=factor,

                    initial_guess=None,
                )
            )


            best_te = float(
                fit[
                    "best_tE"
                ]
            )


            upper_days = (
                factor
                * TE_TRUE
            )


            at_upper = bool(
                abs(
                    best_te
                    - upper_days
                )
                <= max(
                    1.0e-4,
                    1.0e-7
                    * upper_days,
                )
            )


            dchi2 = float(
                fit[
                    "chi2"
                ]
            )


            Droman = float(
                np.sqrt(
                    max(
                        dchi2,
                        0.0,
                    )
                )
                / snr
            )


            row = {
                "i": int(
                    i
                ),

                "j": int(
                    j
                ),

                "W149": float(
                    mag
                ),

                "u0_true": (
                    u0_true
                ),

                "P_days": (
                    P_days
                ),

                "P_over_tE": float(
                    P_days
                    / TE_TRUE
                ),

                "upper_factor": float(
                    factor
                ),

                "upper_tE_days": float(
                    upper_days
                ),

                "best_t0": float(
                    fit[
                        "best_t0"
                    ]
                ),

                "best_u0": float(
                    fit[
                        "best_u0"
                    ]
                ),

                "best_tE": (
                    best_te
                ),

                "at_upper_bound": (
                    at_upper
                ),

                "delta_chi2": (
                    dchi2
                ),

                "D_roman_w": (
                    Droman
                ),

                "snr_event": (
                    snr
                ),

                "stored_best_tE": (
                    stored_te
                ),

                "stored_delta_chi2": (
                    stored_chi2
                ),

                "stored_D_roman_w": (
                    stored_Droman
                ),
            }


            if (
                factor
                == 20.0
            ):

                row[
                    "delta_chi2_over_stored"
                ] = (
                    dchi2
                    / stored_chi2
                )

                row[
                    "Droman_over_stored"
                ] = (
                    Droman
                    / stored_Droman
                )

            else:

                row[
                    "delta_chi2_over_stored"
                ] = np.nan

                row[
                    "Droman_over_stored"
                ] = np.nan


            rows.append(
                row
            )


            previous_best = [
                float(
                    fit[
                        "best_t0"
                    ]
                ),

                float(
                    fit[
                        "best_u0"
                    ]
                ),

                float(
                    fit[
                        "best_tE"
                    ]
                ),
            ]


            print(
                f"W149={mag:4.0f} "
                f"upper={factor:6.0f} tE "
                f"best_tE={best_te:12.5f} "
                f"at_upper={str(at_upper):5s} "
                f"dchi2={dchi2:14.6f} "
                f"Drom={Droman:.8e}"
            )


    return rows


# ============================================================
# Main
# ============================================================

def main():

    section(
        "ROMAN tE UPPER-BOUND VALIDATION"
    )


    t, t0_true = (
        resolve_roman_times()
    )


    print(
        "N Roman epochs =",
        len(t),
    )

    print(
        "t0_true        =",
        t0_true,
    )

    print(
        "tE_true        =",
        TE_TRUE,
    )


    products = (
        load_products()
    )


    candidates = (
        find_candidate_cells(
            products
        )
    )


    section(
        "CANDIDATE PHYSICAL CELLS"
    )


    print(
        "N candidate cells =",
        len(candidates),
    )


    reference = (
        products[
            MAGS[0]
        ]
    )

    u0_grid = np.asarray(
        reference[
            "u0_grid"
        ],
        dtype=float,
    )

    P_grid = np.asarray(
        reference[
            "P_grid"
        ],
        dtype=float,
    )


    for i, j in candidates:

        print(
            f"  {(i, j)} "
            f"u0={u0_grid[i]:.10g} "
            f"P={P_grid[j]:.10g} d "
            f"P/tE={P_grid[j]/TE_TRUE:.10g}"
        )


    all_rows = []


    for i, j in candidates:

        all_rows.extend(
            run_candidate(
                products=products,
                i=i,
                j=j,
                t=t,
                t0_true=t0_true,
            )
        )


    df = pd.DataFrame(
        all_rows
    )


    csv_path = (
        OUTDIR
        / "roman_te_bound_relaxation.csv"
    )


    df.to_csv(
        csv_path,
        index=False,
    )


    # ========================================================
    # Relative-to-production-factor-20 summary
    # ========================================================

    summaries = []


    for (
        i,
        j,
        mag,
    ), sub in df.groupby(
        [
            "i",
            "j",
            "W149",
        ]
    ):

        sub = (
            sub.sort_values(
                "upper_factor"
            )
        )


        ref = sub[
            np.isclose(
                sub[
                    "upper_factor"
                ],
                20.0,
            )
        ].iloc[0]


        final = sub.iloc[
            -1
        ]


        summaries.append(
            {
                "i": int(i),
                "j": int(j),

                "W149": float(
                    mag
                ),

                "u0_true": float(
                    ref[
                        "u0_true"
                    ]
                ),

                "P_days": float(
                    ref[
                        "P_days"
                    ]
                ),

                "best_tE_factor20": float(
                    ref[
                        "best_tE"
                    ]
                ),

                "best_tE_factor200": float(
                    final[
                        "best_tE"
                    ]
                ),

                "factor200_at_bound": bool(
                    final[
                        "at_upper_bound"
                    ]
                ),

                "dchi2_factor20": float(
                    ref[
                        "delta_chi2"
                    ]
                ),

                "dchi2_factor200": float(
                    final[
                        "delta_chi2"
                    ]
                ),

                "dchi2_ratio_200_over_20": float(
                    final[
                        "delta_chi2"
                    ]
                    / ref[
                        "delta_chi2"
                    ]
                ),

                "Droman_factor20": float(
                    ref[
                        "D_roman_w"
                    ]
                ),

                "Droman_factor200": float(
                    final[
                        "D_roman_w"
                    ]
                ),

                "Droman_ratio_200_over_20": float(
                    final[
                        "D_roman_w"
                    ]
                    / ref[
                        "D_roman_w"
                    ]
                ),
            }
        )


    summary_df = pd.DataFrame(
        summaries
    )


    summary_path = (
        OUTDIR
        / "roman_te_bound_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # Console summary
    # ========================================================

    section(
        "BOUND-RELAXATION SUMMARY"
    )


    print(
        summary_df.to_string(
            index=False
        )
    )


    metadata = {
        "upper_factors": (
            UPPER_FACTORS
        ),

        "tE_true": (
            TE_TRUE
        ),

        "candidate_cells": [
            [
                int(i),
                int(j),
            ]
            for i, j
            in candidates
        ],

        "n_candidates": int(
            len(
                candidates
            )
        ),
    }


    (
        OUTDIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


    print()
    print(
        "Saved:",
        csv_path
    )

    print(
        "Saved:",
        summary_path
    )


if __name__ == "__main__":
    main()
