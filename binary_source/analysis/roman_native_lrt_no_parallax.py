#!/usr/bin/env python3

"""
One-realization Roman LRT:

    H0 fit: PSPL
    H1 fit: PSPL + Circular double source

Both fits:
    - no parallax
    - canonical roman_rubin fit_lc.fit_rubin_roman
    - native pyLIMA TRFfit underneath

Simulation:
    - canonical roman_rubin
    - full Roman observing schedule (time_window=None)
    - Roman F146/Pandeia photometric uncertainties
    - Roman ephemerides
    - instrumental noise ON

This module contains no custom least-squares fitter.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


# ============================================================================
# Canonical roman_rubin checkout
# ============================================================================

ROMAN_RUBIN_REPO = Path(
    os.environ.get(
        "ROMAN_RUBIN_REPO",
        "/share/storage3/rubin/microlensing/"
        "romanrubin/roman_rubin",
    )
).resolve()

if not ROMAN_RUBIN_REPO.exists():
    raise RuntimeError(
        f"roman_rubin repository not found: {ROMAN_RUBIN_REPO}"
    )

sys.path.insert(
    0,
    str(ROMAN_RUBIN_REPO),
)

import functions_roman_rubin as frr
import fit_lc
import set_model_pyLIMA as sm



EPHEMERIDES = (
    ROMAN_RUBIN_REPO
    / "ephemerides"
    / "Roman_positions.npy"
)


# ============================================================================
# Fiducial truth physics
# ============================================================================

M1_MSUN_DEFAULT = 2.0
M2_MSUN_DEFAULT = 1.0
REHAT_AU_DEFAULT = 5.0

PHASE_TRUE_DEFAULT = 0.0
INCLINATION_TRUE_DEFAULT = np.pi / 2.0
THETA_TRUE_DEFAULT = 0.0
Q_FLUX_TRUE_DEFAULT = 0.0


# ============================================================================
# Small helpers
# ============================================================================

def scalar(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 1:
        raise ValueError(
            f"Expected scalar, got shape={arr.shape}"
        )
    return float(arr.reshape(-1)[0])


def fitter_name(fit):
    return (
        f"{fit.__class__.__module__}."
        f"{fit.__class__.__name__}"
    )


def assert_native_trf(fit):
    name = fitter_name(fit)

    if name != "pyLIMA.fits.TRF_fit.TRFfit":
        raise RuntimeError(
            "Non-native fitter detected: "
            f"{name}"
        )


def array_hash(a):
    arr = np.ascontiguousarray(
        np.asarray(a, dtype=float)
    )

    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))

    return h.hexdigest()


def a_from_period_days(
    P_days,
    total_mass_msun,
):
    """
    Kepler in AU, yr, Msun:
        a^3 = Mtot P^2
    """
    return (
        float(total_mass_msun)
        * (float(P_days) / 365.25) ** 2
    ) ** (1.0 / 3.0)


def xi_from_period_days(
    P_days,
    total_mass_msun,
    rehat_au,
):
    return (
        a_from_period_days(
            P_days,
            total_mass_msun,
        )
        / float(rehat_au)
    )


def move_inside_bounds(
    value,
    bounds,
):
    value = float(value)
    lo = float(bounds[0])
    hi = float(bounds[1])

    if not (
        np.isfinite(lo)
        and np.isfinite(hi)
        and hi > lo
    ):
        raise RuntimeError(
            f"Invalid bounds: {bounds}"
        )

    width = hi - lo

    eps = max(
        1.0e-9 * width,
        1.0e-10,
    )

    if value <= lo:
        return lo + eps

    if value >= hi:
        return hi - eps

    return value


def set_bound(
    fit,
    parameter,
    lo,
    hi,
):
    if parameter not in fit.fit_parameters:
        raise KeyError(
            f"{parameter} not present in fitter. "
            f"Available: {list(fit.fit_parameters)}"
        )

    fit.fit_parameters[
        parameter
    ][1] = [
        float(lo),
        float(hi),
    ]


def get_parameter(
    py_params,
    name,
):
    if hasattr(
        py_params,
        name,
    ):
        return float(
            getattr(
                py_params,
                name,
            )
        )

    try:
        return float(
            py_params[name]
        )

    except Exception as exc:
        raise KeyError(
            f"Could not read fitted parameter {name}"
        ) from exc


# ============================================================================
# Truth configuration
# ============================================================================

def truth_configuration(args):
    mtot = (
        float(args.m1_msun)
        + float(args.m2_msun)
    )

    q_mass = (
        float(args.m2_msun)
        / float(args.m1_msun)
    )

    a_rel = a_from_period_days(
        args.truth_period,
        mtot,
    )

    xi_rel = (
        a_rel
        / float(args.rehat_au)
    )

    theta = float(args.theta)

    xi_para = (
        xi_rel
        * np.cos(theta)
    )

    xi_perp = (
        xi_rel
        * np.sin(theta)
    )

    omega = (
        2.0
        * np.pi
        / float(args.truth_period)
    )

    return {
        "M1_Msun":
            float(args.m1_msun),

        "M2_Msun":
            float(args.m2_msun),

        "Mtot_Msun":
            float(mtot),

        "q_mass":
            float(q_mass),

        "P_days":
            float(args.truth_period),

        "a_rel_AU":
            float(a_rel),

        "rEhat_AU":
            float(args.rehat_au),

        "xi_rel":
            float(xi_rel),

        "xi_para":
            float(xi_para),

        "xi_perp":
            float(xi_perp),

        "omega":
            float(omega),

        "phase":
            float(args.phase),

        "inclination":
            float(args.inclination),

        "theta":
            float(theta),

        "q_flux":
            float(args.q_flux),
    }


def build_event_params(
    args,
    truth,
):
    mag = float(
        args.source_mag
    )

    return {
        "t0":
            float(args.t0),

        "u0":
            float(args.u0),

        "tE":
            float(args.tE),

        # Explicitly no parallax.
        "piEN":
            0.0,

        "piEE":
            0.0,

        # Native pyLIMA circular double-source coordinates.
        "xi_para":
            truth["xi_para"],

        "xi_perp":
            truth["xi_perp"],

        "xi_angular_velocity":
            truth["omega"],

        "xi_phase":
            truth["phase"],

        "xi_inclination":
            truth["inclination"],

        "xi_mass_ratio":
            truth["q_mass"],

        "q_flux":
            truth["q_flux"],

        # roman_rubin magnitude keys.
        "W149":
            mag,

        "u":
            mag,

        "g":
            mag,

        "r":
            mag,

        "i":
            mag,

        "z":
            mag,

        "Y":
            mag,
    }


# ============================================================================
# Canonical Roman simulation
# ============================================================================

def simulate_dataset(
    truth_kind,
    seed,
    args,
    truth,
):
    if truth_kind not in {
        "h0",
        "h1",
    }:
        raise ValueError(
            truth_kind
        )

    sig = inspect.signature(
        frr.simulate_event_for_fit
    )

    required_api = [
        "truth_double_source",
        "add_photometric_noise",
    ]

    missing_api = [
        x
        for x in required_api
        if x not in sig.parameters
    ]

    if missing_api:
        raise RuntimeError(
            "Current roman_rubin checkout lacks "
            f"required API: {missing_api}"
        )

    event_params = (
        build_event_params(
            args,
            truth,
        )
    )

    # Extra guard for reproducibility.
    np.random.seed(
        int(seed)
    )

    truth_model, truth_params, decision = (
        frr.simulate_event_for_fit(
            i=int(seed),
            event_params=event_params,
            path_ephemerides=str(
                EPHEMERIDES
            ),
            model="PSPL",

            # CRITICAL:
            # use the complete Roman schedule.
            time_window=None,

            use_roman=True,
            use_rubin=False,

            truth_parallax=False,

            truth_double_source=(
                truth_kind == "h1"
            ),

            apply_detection_criteria=False,

            apply_photometric_filter=True,

            # Production-like noisy realization.
            add_photometric_noise=True,
        )
    )

    if truth_model is None:
        raise RuntimeError(
            "roman_rubin returned no model."
        )

    roman_telescopes = [
        tel
        for tel in truth_model.event.telescopes
        if (
            getattr(
                tel,
                "filter",
                None,
            )
            == "W149"
            or
            getattr(
                tel,
                "name",
                None,
            )
            in {
                "W149",
                "Roman",
            }
        )
    ]

    if len(
        roman_telescopes
    ) != 1:
        raise RuntimeError(
            "Expected exactly one Roman telescope; "
            f"found {[x.name for x in roman_telescopes]}"
        )

    tel = roman_telescopes[
        0
    ]

    t = np.asarray(
        tel.lightcurve[
            "time"
        ],
        dtype=float,
    )

    mag = np.asarray(
        tel.lightcurve[
            "mag"
        ],
        dtype=float,
    )

    err_mag = np.asarray(
        tel.lightcurve[
            "err_mag"
        ],
        dtype=float,
    )

    lc = np.column_stack(
        [
            t,
            mag,
            err_mag,
        ]
    )

    if len(lc) < 100:
        raise RuntimeError(
            "Too few Roman points."
        )

    if not np.all(
        np.isfinite(lc)
    ):
        raise RuntimeError(
            "Non-finite Roman photometry."
        )

    span_days = float(
        np.max(t)
        - np.min(t)
    )

    # Architectural guard:
    # a one-season crop was ~73 d / 8688 points.
    # Full mission must clearly span multiple seasons/years.
    full_schedule_ok = (
        len(lc) > 30000
        and
        span_days > 1000.0
    )

    if not full_schedule_ok:
        raise RuntimeError(
            "Roman dataset does not look like the full "
            "mission schedule: "
            f"N={len(lc)}, span_days={span_days}"
        )

    return {
        "truth_kind":
            truth_kind,

        "seed":
            int(seed),

        "decision":
            bool(decision),

        "truth_model":
            truth_model,

        "truth_params":
            truth_params,

        "roman_lc":
            lc,

        "n_points":
            int(len(lc)),

        "t_min":
            float(np.min(t)),

        "t_max":
            float(np.max(t)),

        "span_days":
            span_days,

        "median_cadence_minutes":
            float(
                np.median(
                    np.diff(t)
                )
                * 24.0
                * 60.0
            ),

        "median_err_mag":
            float(
                np.median(
                    err_mag
                )
            ),

        "data_sha256":
            array_hash(
                lc
            ),

        "ra":
            float(
                truth_model.event.ra
            ),

        "dec":
            float(
                truth_model.event.dec
            ),

        "truth_parallax_model":
            getattr(
                truth_model,
                "parallax_model",
                None,
            ),

        "truth_double_source_model":
            getattr(
                truth_model,
                "double_source_model",
                None,
            ),
    }


# ============================================================================
# Fresh fit Event from one fixed numerical dataset
# ============================================================================



# ============================================================================
# One native pyLIMA fit
# ============================================================================


def run_native_fit(
    *,
    dataset,
    hypothesis,
    initial,
    args,
    xi_bound,
):
    """
    Run one observational Roman fit through the canonical
    roman_rubin fit_lc.fit_rubin_roman API.

    binary_source defines the scientific hypothesis, starts and bounds.
    roman_rubin owns Event/model construction and fitting.

    No TRF fitter is instantiated directly in this repository.
    """
    if hypothesis not in {
        "pspl",
        "bspl",
    }:
        raise ValueError(
            hypothesis
        )

    use_double_source = (
        hypothesis
        == "bspl"
    )

    # --------------------------------------------------------------
    # Canonical physical parameter order expected from fit_lc.
    # --------------------------------------------------------------

    if use_double_source:
        order = [
            "t0",
            "u0",
            "tE",
            "xi_para",
            "xi_perp",
            "xi_angular_velocity",
            "xi_phase",
            "xi_inclination",
            "xi_mass_ratio",
            "q_flux_W149",
        ]
    else:
        order = [
            "t0",
            "u0",
            "tE",
        ]

    missing_initial = [
        p
        for p in order
        if p not in initial
    ]

    if missing_initial:
        raise KeyError(
            "Missing initial values: "
            f"{missing_initial}"
        )

    initial = {
        key:
            float(value)
        for key, value
        in initial.items()
    }

    # --------------------------------------------------------------
    # fit_lc needs t0/u0/tE as its base fit parameters.
    #
    # Deliberately construct these from THIS fit's chosen start,
    # rather than passing the simulation truth dictionary.
    # --------------------------------------------------------------

    fit_event_params = {
        "t0":
            initial["t0"],

        "u0":
            initial["u0"],

        "tE":
            initial["tE"],
    }

    fit_defaults = None
    fit_bounds = None

    if use_double_source:

        omega_min = (
            2.0
            * np.pi
            / float(
                args.p_max
            )
        )

        omega_max = (
            2.0
            * np.pi
            / float(
                args.p_min
            )
        )

        # These values initialize the extra pyLIMA circular
        # double-source parameters. They are NOT fixed.
        fit_defaults = {
            p:
                initial[p]
            for p in order[3:]
        }

        # Scientific bounds controlled by binary_source.
        # All other double-source parameters keep the native
        # pyLIMA bounds exposed through roman_rubin.
        fit_bounds = {
            "xi_para":
                [
                    -float(xi_bound),
                    float(xi_bound),
                ],

            "xi_perp":
                [
                    -float(xi_bound),
                    float(xi_bound),
                ],

            "xi_angular_velocity":
                [
                    float(omega_min),
                    float(omega_max),
                ],
        }

    # --------------------------------------------------------------
    # fit_rubin_roman always writes its fit-result .npy.
    #
    # For this LRT driver those files are intermediate products;
    # keep them isolated and temporary. The complete selected result
    # is returned below and written by this script to JSON.
    # --------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix=(
            f"binary_source_{hypothesis}_"
            f"seed{dataset['seed']}_"
        )
    ) as tmpdir:

        fit, event_fit, model_fit = (
            fit_lc.fit_rubin_roman(
                Source=int(
                    dataset[
                        "seed"
                    ]
                ),

                event_params=fit_event_params,

                path_save=Path(
                    tmpdir
                ),

                path_ephemerides=str(
                    EPHEMERIDES
                ),

                model="PSPL",
                algo="TRF",
                Origin=None,

                # Use the canonical wide-bound policy in fit_lc.
                # Binary-source-specific bounds above then override
                # xi and omega where requested.
                rango=0,

                wfirst_lc=(
                    dataset[
                        "roman_lc"
                    ].copy()
                ),

                lsst_u=[],
                lsst_g=[],
                lsst_r=[],
                lsst_i=[],
                lsst_z=[],
                lsst_y=[],

                fit_model="PSPL",

                fit_parallax=False,

                fit_double_source=(
                    use_double_source
                ),

                fit_defaults=fit_defaults,
                fit_bounds=fit_bounds,

                # We supply an explicit deterministic start.
                random_state=None,
                initial_guess=initial,

                optimizer_options=None,

                event_ra=float(
                    dataset[
                        "ra"
                    ]
                ),

                event_dec=float(
                    dataset[
                        "dec"
                    ]
                ),
            )
        )

    assert_native_trf(
        fit
    )

    # --------------------------------------------------------------
    # Architecture guards.
    # --------------------------------------------------------------

    parallax_model = getattr(
        model_fit,
        "parallax_model",
        [None],
    )

    if parallax_model[0] != "None":
        raise RuntimeError(
            "Parallax unexpectedly active."
        )

    ds_model = getattr(
        model_fit,
        "double_source_model",
        None,
    )

    if use_double_source:

        if (
            ds_model is None
            or ds_model[0] != "Circular"
        ):
            raise RuntimeError(
                "fit_double_source=True did not produce "
                "a Circular double-source model."
            )

    else:

        if (
            ds_model is not None
            and ds_model[0] != "None"
        ):
            raise RuntimeError(
                "Double source unexpectedly active in H0."
            )

    fit_results = fit.fit_results

    if (
        bool(
            fit_results.get(
                "fit_parallax",
                False,
            )
        )
        is not False
    ):
        raise RuntimeError(
            "fit_results says parallax is active."
        )

    if (
        bool(
            fit_results.get(
                "fit_double_source",
                False,
            )
        )
        != use_double_source
    ):
        raise RuntimeError(
            "fit_double_source metadata mismatch."
        )

    # --------------------------------------------------------------
    # Confirm that fit_lc saw exactly the numerical dataset produced
    # by simulate_dataset: same t, mag and err_mag.
    # --------------------------------------------------------------

    roman_telescopes = [
        tel
        for tel in event_fit.telescopes
        if (
            getattr(
                tel,
                "filter",
                None,
            )
            == "W149"
            or
            getattr(
                tel,
                "name",
                None,
            )
            in {
                "W149",
                "Roman",
            }
        )
    ]

    if len(roman_telescopes) != 1:
        raise RuntimeError(
            "Expected exactly one Roman telescope "
            f"in fit Event; found "
            f"{[x.name for x in roman_telescopes]}"
        )

    tel = roman_telescopes[0]

    fit_lc_array = np.column_stack(
        [
            np.asarray(
                tel.lightcurve[
                    "time"
                ],
                dtype=float,
            ),

            np.asarray(
                tel.lightcurve[
                    "mag"
                ],
                dtype=float,
            ),

            np.asarray(
                tel.lightcurve[
                    "err_mag"
                ],
                dtype=float,
            ),
        ]
    )

    if not np.array_equal(
        fit_lc_array,
        dataset[
            "roman_lc"
        ],
    ):
        raise RuntimeError(
            "fit_lc did not receive exactly the "
            "simulated Roman dataset."
        )

    fit_data_sha256 = array_hash(
        fit_lc_array
    )

    if (
        fit_data_sha256
        != dataset[
            "data_sha256"
        ]
    ):
        raise RuntimeError(
            "Roman dataset SHA256 changed before fitting."
        )

    # --------------------------------------------------------------
    # Retrieve the canonical fit result.
    # --------------------------------------------------------------

    if "best_model" not in fit_results:
        raise RuntimeError(
            "fit_rubin_roman returned no best_model."
        )

    if "chi2" not in fit_results:
        raise RuntimeError(
            "fit_rubin_roman returned no chi2."
        )

    best_model = np.asarray(
        fit_results[
            "best_model"
        ],
        dtype=float,
    ).reshape(-1)

    if not np.all(
        np.isfinite(
            best_model
        )
    ):
        raise RuntimeError(
            "Non-finite best_model."
        )

    chi2 = scalar(
        fit_results[
            "chi2"
        ]
    )

    if not np.isfinite(
        chi2
    ):
        raise RuntimeError(
            "Non-finite chi2."
        )

    actual_order = list(
        fit_results.get(
            "initial_guess_parameter_order",
            [],
        )
    )

    if actual_order != order:
        raise RuntimeError(
            "Unexpected fitted physical parameter order.\n"
            f"expected={order}\n"
            f"actual={actual_order}"
        )

    py_best = (
        model_fit.compute_pyLIMA_parameters(
            best_model
        )
    )

    best_physical = {
        p:
            get_parameter(
                py_best,
                p,
            )
        for p in order
    }

    print()
    print(
        "    fitter =",
        fitter_name(
            fit
        ),
    )

    print(
        "    hypothesis =",
        hypothesis,
    )

    print(
        "    parameter order =",
        order,
    )

    print(
        "    fit_double_source =",
        use_double_source,
    )

    if use_double_source:
        print(
            "    xi bounds =",
            fit_bounds[
                "xi_para"
            ],
        )

        print(
            "    omega bounds =",
            fit_bounds[
                "xi_angular_velocity"
            ],
        )

    result = {
        "hypothesis":
            hypothesis,

        "fitter":
            fitter_name(
                fit
            ),

        "fit_parallax":
            False,

        "fit_double_source":
            bool(
                use_double_source
            ),

        "chi2":
            float(
                chi2
            ),

        "reported_chi2":
            float(
                chi2
            ),

        "physical_parameter_order":
            list(
                order
            ),

        "best_physical":
            best_physical,

        "best_model":
            [
                float(x)
                for x
                in best_model
            ],

        "data_sha256":
            fit_data_sha256,
    }

    if use_double_source:

        omega = (
            best_physical[
                "xi_angular_velocity"
            ]
        )

        result[
            "best_period_days"
        ] = (
            float(
                2.0
                * np.pi
                / omega
            )
            if omega > 0.0
            else np.inf
        )

        result[
            "xi_bound"
        ] = float(
            xi_bound
        )

    return result



# ============================================================================
# H1 deterministic native-TRF multistart
# ============================================================================

def fit_bspl_multistart(
    *,
    dataset,
    h0_fit,
    args,
    xi_bound,
):
    shape = (
        h0_fit[
            "best_physical"
        ]
    )

    omega_start = (
        2.0
        * np.pi
        / float(
            args.fit_start_period
        )
    )

    q_flux_start = (
        max(
            float(
                args.fit_start_q_flux
            ),
            1.0e-4,
        )
    )

    # IMPORTANT:
    # This start policy is identical for datasets generated
    # under H0 and H1.  It does not inspect truth_kind.
    starts = [
        (
            "near_null",
            {
                "xi_para":
                    1.0e-4,

                "xi_perp":
                    0.0,
            },
        ),

        (
            "positive_xi",
            {
                "xi_para":
                    float(
                        args.fit_start_xi
                    ),

                "xi_perp":
                    0.0,
            },
        ),

        (
            "negative_xi",
            {
                "xi_para":
                    -float(
                        args.fit_start_xi
                    ),

                "xi_perp":
                    0.0,
            },
        ),
    ]

    results = []

    for (
        label,
        xistart,
    ) in starts:

        initial = {
            "t0":
                shape["t0"],

            "u0":
                shape["u0"],

            "tE":
                shape["tE"],

            "xi_para":
                xistart[
                    "xi_para"
                ],

            "xi_perp":
                xistart[
                    "xi_perp"
                ],

            "xi_angular_velocity":
                omega_start,

            # Interior values avoid starting exactly
            # on singular/boundary geometries.
            "xi_phase":
                0.25,

            "xi_inclination":
                1.0,

            "xi_mass_ratio":
                0.5,

            "q_flux_W149":
                q_flux_start,
        }

        print()
        print(
            f"  H1 start: {label}"
        )

        try:
            result = run_native_fit(
                dataset=dataset,
                hypothesis="bspl",
                initial=initial,
                args=args,
                xi_bound=xi_bound,
            )

            result[
                "start_label"
            ] = label

            result[
                "success"
            ] = True

            results.append(
                result
            )

        except Exception as exc:
            print(
                "    FAILED:",
                repr(
                    exc
                ),
            )

            results.append(
                {
                    "start_label":
                        label,

                    "success":
                        False,

                    "error":
                        repr(
                            exc
                        ),
                }
            )

    successful = [
        r
        for r in results
        if (
            r.get(
                "success",
                False,
            )
            and
            np.isfinite(
                r[
                    "chi2"
                ]
            )
        )
    ]

    if not successful:
        raise RuntimeError(
            "All native BSPL TRF starts failed."
        )

    best = min(
        successful,
        key=lambda r:
            r["chi2"],
    )

    return (
        best,
        results,
    )


# ============================================================================
# One LRT
# ============================================================================

def run_one_lrt(
    *,
    truth_kind,
    seed,
    args,
    truth,
    xi_bound,
):
    print()
    print("=" * 100)
    print(
        f"DATASET GENERATED UNDER {truth_kind.upper()}"
    )
    print("=" * 100)

    dataset = (
        simulate_dataset(
            truth_kind,
            seed,
            args,
            truth,
        )
    )

    print(
        "seed                  =",
        dataset[
            "seed"
        ],
    )

    print(
        "N Roman points        =",
        dataset[
            "n_points"
        ],
    )

    print(
        "time min/max          =",
        dataset[
            "t_min"
        ],
        dataset[
            "t_max"
        ],
    )

    print(
        "time span [d]         =",
        dataset[
            "span_days"
        ],
    )

    print(
        "median cadence [min]  =",
        dataset[
            "median_cadence_minutes"
        ],
    )

    print(
        "median err_mag        =",
        dataset[
            "median_err_mag"
        ],
    )

    print(
        "data sha256           =",
        dataset[
            "data_sha256"
        ],
    )

    print(
        "truth parallax        =",
        dataset[
            "truth_parallax_model"
        ],
    )

    print(
        "truth double source   =",
        dataset[
            "truth_double_source_model"
        ],
    )

    # ------------------------------------------------------------------
    # H0 fit
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("FIT H0 = PSPL / NO PARALLAX")
    print("-" * 100)

    h0_initial = {
        "t0":
            float(
                args.t0
            ),

        "u0":
            float(
                args.u0
            ),

        "tE":
            float(
                args.tE
            ),
    }

    h0_fit = (
        run_native_fit(
            dataset=dataset,
            hypothesis="pspl",
            initial=h0_initial,
            args=args,
            xi_bound=xi_bound,
        )
    )

    print(
        "H0 chi2 =",
        h0_fit[
            "chi2"
        ],
    )

    # ------------------------------------------------------------------
    # H1 fit
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print(
        "FIT H1 = PSPL + CIRCULAR DOUBLE SOURCE / NO PARALLAX"
    )
    print("-" * 100)

    h1_fit, h1_starts = (
        fit_bspl_multistart(
            dataset=dataset,
            h0_fit=h0_fit,
            args=args,
            xi_bound=xi_bound,
        )
    )

    print()
    print(
        "best H1 start =",
        h1_fit[
            "start_label"
        ],
    )

    print(
        "H1 chi2       =",
        h1_fit[
            "chi2"
        ],
    )

    print(
        "best P [d]    =",
        h1_fit[
            "best_period_days"
        ],
    )

    # ------------------------------------------------------------------
    # LRT
    # ------------------------------------------------------------------

    delta_raw = (
        float(
            h0_fit[
                "chi2"
            ]
        )
        -
        float(
            h1_fit[
                "chi2"
            ]
        )
    )

    T = max(
        0.0,
        delta_raw,
    )

    tolerance = (
        1.0e-5
        * max(
            1.0,
            abs(
                h0_fit[
                    "chi2"
                ]
            ),
        )
    )

    nesting_ok = (
        h1_fit[
            "chi2"
        ]
        <=
        h0_fit[
            "chi2"
        ]
        + tolerance
    )

    backend_ok = (
        h0_fit[
            "fitter"
        ]
        ==
        "pyLIMA.fits.TRF_fit.TRFfit"
        and
        h1_fit[
            "fitter"
        ]
        ==
        "pyLIMA.fits.TRF_fit.TRFfit"
    )

    same_dataset_ok = (
        h0_fit[
            "data_sha256"
        ]
        ==
        h1_fit[
            "data_sha256"
        ]
        ==
        dataset[
            "data_sha256"
        ]
    )

    checks = {
        "native_TRFfit":
            bool(
                backend_ok
            ),

        "same_dataset_H0_H1":
            bool(
                same_dataset_ok
            ),

        "H1_not_worse_than_H0":
            bool(
                nesting_ok
            ),

        "full_Roman_schedule":
            bool(
                dataset[
                    "n_points"
                ]
                > 30000
                and
                dataset[
                    "span_days"
                ]
                > 1000.0
            ),
    }

    print()
    print("-" * 100)
    print("LRT RESULT")
    print("-" * 100)

    print(
        "chi2_H0       =",
        h0_fit[
            "chi2"
        ],
    )

    print(
        "chi2_H1       =",
        h1_fit[
            "chi2"
        ],
    )

    print(
        "Delta chi2 raw =",
        delta_raw,
    )

    print(
        "T=max(0,Dchi2) =",
        T,
    )

    print()
    print("CHECKS")

    for (
        key,
        value,
    ) in checks.items():
        print(
            f"  {key:32s}",
            (
                "PASS"
                if value
                else "FAIL"
            ),
        )

    if not all(
        checks.values()
    ):
        raise RuntimeError(
            "One or more LRT validation checks failed."
        )

    return {
        "truth_kind":
            truth_kind,

        "seed":
            int(
                seed
            ),

        "dataset": {
            key:
                dataset[
                    key
                ]
            for key in [
                "n_points",
                "t_min",
                "t_max",
                "span_days",
                "median_cadence_minutes",
                "median_err_mag",
                "data_sha256",
                "truth_parallax_model",
                "truth_double_source_model",
            ]
        },

        "h0":
            h0_fit,

        "h1":
            h1_fit,

        "h1_all_starts":
            h1_starts,

        "delta_chi2_raw":
            float(
                delta_raw
            ),

        "T":
            float(
                T
            ),

        "checks":
            checks,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--truth",
        choices=[
            "h0",
            "h1",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261020,
        help=(
            "H0 uses this seed. "
            "With --truth both, H1 uses seed+1."
        ),
    )

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
    )

    parser.add_argument(
        "--t0",
        type=float,
        default=2461849.0,
    )

    parser.add_argument(
        "--u0",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--truth-period",
        type=float,
        default=365.25,
    )

    parser.add_argument(
        "--m1-msun",
        type=float,
        default=M1_MSUN_DEFAULT,
    )

    parser.add_argument(
        "--m2-msun",
        type=float,
        default=M2_MSUN_DEFAULT,
    )

    parser.add_argument(
        "--rehat-au",
        type=float,
        default=REHAT_AU_DEFAULT,
    )

    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_TRUE_DEFAULT,
    )

    parser.add_argument(
        "--phase",
        type=float,
        default=PHASE_TRUE_DEFAULT,
    )

    parser.add_argument(
        "--inclination",
        type=float,
        default=INCLINATION_TRUE_DEFAULT,
    )

    parser.add_argument(
        "--q-flux",
        type=float,
        default=Q_FLUX_TRUE_DEFAULT,
    )

    # Preserve the period domain of the old experiment,
    # but now as bounds on native pyLIMA omega.
    parser.add_argument(
        "--p-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--p-max",
        type=float,
        default=100000.0,
    )

    # IMPORTANT:
    # fit-start values are independent from truth_period.
    parser.add_argument(
        "--fit-start-period",
        type=float,
        default=365.25,
    )

    parser.add_argument(
        "--fit-start-xi",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--fit-start-q-flux",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--output",
        default=(
            "/tmp/"
            "roman_native_lrt_no_parallax.json"
        ),
    )

    args = parser.parse_args()

    if not (
        args.p_min
        > 0.0
        and
        args.p_max
        > args.p_min
    ):
        raise ValueError(
            "Require 0 < p_min < p_max."
        )

    if not (
        args.p_min
        <= args.fit_start_period
        <= args.p_max
    ):
        raise ValueError(
            "fit_start_period outside fit period domain."
        )

    truth = (
        truth_configuration(
            args
        )
    )

    mtot = (
        args.m1_msun
        + args.m2_msun
    )

    # Bound wide enough to contain every Kepler amplitude
    # in the old P=[p_min,p_max] truth domain, with margin.
    xi_max_physical = (
        xi_from_period_days(
            args.p_max,
            mtot,
            args.rehat_au,
        )
    )

    xi_bound = (
        1.25
        * xi_max_physical
    )

    if (
        abs(
            args.fit_start_xi
        )
        >= xi_bound
    ):
        raise ValueError(
            "fit_start_xi lies outside derived xi bounds."
        )

    print("=" * 100)
    print("NATIVE PYLIMA ROMAN LRT")
    print("=" * 100)

    print(
        "roman_rubin repo =",
        ROMAN_RUBIN_REPO,
    )

    print(
        "functions backend =",
        Path(
            inspect.getfile(
                frr
            )
        ).resolve(),
    )

    print(
        "model backend     =",
        Path(
            inspect.getfile(
                sm
            )
        ).resolve(),
    )

    print(
        "fit helper        =",
        Path(
            inspect.getfile(
                fit_lc
            )
        ).resolve(),
    )

    print(
        "fit backend       =",
        "fit_lc.fit_rubin_roman -> pyLIMA.fits.TRF_fit.TRFfit",
    )

    print(
        "ephemerides      =",
        EPHEMERIDES,
    )

    print()
    print(
        "truth xi_rel     =",
        truth[
            "xi_rel"
        ],
    )

    print(
        "truth omega      =",
        truth[
            "omega"
        ],
    )

    print(
        "xi fit bound     = +/-",
        xi_bound,
    )

    print(
        "omega fit domain =",
        (
            2.0
            * np.pi
            / args.p_max
        ),
        (
            2.0
            * np.pi
            / args.p_min
        ),
    )

    print(
        "time_window      = None "
        "(full Roman schedule)"
    )

    print(
        "photometric noise= ON"
    )

    jobs = []

    if args.truth in {
        "h0",
        "both",
    }:
        jobs.append(
            (
                "h0",
                int(
                    args.seed
                ),
            )
        )

    if args.truth in {
        "h1",
        "both",
    }:
        h1_seed = (
            int(
                args.seed
            )
            if args.truth
            == "h1"
            else
            int(
                args.seed
            )
            + 1
        )

        jobs.append(
            (
                "h1",
                h1_seed,
            )
        )

    results = {}

    for (
        truth_kind,
        seed,
    ) in jobs:

        results[
            truth_kind
        ] = (
            run_one_lrt(
                truth_kind=truth_kind,
                seed=seed,
                args=args,
                truth=truth,
                xi_bound=xi_bound,
            )
        )

    summary = {
        "configuration": {
            "source_mag":
                float(
                    args.source_mag
                ),

            "t0":
                float(
                    args.t0
                ),

            "u0":
                float(
                    args.u0
                ),

            "tE":
                float(
                    args.tE
                ),

            "truth":
                truth,

            "p_min":
                float(
                    args.p_min
                ),

            "p_max":
                float(
                    args.p_max
                ),

            "xi_bound":
                float(
                    xi_bound
                ),

            "fit_start_period":
                float(
                    args.fit_start_period
                ),

            "fit_start_xi":
                float(
                    args.fit_start_xi
                ),

            "time_window":
                None,

            "add_photometric_noise":
                True,

            "truth_parallax":
                False,

            "fit_parallax":
                False,

            "H0":
                "PSPL",

            "H1":
                (
                    "PSPL + native pyLIMA "
                    "Circular double source"
                ),

            "fitter":
                (
                    "pyLIMA.fits."
                    "TRF_fit.TRFfit"
                ),
        },

        "results":
            results,
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for (
        truth_kind,
        row,
    ) in results.items():

        print(
            f"{truth_kind.upper():3s} truth: "
            f"N={row['dataset']['n_points']}  "
            f"chi2_H0={row['h0']['chi2']:.8g}  "
            f"chi2_H1={row['h1']['chi2']:.8g}  "
            f"T={row['T']:.8g}  "
            f"Pbest={row['h1']['best_period_days']:.8g} d"
        )

    print()
    print(
        "summary =",
        output,
    )

    print()
    print("=" * 100)
    print(
        "NATIVE H0/H1 ONE-REALIZATION RUN PASSED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
