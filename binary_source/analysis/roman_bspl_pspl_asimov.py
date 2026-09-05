#!/usr/bin/env python3

import argparse
import subprocess
import sys
import traceback
import re
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.time import Time

from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.simulations import simulator
from pyLIMA.fits import TRF_fit


# ============================================================
# Paths del proyecto
# ============================================================

THIS_FILE = Path(__file__).resolve()
BINARY_SOURCE_DIR = THIS_FILE.parents[1]
REPO_ROOT = BINARY_SOURCE_DIR.parent

if str(BINARY_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(BINARY_SOURCE_DIR))

from functions_aux import (
    sigma_F146_func,
    a_from_P_kepler_days,
)


# ============================================================
# Default physical configuration
#
# Must match the main intrinsic experiment.
# ============================================================

M1_MSUN = 2.0
M2_MSUN = 1.0

Q_MASS = M2_MSUN / M1_MSUN
MTOT_MSUN = M1_MSUN + M2_MSUN

REHAT_AU = 5.0

TE_TRUE = 150.0

PHI_TRUE = 0.0
INCLINATION_TRUE = np.pi / 2.0
THETA_TRUE = 0.0

QFLUX_TRUE = 0.0

FIT_WINDOW_TE = 3.5


# ============================================================
# Roman seasons
#
# Same prescription used by the current Roman machinery.
# sampling is expressed in hours in simulate_a_telescope.
# ============================================================

NOMINAL_SEASONS = [
    ("2027-02-11T00:00:00", "2027-04-24T00:00:00"),
    ("2027-08-16T00:00:00", "2027-10-27T00:00:00"),
    ("2028-02-11T00:00:00", "2028-04-24T00:00:00"),
    ("2030-02-11T00:00:00", "2030-04-24T00:00:00"),
    ("2030-08-16T00:00:00", "2030-10-27T00:00:00"),
    ("2031-02-11T00:00:00", "2031-04-24T00:00:00"),
]

OFF_SEASONS = [
    ("2028-08-15T00:00:00", "2028-10-27T00:00:00"),
    ("2029-02-11T00:00:00", "2029-04-24T00:00:00"),
    ("2029-08-16T00:00:00", "2029-10-27T00:00:00"),
]

# Same as in the current Roman code
ROMAN_NOMINAL_SAMPLING_HOURS = 121.0 / 600.0
ROMAN_OFF_SAMPLING_HOURS = 24.0 * 3.0

# Current GBTDS F146 prescription:
# approximately 8390 high-cadence epochs per season.
ROMAN_NOMINAL_EPOCHS = 8390


# ============================================================
# Helpers
# ============================================================

def as_array(x):
    if hasattr(x, "value"):
        return np.asarray(x.value, dtype=float)

    return np.asarray(x, dtype=float)


def scalar(x):
    arr = np.asarray(x)

    if arr.size != 1:
        raise ValueError(
            f"Expected a scalar but received shape={arr.shape}"
        )

    return float(arr.reshape(-1)[0])


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        dirty = bool(status.strip())

    except Exception:
        commit = "unknown"
        dirty = True

    return commit, dirty


# ============================================================
# Roman sampling
# ============================================================

def simulate_roman_season(
    start,
    end,
    sampling_hours,
    max_epochs=None,
):
    """
    Return only the observing times for one Roman season.

    We use pyLIMA's simulate_a_telescope to reproduce
    the same time-sampling machinery as the Roman pipeline.
    """

    tstart = Time(
        start,
        format="isot",
    ).jd

    tend = Time(
        end,
        format="isot",
    ).jd

    roman = simulator.simulate_a_telescope(
        name="F146",
        time_start=tstart,
        time_end=tend,
        sampling=sampling_hours,
        location="Space",
        camera_filter="F146",
        uniform_sampling=True,
        astrometry=False,
    )

    times = as_array(
        roman.lightcurve["time"]
    )

    # The old visibility-window prescription generates slightly
    # more F146 measurements than the current GBTDS allocation.
    # For high-cadence seasons, retain the central max_epochs
    # observations so that the season remains centered on the
    # same visibility window.
    if (
        max_epochs is not None
        and len(times) > max_epochs
    ):
        excess = len(times) - max_epochs
        i0 = excess // 2
        times = times[
            i0:i0 + max_epochs
        ]

    return times


def build_roman_times(
    tE_true,
    anchor_season_index=2,
    fit_window_te=3.5,
    include_off_seasons=True,
):
    """
    Build the complete Roman cadence and place t0 at the center
    of a nominal observing season.

    No trasladamos los JDs: mantenemos tiempos absolutos.
    This will make it easier to incorporate parallax later.

    Parameters
    ----------
    anchor_season_index : int
        Nominal season whose midpoint defines t0.

        Default = 2:
        2028-02-11 -- 2028-04-24

    fit_window_te : float
        Keep observations within
        |t-t0| <= fit_window_te * tE.
    """

    if not (
        0 <= anchor_season_index < len(NOMINAL_SEASONS)
    ):
        raise ValueError(
            "anchor_season_index fuera de rango."
        )

    times = []

    for start, end in NOMINAL_SEASONS:

        tt = simulate_roman_season(
            start=start,
            end=end,
            sampling_hours=ROMAN_NOMINAL_SAMPLING_HOURS,
            max_epochs=ROMAN_NOMINAL_EPOCHS,
        )

        times.append(tt)

    if include_off_seasons:

        for start, end in OFF_SEASONS:

            tt = simulate_roman_season(
                start=start,
                end=end,
                sampling_hours=ROMAN_OFF_SAMPLING_HOURS,
            )

            times.append(tt)

    t_all = np.concatenate(times)

    t_all = np.unique(
        np.sort(t_all)
    )

    anchor_start, anchor_end = (
        NOMINAL_SEASONS[
            anchor_season_index
        ]
    )

    anchor_start_jd = Time(
        anchor_start,
        format="isot",
    ).jd

    anchor_end_jd = Time(
        anchor_end,
        format="isot",
    ).jd

    t0_true = (
        0.5
        * (
            anchor_start_jd
            + anchor_end_jd
        )
    )

    mask = (
        np.abs(
            t_all - t0_true
        )
        <= fit_window_te * tE_true
    )

    t = t_all[mask]

    if len(t) == 0:
        raise RuntimeError(
            "The selected window contains no "
            "Roman observations."
        )

    return t, float(t0_true)


# ============================================================
# F146 photometric precision
# ============================================================

def sigma_f146_safe(magnitude):
    """
    Single-epoch F146 precision used by the Roman forecast.

    The underlying sigma_F146_func is normalized to the
    current GBTDS requirement S/N~100 at F146_AB=21.2.

    We restrict the faint end to F146<=27 to avoid using the
    simplified prescription arbitrarily far into the
    noise-dominated regime.
    """

    magnitude = np.asarray(
        magnitude,
        dtype=float,
    )

    if np.any(
        ~np.isfinite(magnitude)
    ):
        raise ValueError(
            "Non-finite F146 magnitudes."
        )

    if np.any(
        magnitude > 27.0
    ):
        raise ValueError(
            "The light curve contains F146 > 27. "
            "The adopted F146 precision model is not "
            "used beyond this range."
        )

    sigma = sigma_F146_func(
        magnitude
    )

    return np.asarray(
        sigma,
        dtype=float,
    )


# ============================================================
# Generic pyLIMA event
# ============================================================

def build_event_from_magnitudes(
    t,
    magnitude,
    err_mag,
    name="Roman",
):
    """
    Build the Event used for the fit.

    Parallax is disabled in this first experiment.
    Therefore, the physical Telescope location does not enter
    the trajectory. We use location='Space' to preserve
    the Roman semantics.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    magnitude = np.asarray(
        magnitude,
        dtype=float,
    )

    err_mag = np.asarray(
        err_mag,
        dtype=float,
    )

    if not (
        len(t)
        == len(magnitude)
        == len(err_mag)
    ):
        raise ValueError(
            "t, magnitude y err_mag "
            "must have the same length."
        )

    ev = event.Event()

    ev.name = (
        "Roman_BSPL_Asimov"
    )

    ev.ra = 267.92497054815516
    ev.dec = -29.152232510353276

    lc = np.column_stack(
        [
            t,
            magnitude,
            err_mag,
        ]
    ).astype(float)

    tel = telescopes.Telescope(
        name=name,
        camera_filter="F146",
        lightcurve=lc,
        lightcurve_names=[
            "time",
            "mag",
            "err_mag",
        ],
        lightcurve_units=[
            "JD",
            "mag",
            "mag",
        ],
        location="Space",
    )

    ev.telescopes.append(
        tel
    )

    return ev, tel


# ============================================================
# Event used only to compute A_BSPL
# ============================================================

def build_truth_event(
    t,
):
    """
    Auxiliary event used to compute the magnification.

    The photometric values here are placeholders.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    mag0 = np.full_like(
        t,
        20.0,
    )

    emag = np.full_like(
        t,
        0.01,
    )

    return build_event_from_magnitudes(
        t=t,
        magnitude=mag0,
        err_mag=emag,
        name="Roman_truth",
    )


# ============================================================
# BSPL truth
# ============================================================

def bspl_truth_magnification(
    t,
    t0_true,
    u0_true,
    tE_true,
    P_days,
    q_mass=Q_MASS,
    qflux=QFLUX_TRUE,
    Mtot_Msun=MTOT_MSUN,
    rEhat_AU=REHAT_AU,
    theta=THETA_TRUE,
    phi=PHI_TRUE,
    inclination=INCLINATION_TRUE,
):
    """
    Same physical parameterization used in the intrinsic scans.

    xi_rel = a_rel / rEhat

    and pyLIMA distributes the orbital motion between the two sources
    mediante q_mass.
    """

    P_days = float(P_days)

    if P_days <= 0.0:
        raise ValueError(
            "P_days must be positive."
        )

    a_rel_AU = float(
        a_from_P_kepler_days(
            P_days,
            Mtot_Msun,
        )
    )

    xi_rel = (
        a_rel_AU
        / float(rEhat_AU)
    )

    omega = (
        2.0
        * np.pi
        / P_days
    )

    xi_para = (
        xi_rel
        * np.cos(theta)
    )

    xi_perp = (
        xi_rel
        * np.sin(theta)
    )

    ev_truth, tel_truth = (
        build_truth_event(t)
    )

    model_bspl = (
        PSPL_model.PSPLmodel(
            ev_truth,
            parallax=[
                "None",
                t0_true,
            ],
            double_source=[
                "Circular",
                t0_true,
            ],
        )
    )

    model_bspl.define_model_parameters()

    # Only required to construct pyLIMA_parameters.
    # No intervienen en model_magnification.
    fsource_dummy = 1.0
    ftotal_dummy = 1.0

    params_bspl = [
        t0_true,
        u0_true,
        tE_true,

        xi_para,
        xi_perp,
        omega,

        phi,
        inclination,

        q_mass,
        qflux,

        fsource_dummy,
        ftotal_dummy,
    ]

    py_params_bspl = (
        model_bspl.compute_pyLIMA_parameters(
            params_bspl
        )
    )

    A_bspl = (
        model_bspl.model_magnification(
            tel_truth,
            py_params_bspl,
        )
    )

    A_bspl = as_array(
        A_bspl
    )

    # pyLIMA returns A1 + qf A2.
    # We want the magnification relative to the total unlensed flux
    # total no magnificado.
    A_bspl = (
        A_bspl
        / (1.0 + qflux)
    )

    if np.any(
        ~np.isfinite(A_bspl)
    ):
        raise RuntimeError(
            "A_BSPL contiene valores no finitos."
        )

    if np.any(
        A_bspl <= 0.0
    ):
        raise RuntimeError(
            "A_BSPL contiene valores <= 0."
        )

    return {
        "A_bspl": A_bspl,
        "xi_rel": xi_rel,
        "a_rel_AU": a_rel_AU,
        "omega": omega,
    }


# ============================================================
# Asimov dataset construction
# ============================================================

def make_roman_asimov_event(
    t,
    A_bspl,
    source_mag,
):
    """
    Observed data = expected value under the BSPL model.

    No random noise realization is added.

    m(t) = m_base - 2.5 log10(A_BSPL).
    """

    source_mag = float(
        source_mag
    )

    A_bspl = np.asarray(
        A_bspl,
        dtype=float,
    )

    mag_truth = (
        source_mag
        - 2.5
        * np.log10(A_bspl)
    )

    err_mag = (
        sigma_f146_safe(
            mag_truth
        )
    )

    ev, tel = (
        build_event_from_magnitudes(
            t=t,
            magnitude=mag_truth,
            err_mag=err_mag,
            name="Roman",
        )
    )

    return {
        "event": ev,
        "telescope": tel,
        "mag_truth": mag_truth,
        "err_mag": err_mag,
        "n_bright_floor": int(
            np.count_nonzero(
                mag_truth < 12.0
            )
        ),
    }


# ============================================================
# PSPL fit con pyLIMA
# ============================================================

def fit_pspl_roman(
    ev,
    t0_guess,
    u0_guess,
    tE_guess,
):
    """
    Fit a standard PSPL model with TRF.

    The free physical parameters are:
        t0, u0, tE

    pyLIMA simultaneously determines the parameters
    photometric parameters associated with the telescope.

    This is deliberate: we want the best PSPL model that
    would actually be obtained from the Roman data.
    """

    model_pspl = (
        PSPL_model.PSPLmodel(
            ev,
            blend_flux_parameter="ftotal",
        )
    )

    model_pspl.define_model_parameters()

    fit = TRF_fit.TRFfit(
        model_pspl
    )

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    if "t0" in fit.fit_parameters:

        fit.fit_parameters[
            "t0"
        ][1] = [
            t0_guess
            - 2.0 * tE_guess,

            t0_guess
            + 2.0 * tE_guess,
        ]

    if "u0" in fit.fit_parameters:

        u_bound = max(
            20.0,
            2.5 * abs(u0_guess),
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
            0.02 * tE_guess,
            20.0 * tE_guess,
        ]

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    fit.model_parameters_guess = [
        float(t0_guess),
        float(u0_guess),
        float(tE_guess),
    ]

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fit.fit()

    results = fit.fit_results

    if "best_model" not in results:
        raise RuntimeError(
            "pyLIMA did not return best_model. "
            f"Keys: {list(results.keys())}"
        )

    best_model = np.asarray(
        results["best_model"],
        dtype=float,
    ).reshape(-1)

    if len(best_model) < 3:
        raise RuntimeError(
            "best_model has fewer than three parameters."
        )

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
    # chi2 reportado por pyLIMA
    # --------------------------------------------------------

    chi2_reported = np.nan

    if "chi2" in results:
        try:
            chi2_reported = scalar(
                results["chi2"]
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Recompute chi2 explicitly as a validation check
    # --------------------------------------------------------

    chi2_recomputed = np.nan

    try:

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
            )["photometry"]
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
            np.isfinite(F_data)
            & np.isfinite(F_model)
            & np.isfinite(sigma_F)
            & (sigma_F > 0.0)
        )

        chi2_recomputed = float(
            np.sum(
                (
                    (
                        F_data[valid]
                        - F_model[valid]
                    )
                    / sigma_F[valid]
                ) ** 2
            )
        )

    except Exception:
        # Do not stop the analysis:
        # fit_results["chi2"] sigue siendo la referencia.
        pass

    if np.isfinite(
        chi2_recomputed
    ):
        chi2_use = (
            chi2_recomputed
        )

    elif np.isfinite(
        chi2_reported
    ):
        chi2_use = (
            chi2_reported
        )

    else:
        raise RuntimeError(
            "Could not obtain chi2 from the fit."
        )

    return {
        "fit": fit,
        "model": model_pspl,
        "best_model": best_model,
        "best_t0": best_t0,
        "best_u0": best_u0,
        "best_tE": best_tE,
        "chi2": chi2_use,
        "chi2_reported": chi2_reported,
        "chi2_recomputed": chi2_recomputed,
    }


# ============================================================
# Event S/N
# ============================================================

def event_snr(
    telescope,
    A_truth,
):
    """
    SNR of the microlensing flux excess relative to baseline:

        SNR_event^2 =
            sum [(F_i - F_base)/sigma_i]^2.

    This is useful for comparison with:

        sqrt(Delta chi2) ~ D * SNR_event.
    """

    F = as_array(
        telescope.lightcurve[
            "flux"
        ]
    )

    sigma_F = as_array(
        telescope.lightcurve[
            "err_flux"
        ]
    )

    A_truth = np.asarray(
        A_truth,
        dtype=float,
    )

    valid = (
        np.isfinite(F)
        & np.isfinite(sigma_F)
        & np.isfinite(A_truth)
        & (sigma_F > 0.0)
        & (A_truth > 0.0)
    )

    if not np.any(valid):
        return np.nan

    Fbase_values = (
        F[valid]
        / A_truth[valid]
    )

    Fbase = float(
        np.nanmedian(
            Fbase_values
        )
    )

    snr2 = np.sum(
        (
            (
                F[valid]
                - Fbase
            )
            / sigma_F[valid]
        ) ** 2
    )

    return float(
        np.sqrt(
            max(
                snr2,
                0.0,
            )
        )
    )


# ============================================================
# One case
# ============================================================

def run_case(
    t,
    t0_true,
    u0_true,
    tE_true,
    P_days,
    source_mag,
    q_mass=Q_MASS,
    qflux=QFLUX_TRUE,
):
    """
    Ejecuta:

        BSPL truth
        -> Roman Asimov photometry
        -> weighted PSPL fit
        -> Delta chi2
    """

    truth = (
        bspl_truth_magnification(
            t=t,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
            P_days=P_days,
            q_mass=q_mass,
            qflux=qflux,
        )
    )

    A_bspl = truth[
        "A_bspl"
    ]

    asimov = (
        make_roman_asimov_event(
            t=t,
            A_bspl=A_bspl,
            source_mag=source_mag,
        )
    )

    fit_result = (
        fit_pspl_roman(
            ev=asimov["event"],
            t0_guess=t0_true,
            u0_guess=u0_true,
            tE_guess=tE_true,
        )
    )

    # Asimov:
    #
    # chi2_BSPL = 0
    #
    # Delta chi2 =
    # chi2_PSPL - chi2_BSPL
    #
    delta_chi2 = float(
        fit_result["chi2"]
    )

    snr = event_snr(
        telescope=asimov[
            "telescope"
        ],
        A_truth=A_bspl,
    )

    # ========================================================
    # Effective Roman-weighted fractional mismatch
    #
    # In the idealized uniform-weight limit:
    #
    #     sqrt(Delta chi2)
    #         ~ D * SNR_event
    #
    # therefore
    #
    #     D_Roman_eff =
    #         sqrt(Delta chi2_Roman) / SNR_event
    #
    # This is NOT the intrinsic D used in the main paper.
    # It is a useful Roman-weighted diagnostic that can be
    # compared with the intrinsic metric.
    # ========================================================

    if (
        np.isfinite(snr)
        and snr > 0.0
    ):

        D_roman_eff = (
            np.sqrt(
                max(
                    delta_chi2,
                    0.0,
                )
            )
            / snr
        )

    else:

        D_roman_eff = np.nan

    return {
        "success": True,

        "source_mag": float(
            source_mag
        ),

        "u0_true": float(
            u0_true
        ),

        "P_days": float(
            P_days
        ),

        "P_over_tE": float(
            P_days
            / tE_true
        ),

        "xi_rel": float(
            truth["xi_rel"]
        ),

        "a_rel_AU": float(
            truth["a_rel_AU"]
        ),

        "delta_chi2": (
            delta_chi2
        ),

        "sqrt_delta_chi2": float(
            np.sqrt(
                max(
                    delta_chi2,
                    0.0,
                )
            )
        ),

        "snr_event": float(
            snr
        ),

        "D_roman_eff": float(
            D_roman_eff
        ),

        "best_t0": float(
            fit_result["best_t0"]
        ),

        "best_u0": float(
            fit_result["best_u0"]
        ),

        "best_tE": float(
            fit_result["best_tE"]
        ),

        "dt0_over_tE": float(
            (
                fit_result["best_t0"]
                - t0_true
            )
            / tE_true
        ),

        "du0_over_u0": (
            float(
                (
                    fit_result["best_u0"]
                    - u0_true
                )
                / u0_true
            )
            if u0_true != 0.0
            else np.nan
        ),

        "dtE_over_tE": float(
            (
                fit_result["best_tE"]
                - tE_true
            )
            / tE_true
        ),

        "chi2_reported": float(
            fit_result[
                "chi2_reported"
            ]
        ),

        "chi2_recomputed": float(
            fit_result[
                "chi2_recomputed"
            ]
        ),

        "n_obs": int(
            len(t)
        ),

        "n_bright_floor": int(
            asimov[
                "n_bright_floor"
            ]
        ),

        "min_mag": float(
            np.min(
                asimov[
                    "mag_truth"
                ]
            )
        ),

        "max_mag": float(
            np.max(
                asimov[
                    "mag_truth"
                ]
            )
        ),
    }


# ============================================================
# Smoke test
# ============================================================

def run_smoke_test(
    t,
    t0_true,
    tE_true,
    source_mag,
):
    print()
    print("=" * 80)
    print("ROMAN BSPL -> PSPL ASIMOV SMOKE TEST")
    print("=" * 80)

    print(
        f"N Roman observations = {len(t)}"
    )

    print(
        f"t0 = {t0_true:.6f} JD"
    )

    print(
        f"tE = {tE_true:.3f} d"
    )

    print(
        f"F146 baseline = {source_mag:.2f}"
    )

    print()

    u0_true = 0.1

    P_over_tE_test = [
        0.3,
        1.0,
        3.0,
    ]

    rows = []

    for ratio in P_over_tE_test:

        P_days = (
            ratio
            * tE_true
        )

        try:

            result = run_case(
                t=t,
                t0_true=t0_true,
                u0_true=u0_true,
                tE_true=tE_true,
                P_days=P_days,
                source_mag=source_mag,
            )

            rows.append(
                result
            )

            print(
                f"P/tE={ratio:6.3f}  "
                f"xi_rel={result['xi_rel']:.6e}  "
                f"DeltaChi2={result['delta_chi2']:.6e}  "
                f"sqrt={result['sqrt_delta_chi2']:.6e}  "
                f"SNR={result['snr_event']:.6e}  "
                f"D_Roman_eff={result['D_roman_eff']:.6e}"
            )

            print(
                "    best PSPL = "
                f"[{result['best_t0']:.8f}, "
                f"{result['best_u0']:.8g}, "
                f"{result['best_tE']:.8g}]"
            )

            print(
                "    biases = "
                f"dt0/tE={result['dt0_over_tE']:+.3e}, "
                f"du0/u0={result['du0_over_u0']:+.3e}, "
                f"dtE/tE={result['dtE_over_tE']:+.3e}"
            )

            if (
                np.isfinite(
                    result["chi2_reported"]
                )
                and np.isfinite(
                    result["chi2_recomputed"]
                )
            ):

                denom = max(
                    1.0,
                    abs(
                        result[
                            "chi2_reported"
                        ]
                    ),
                )

                relative_difference = (
                    abs(
                        result[
                            "chi2_recomputed"
                        ]
                        - result[
                            "chi2_reported"
                        ]
                    )
                    / denom
                )

                print(
                    "    chi2 validation: "
                    f"reported={result['chi2_reported']:.6e}, "
                    f"recomputed={result['chi2_recomputed']:.6e}, "
                    f"reldiff={relative_difference:.3e}"
                )

        except Exception as exc:

            print(
                f"P/tE={ratio:g} FAILED:"
            )

            print(
                f"    {type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

    print("=" * 80)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Intrinsic u0 x P grid
# ============================================================

def load_intrinsic_u0_period_grid(
    directory,
    u0_max=None,
):
    """
    Reconstruct the intrinsic u0 x P grid used in the
    BSPL -> PSPL analysis.

    Expected files:
        scan_kepler_u0_000.npz
        scan_kepler_u0_001.npz
        ...

    Returns the exact u0 and P nodes together with the
    intrinsic D_BSPL-PSPL map.

    If u0_max is supplied, rows above that value are removed
    without interpolating or regenerating the grid.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Intrinsic grid directory not found: {directory}"
        )

    files = list(
        directory.glob(
            "scan_kepler_u0_*.npz"
        )
    )

    if not files:
        raise RuntimeError(
            f"No scan_kepler_u0_*.npz files in {directory}"
        )

    def file_index(fn):

        match = re.search(
            r"scan_kepler_u0_(\d+)\.npz$",
            fn.name,
        )

        if match is None:
            raise RuntimeError(
                f"Cannot parse u0 index from {fn}"
            )

        return int(
            match.group(1)
        )

    files = sorted(
        files,
        key=file_index,
    )

    u0_values = []
    D_rows = []
    success_rows = []

    P_ref = None
    tE_ref = None

    for fn in files:

        with np.load(
            fn,
            allow_pickle=False,
        ) as d:

            required = [
                "truth",
                "P_grid",
                "D",
                "SUCCESS",
            ]

            missing = [
                key
                for key in required
                if key not in d.files
            ]

            if missing:
                raise RuntimeError(
                    f"{fn} missing keys: {missing}"
                )

            truth = np.asarray(
                d["truth"],
                dtype=float,
            )

            u0 = float(
                truth[1]
            )

            tE = float(
                truth[2]
            )

            P = np.asarray(
                d["P_grid"],
                dtype=float,
            )

            D = np.asarray(
                d["D"],
                dtype=float,
            )

            success = np.asarray(
                d["SUCCESS"],
                dtype=bool,
            )

            if not (
                len(P)
                == len(D)
                == len(success)
            ):
                raise RuntimeError(
                    f"Inconsistent array lengths in {fn}"
                )

            if P_ref is None:

                P_ref = P.copy()
                tE_ref = tE

            else:

                if not np.allclose(
                    P,
                    P_ref,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        f"Inconsistent P_grid in {fn}"
                    )

                if not np.isclose(
                    tE,
                    tE_ref,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        f"Inconsistent tE in {fn}"
                    )

            u0_values.append(
                u0
            )

            D_rows.append(
                D
            )

            success_rows.append(
                success
            )

    u0_grid = np.asarray(
        u0_values,
        dtype=float,
    )

    D_intrinsic = np.asarray(
        D_rows,
        dtype=float,
    )

    success_intrinsic = np.asarray(
        success_rows,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Optional exact row selection
    # --------------------------------------------------------

    if u0_max is not None:

        mask = (
            u0_grid
            <= float(u0_max)
        )

        if not np.any(mask):
            raise RuntimeError(
                f"No intrinsic u0 values <= {u0_max}"
            )

        u0_grid = (
            u0_grid[mask]
        )

        D_intrinsic = (
            D_intrinsic[mask]
        )

        success_intrinsic = (
            success_intrinsic[mask]
        )

    if not np.all(
        success_intrinsic
    ):
        nbad = int(
            np.count_nonzero(
                ~success_intrinsic
            )
        )

        raise RuntimeError(
            f"Intrinsic grid contains {nbad} failed fits"
        )

    if not np.all(
        np.isfinite(
            D_intrinsic
        )
    ):
        raise RuntimeError(
            "Intrinsic D map contains non-finite values"
        )

    return {
        "directory": str(
            directory
        ),

        "u0_grid": u0_grid,

        "P_grid": P_ref,

        "P_over_tE": (
            P_ref
            / float(tE_ref)
        ),

        "D_intrinsic": (
            D_intrinsic
        ),

        "success_intrinsic": (
            success_intrinsic
        ),

        "tE_intrinsic": float(
            tE_ref
        ),
    }


# ============================================================
# Full grid
# ============================================================

def run_grid(
    t,
    t0_true,
    tE_true,
    u0_grid,
    P_grid,
    magnitudes,
    output_npz,
    D_intrinsic=None,
    intrinsic_grid_dir=None,
):
    """
    Ejecuta la grilla Roman.

    The result has shape:

        (Nmag, Nu0, NP)
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    u0_grid = np.asarray(
        u0_grid,
        dtype=float,
    )

    P_grid = np.asarray(
        P_grid,
        dtype=float,
    )

    shape = (
        len(magnitudes),
        len(u0_grid),
        len(P_grid),
    )

    if D_intrinsic is not None:

        D_intrinsic = np.asarray(
            D_intrinsic,
            dtype=float,
        )

        expected_intrinsic_shape = (
            len(u0_grid),
            len(P_grid),
        )

        if (
            D_intrinsic.shape
            != expected_intrinsic_shape
        ):
            raise ValueError(
                "D_intrinsic has shape "
                f"{D_intrinsic.shape}, expected "
                f"{expected_intrinsic_shape}"
            )

    DELTA_CHI2 = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    SQRT_DELTA_CHI2 = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    SNR_EVENT = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    D_ROMAN_EFF = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    XI_REL = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    BEST_T0 = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    BEST_U0 = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    BEST_TE = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    DT0_OVER_TE = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    DU0_OVER_U0 = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    DTE_OVER_TE = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    SUCCESS = np.zeros(
        shape,
        dtype=bool,
    )

    N_BRIGHT_FLOOR = np.zeros(
        shape,
        dtype=np.int32,
    )

    rows = []

    n_total = (
        len(magnitudes)
        * len(u0_grid)
        * len(P_grid)
    )

    counter = 0

    print()
    print("=" * 80)
    print("ROMAN ASIMOV GRID")
    print("=" * 80)
    print(
        f"Nmag = {len(magnitudes)}"
    )
    print(
        f"Nu0  = {len(u0_grid)}"
    )
    print(
        f"NP   = {len(P_grid)}"
    )
    print(
        f"Total fits = {n_total}"
    )
    print(
        f"N Roman epochs / fit = {len(t)}"
    )
    print("=" * 80)

    for im, source_mag in enumerate(
        magnitudes
    ):

        for iu, u0_true in enumerate(
            u0_grid
        ):

            print()
            print(
                f"[F146={source_mag:.2f}] "
                f"u0 {iu+1}/{len(u0_grid)} = "
                f"{u0_true:.6g}"
            )

            for ip, P_days in enumerate(
                P_grid
            ):

                counter += 1

                try:

                    result = run_case(
                        t=t,
                        t0_true=t0_true,
                        u0_true=float(
                            u0_true
                        ),
                        tE_true=tE_true,
                        P_days=float(
                            P_days
                        ),
                        source_mag=float(
                            source_mag
                        ),
                    )

                    DELTA_CHI2[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "delta_chi2"
                        ]
                    )

                    SQRT_DELTA_CHI2[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "sqrt_delta_chi2"
                        ]
                    )

                    SNR_EVENT[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "snr_event"
                        ]
                    )

                    D_ROMAN_EFF[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "D_roman_eff"
                        ]
                    )

                    XI_REL[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "xi_rel"
                        ]
                    )

                    BEST_T0[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "best_t0"
                        ]
                    )

                    BEST_U0[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "best_u0"
                        ]
                    )

                    BEST_TE[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "best_tE"
                        ]
                    )

                    DT0_OVER_TE[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "dt0_over_tE"
                        ]
                    )

                    DU0_OVER_U0[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "du0_over_u0"
                        ]
                    )

                    DTE_OVER_TE[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "dtE_over_tE"
                        ]
                    )

                    SUCCESS[
                        im,
                        iu,
                        ip,
                    ] = True

                    N_BRIGHT_FLOOR[
                        im,
                        iu,
                        ip,
                    ] = (
                        result[
                            "n_bright_floor"
                        ]
                    )

                    rows.append(
                        result
                    )

                except Exception as exc:

                    print()
                    print(
                        f"[FAILED {counter}/{n_total}] "
                        f"F146={source_mag:g}, "
                        f"u0={u0_true:g}, "
                        f"P/tE={P_days/tE_true:g}"
                    )

                    print(
                        f"{type(exc).__name__}: {exc}"
                    )

            print(
                f"  completed: "
                f"{counter}/{n_total}"
            )

        # ----------------------------------------------------
        # Checkpoint after each magnitude
        # ----------------------------------------------------

        save_grid_npz(
            output_npz=output_npz,
            t=t,
            t0_true=t0_true,
            tE_true=tE_true,
            magnitudes=magnitudes,
            u0_grid=u0_grid,
            P_grid=P_grid,
            DELTA_CHI2=DELTA_CHI2,
            SQRT_DELTA_CHI2=(
                SQRT_DELTA_CHI2
            ),
            SNR_EVENT=SNR_EVENT,
            D_ROMAN_EFF=D_ROMAN_EFF,
            XI_REL=XI_REL,
            BEST_T0=BEST_T0,
            BEST_U0=BEST_U0,
            BEST_TE=BEST_TE,
            DT0_OVER_TE=DT0_OVER_TE,
            DU0_OVER_U0=DU0_OVER_U0,
            DTE_OVER_TE=DTE_OVER_TE,
            SUCCESS=SUCCESS,
            N_BRIGHT_FLOOR=(
                N_BRIGHT_FLOOR
            ),
            D_intrinsic=D_intrinsic,
            intrinsic_grid_dir=(
                intrinsic_grid_dir
            ),
        )

    # ========================================================
    # CSV plano
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    csv_path = (
        Path(output_npz)
        .with_suffix(".csv")
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    return df


# ============================================================
# Save
# ============================================================

def save_grid_npz(
    output_npz,
    t,
    t0_true,
    tE_true,
    magnitudes,
    u0_grid,
    P_grid,
    DELTA_CHI2,
    SQRT_DELTA_CHI2,
    SNR_EVENT,
    D_ROMAN_EFF,
    XI_REL,
    BEST_T0,
    BEST_U0,
    BEST_TE,
    DT0_OVER_TE,
    DU0_OVER_U0,
    DTE_OVER_TE,
    SUCCESS,
    N_BRIGHT_FLOOR,
    D_intrinsic=None,
    intrinsic_grid_dir=None,
):
    output_npz = Path(
        output_npz
    )

    output_npz.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    commit, dirty = (
        git_state()
    )

    intrinsic_payload = {}

    if D_intrinsic is not None:

        intrinsic_payload[
            "D_INTRINSIC"
        ] = np.asarray(
            D_intrinsic,
            dtype=float,
        )

    if intrinsic_grid_dir is not None:

        intrinsic_payload[
            "intrinsic_grid_dir"
        ] = np.array(
            str(intrinsic_grid_dir)
        )

    np.savez_compressed(
        output_npz,

        **intrinsic_payload,

        # grids
        roman_times=np.asarray(
            t,
            dtype=float,
        ),

        f146_magnitudes=np.asarray(
            magnitudes,
            dtype=float,
        ),

        u0_grid=np.asarray(
            u0_grid,
            dtype=float,
        ),

        P_grid=np.asarray(
            P_grid,
            dtype=float,
        ),

        P_over_tE=np.asarray(
            P_grid,
            dtype=float,
        ) / float(tE_true),

        # observability
        DELTA_CHI2=DELTA_CHI2,
        SQRT_DELTA_CHI2=(
            SQRT_DELTA_CHI2
        ),
        SNR_EVENT=SNR_EVENT,

        D_ROMAN_EFF=D_ROMAN_EFF,

        # physical scale
        XI_REL=XI_REL,

        # fit
        BEST_T0=BEST_T0,
        BEST_U0=BEST_U0,
        BEST_TE=BEST_TE,

        DT0_OVER_TE=DT0_OVER_TE,
        DU0_OVER_U0=DU0_OVER_U0,
        DTE_OVER_TE=DTE_OVER_TE,

        SUCCESS=SUCCESS,

        N_BRIGHT_FLOOR=(
            N_BRIGHT_FLOOR
        ),

        # true configuration
        t0_true=np.float64(
            t0_true
        ),

        tE_true=np.float64(
            tE_true
        ),

        M1_Msun=np.float64(
            M1_MSUN
        ),

        M2_Msun=np.float64(
            M2_MSUN
        ),

        Mtot_Msun=np.float64(
            MTOT_MSUN
        ),

        q_mass=np.float64(
            Q_MASS
        ),

        qflux=np.float64(
            QFLUX_TRUE
        ),

        rEhat_AU=np.float64(
            REHAT_AU
        ),

        theta=np.float64(
            THETA_TRUE
        ),

        phi=np.float64(
            PHI_TRUE
        ),

        inclination=np.float64(
            INCLINATION_TRUE
        ),

        fit_window_tE=np.float64(
            FIT_WINDOW_TE
        ),

        # method
        asimov=np.bool_(
            True
        ),

        parallax=np.array(
            "None"
        ),

        fit_model=np.array(
            "PSPL"
        ),

        fit_method=np.array(
            "pyLIMA_TRF"
        ),

        photometric_model=np.array(
            "sigma_F146_func"
        ),

        interpretation=np.array(
            "DELTA_CHI2 = chi2_PSPL_best because "
            "chi2_BSPL = 0 for Asimov data; "
            "D_ROMAN_EFF = sqrt(DELTA_CHI2) / SNR_EVENT"
        ),

        code_commit=np.array(
            commit
        ),

        code_dirty=np.bool_(
            dirty
        ),

        n_success=np.int64(
            np.count_nonzero(
                SUCCESS
            )
        ),

        n_total=np.int64(
            SUCCESS.size
        ),
    )

    print()
    print(
        f"Checkpoint saved: {output_npz}"
    )


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Roman-weighted BSPL -> PSPL "
            "Asimov model-separation experiment."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "grid",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=TE_TRUE,
    )

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
        help=(
            "F146 baseline magnitude used "
            "in smoke mode."
        ),
    )

    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[
            19.0,
            21.0,
            23.0,
        ],
        help=(
            "F146 baseline magnitudes "
            "for grid mode."
        ),
    )

    parser.add_argument(
        "--u0-min",
        type=float,
        default=1.0e-2,
    )

    parser.add_argument(
        "--u0-max",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--n-u0",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--P-min",
        type=float,
        default=10.0,
        help="Minimum period [days].",
    )

    parser.add_argument(
        "--P-max",
        type=float,
        default=1.0e5,
        help="Maximum period [days].",
    )

    parser.add_argument(
        "--n-P",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--anchor-season",
        type=int,
        default=2,
        help=(
            "Index of nominal Roman season "
            "whose midpoint is t0."
        ),
    )

    parser.add_argument(
        "--no-off-seasons",
        action="store_true",
        help=(
            "Do not include the sparse "
            "off-season cadence."
        ),
    )

    parser.add_argument(
        "--fit-window-tE",
        type=float,
        default=FIT_WINDOW_TE,
    )

    parser.add_argument(
        "--intrinsic-grid-dir",
        default=None,
        help=(
            "Directory containing scan_kepler_u0_*.npz. "
            "If supplied, Roman uses exactly the same u0 and "
            "period nodes as the intrinsic BSPL-PSPL scan."
        ),
    )

    parser.add_argument(
        "--intrinsic-u0-max",
        type=float,
        default=1.0,
        help=(
            "Maximum u0 retained from --intrinsic-grid-dir. "
            "Use a negative value to keep the full intrinsic grid."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/roman_asimov/"
            "roman_qf0_u0_PoverTE_tE150.npz"
        ),
    )

    return parser


def main():

    args = (
        build_parser()
        .parse_args()
    )

    tE_true = float(
        args.tE
    )

    t, t0_true = (
        build_roman_times(
            tE_true=tE_true,
            anchor_season_index=(
                args.anchor_season
            ),
            fit_window_te=(
                args.fit_window_tE
            ),
            include_off_seasons=(
                not args.no_off_seasons
            ),
        )
    )

    print()
    print("=" * 80)
    print("ROMAN SAMPLING")
    print("=" * 80)

    print(
        f"N epochs = {len(t)}"
    )

    print(
        f"t0 = {t0_true:.8f} JD"
    )

    print(
        f"time min = {np.min(t):.8f}"
    )

    print(
        f"time max = {np.max(t):.8f}"
    )

    print(
        f"window = +/- "
        f"{args.fit_window_tE:g} tE"
    )

    print(
        "off seasons = "
        f"{not args.no_off_seasons}"
    )

    print("=" * 80)

    if args.mode == "smoke":

        df = run_smoke_test(
            t=t,
            t0_true=t0_true,
            tE_true=tE_true,
            source_mag=(
                args.source_mag
            ),
        )

        if len(df) > 0:

            print()
            print(
                df[
                    [
                        "source_mag",
                        "u0_true",
                        "P_over_tE",
                        "xi_rel",
                        "delta_chi2",
                        "sqrt_delta_chi2",
                        "snr_event",
                        "D_roman_eff",
                        "du0_over_u0",
                        "dtE_over_tE",
                    ]
                ].to_string(
                    index=False
                )
            )

        return

    # ========================================================
    # Grid mode
    # ========================================================

    if args.u0_min <= 0:
        raise ValueError(
            "u0-min must be > 0."
        )

    if args.P_min <= 0:
        raise ValueError(
            "P-min must be > 0."
        )

    D_intrinsic = None
    intrinsic_grid_dir = None

    if args.intrinsic_grid_dir is not None:

        if args.intrinsic_u0_max < 0.0:
            intrinsic_u0_max = None
        else:
            intrinsic_u0_max = (
                args.intrinsic_u0_max
            )

        intrinsic = (
            load_intrinsic_u0_period_grid(
                directory=(
                    args.intrinsic_grid_dir
                ),
                u0_max=(
                    intrinsic_u0_max
                ),
            )
        )

        u0_grid = (
            intrinsic[
                "u0_grid"
            ]
        )

        P_grid = (
            intrinsic[
                "P_grid"
            ]
        )

        D_intrinsic = (
            intrinsic[
                "D_intrinsic"
            ]
        )

        intrinsic_grid_dir = (
            intrinsic[
                "directory"
            ]
        )

        if not np.isclose(
            intrinsic[
                "tE_intrinsic"
            ],
            tE_true,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Intrinsic tE does not match Roman tE: "
                f"{intrinsic['tE_intrinsic']} vs {tE_true}"
            )

        print()
        print("=" * 80)
        print("INTRINSIC GRID IMPORTED")
        print("=" * 80)

        print(
            f"directory = {intrinsic_grid_dir}"
        )

        print(
            f"Nu0 = {len(u0_grid)}"
        )

        print(
            f"NP = {len(P_grid)}"
        )

        print(
            f"u0 = "
            f"{u0_grid.min():.8g} -- "
            f"{u0_grid.max():.8g}"
        )

        print(
            f"P/tE = "
            f"{(P_grid/tE_true).min():.8g} -- "
            f"{(P_grid/tE_true).max():.8g}"
        )

        print(
            f"D intrinsic = "
            f"{D_intrinsic.min():.8e} -- "
            f"{D_intrinsic.max():.8e}"
        )

        print("=" * 80)

    else:

        u0_grid = np.logspace(
            np.log10(
                args.u0_min
            ),
            np.log10(
                args.u0_max
            ),
            args.n_u0,
        )

        P_grid = np.logspace(
            np.log10(
                args.P_min
            ),
            np.log10(
                args.P_max
            ),
            args.n_P,
        )

    run_grid(
        t=t,
        t0_true=t0_true,
        tE_true=tE_true,
        u0_grid=u0_grid,
        P_grid=P_grid,
        magnitudes=(
            args.magnitudes
        ),
        output_npz=(
            args.output
        ),
        D_intrinsic=D_intrinsic,
        intrinsic_grid_dir=(
            intrinsic_grid_dir
        ),
    )


if __name__ == "__main__":
    main()

