"""
Roman BSPL/xallarap observational experiment.

Project-specific responsibilities kept here:
- binary-source physical parameterization;
- conversion P -> xi_rel;
- q_M = M2/M1;
- q_flux;
- choice of truth model;
- choice of PSPL vs PSPL+parallax comparison.

Roman/Rubin responsibilities are NOT reimplemented here:
- Roman observing times/cadence;
- ephemerides;
- F146/Pandeia noise model;
- noisy/Asimov photometric realization;
- pyLIMA fitting machinery.

Those come from the canonical roman_rubin repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]


# ============================================================================
# Binary-source science defaults
# ============================================================================

M1_MSUN = 2.0
M2_MSUN = 1.0

Q_MASS = M2_MSUN / M1_MSUN
MTOT_MSUN = M1_MSUN + M2_MSUN

REHAT_AU = 5.0

TE_DAYS = 30.0
U0 = 0.1

PHI = 0.0
INCLINATION = np.pi / 2.0
THETA = 0.0

Q_FLUX = 0.0

T0 = 2461849.0

# Physical event coordinates (deg). The current roman_rubin requires
# data['ra'], data['dec']: they define the pyLIMA Event, hence the
# North/East basis of the parallax, and the Event used by the fits.
# Same Roman/Rubin field as the historical roman_rubin RA_FIELD/DEC_FIELD.
EVENT_RA_DEG = 267.92497054815516
EVENT_DEC_DEG = -29.152232510353276

FIT_WINDOW_TE = 3.5


# ============================================================================
# roman_rubin loader
# ============================================================================

def get_roman_rubin_root():
    env = os.environ.get(
        "ROMAN_RUBIN_REPO"
    )

    candidates = []

    if env:
        candidates.append(
            Path(env).expanduser()
        )

    # CHE layout: binary_source_* and roman_rubin are siblings.
    candidates.append(
        REPO_ROOT.parent
        / "roman_rubin"
    )

    candidates.append(
        Path(
            "/share/storage3/rubin/microlensing/"
            "romanrubin/roman_rubin"
        )
    )

    for candidate in candidates:
        candidate = candidate.resolve()

        if (
            candidate.exists()
            and
            (candidate / "functions_roman_rubin.py").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "No pude encontrar el repo roman_rubin. "
        "Definí ROMAN_RUBIN_REPO."
    )


def load_roman_rubin():
    root = get_roman_rubin_root()

    if str(root) not in sys.path:
        sys.path.insert(
            0,
            str(root),
        )

    import functions_roman_rubin as frr
    import fit_lc

    try:
        from photometry.roman_photometry import (
            get_roman_f146_noise_model,
        )
    except ImportError:
        from stellar_population.noise_models.roman_photometry import (
            get_roman_f146_noise_model,
        )

    return (
        root,
        frr,
        fit_lc,
        get_roman_f146_noise_model,
    )


# ============================================================================
# F146 magnitude convention
# ============================================================================

def f146_ab_to_roman_input_vega(
    magnitude_ab,
):
    """
    binary_source manuscript magnitudes are F146 AB.

    roman_rubin's W149-named simulation channel stores the source
    magnitude in the historical Vega convention and converts to AB
    internally before querying Pandeia.

    Derive the calibrated offset from roman_rubin itself rather than
    duplicating it here.
    """
    (
        _,
        _,
        _,
        get_noise_model,
    ) = load_roman_rubin()

    noise_model = get_noise_model()

    offset = float(
        np.asarray(
            noise_model.vega_to_ab(
                0.0
            )
        )
    )

    magnitude_vega = (
        float(magnitude_ab)
        - offset
    )

    roundtrip = float(
        np.asarray(
            noise_model.vega_to_ab(
                magnitude_vega
            )
        )
    )

    if not np.isclose(
        roundtrip,
        float(magnitude_ab),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "F146 AB/Vega conversion failed: "
            f"AB={magnitude_ab}, "
            f"Vega={magnitude_vega}, "
            f"roundtrip={roundtrip}"
        )

    return magnitude_vega


# ============================================================================
# Binary-source physics
# ============================================================================

def xallarap_parameters(
    P_days,
    M1_Msun=M1_MSUN,
    M2_Msun=M2_MSUN,
    rEhat_AU=REHAT_AU,
    theta=THETA,
    xi_rel_override=None,
):
    """
    Same physical convention as the manuscript intrinsic grids:

        q_M = M2/M1

        a_rel^3 = (M1+M2) P_yr^2

        xi_rel = a_rel / rEhat

    pyLIMA then distributes the relative orbital displacement between
    the two sources through xi_mass_ratio.
    """
    P_days = float(P_days)

    if P_days <= 0:
        raise ValueError(
            "P_days must be positive."
        )

    M1_Msun = float(M1_Msun)
    M2_Msun = float(M2_Msun)

    q_mass = (
        M2_Msun
        / M1_Msun
    )

    Mtot = (
        M1_Msun
        + M2_Msun
    )

    P_years = (
        P_days
        / 365.25
    )

    a_rel_AU = (
        Mtot
        * P_years**2
    ) ** (1.0 / 3.0)

    xi_rel = (
        a_rel_AU
        / float(rEhat_AU)
    )

    # Dimensionless scans fix xi_rel directly (same role as
    # override_xiE in degeneracy_fit); a_rel_AU is then only metadata.
    if xi_rel_override is not None:
        xi_rel = float(xi_rel_override)

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
        / P_days
    )

    return {
        "P_days": P_days,
        "M1_Msun": M1_Msun,
        "M2_Msun": M2_Msun,
        "Mtot_Msun": Mtot,
        "q_mass": q_mass,
        "a_rel_AU": a_rel_AU,
        "rEhat_AU": float(rEhat_AU),
        "xi_rel": xi_rel,
        "xi_para": xi_para,
        "xi_perp": xi_perp,
        "xi_angular_velocity": omega,
        "xi_rel_override": xi_rel_override is not None,
    }


def build_event_params(
    P_days,
    u0=U0,
    tE=TE_DAYS,
    t0=T0,
    source_mag_ab=21.0,
    q_flux=Q_FLUX,
    phi=PHI,
    inclination=INCLINATION,
    theta=THETA,
    xi_rel=None,
):
    xal = xallarap_parameters(
        P_days=P_days,
        theta=theta,
        xi_rel_override=xi_rel,
    )

    roman_mag_vega = (
        f146_ab_to_roman_input_vega(
            source_mag_ab
        )
    )

    params = {
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),

        # Controlled xallarap -> apparent parallax experiment:
        # no real microlensing parallax is injected.
        "piEN": 0.0,
        "piEE": 0.0,

        "xi_para":
            float(
                xal["xi_para"]
            ),

        "xi_perp":
            float(
                xal["xi_perp"]
            ),

        "xi_angular_velocity":
            float(
                xal[
                    "xi_angular_velocity"
                ]
            ),

        "xi_phase":
            float(phi),

        "xi_inclination":
            float(inclination),

        "xi_mass_ratio":
            float(
                xal["q_mass"]
            ),

        "q_flux":
            float(q_flux),

        # Historical channel name in roman_rubin.
        "W149":
            float(
                roman_mag_vega
            ),

        # Explicitly no unrelated third-light blending.
        "blend_ratio_W149":
            0.0,

        # Physical event coordinates (required by roman_rubin sim_event).
        "ra": float(EVENT_RA_DEG),
        "dec": float(EVENT_DEC_DEG),

        # Project metadata.
        "F146_AB":
            float(
                source_mag_ab
            ),
    }

    return (
        params,
        xal,
    )


# ============================================================================
# Canonical Roman simulation
# ============================================================================

def simulate_roman_bspl(
    event_index,
    P_days,
    u0=U0,
    tE=TE_DAYS,
    t0=T0,
    source_mag_ab=21.0,
    asimov=True,
    truth_parallax=False,
    q_flux=Q_FLUX,
    time_window_te=FIT_WINDOW_TE,
    apply_photometric_filter=False,
    phi=PHI,
    inclination=INCLINATION,
    theta=THETA,
    xi_rel=None,
):
    """
    Simulate a Roman BSPL/xallarap light curve entirely through
    roman_rubin.

    asimov=True:
        keep Roman/Pandeia uncertainties, but do not draw Gaussian noise.

    asimov=False:
        standard roman_rubin noisy photometric realization.
    """
    (
        root,
        frr,
        _,
        _,
    ) = load_roman_rubin()

    event_params, xal = (
        build_event_params(
            P_days=P_days,
            u0=u0,
            tE=tE,
            t0=t0,
            source_mag_ab=source_mag_ab,
            q_flux=q_flux,
            phi=phi,
            inclination=inclination,
            theta=theta,
            xi_rel=xi_rel,
        )
    )

    # roman_rubin main requires a materialized realization: fsource_<band>,
    # ftotal_<band> and blend_ratio_<band> (generating the fluxes from the
    # magnitude at simulation time was removed). Same formula as the removed
    # flux_parameters_from_blend_ratio: F = 10**((ZP - mag)/2.5),
    # fsource = F/(1+g), ftotal = F, with g = blend_ratio_W149 and the
    # simulation zero point of roman_rubin.
    from photometry.constants import SIMULATION_BAND_ZERO_POINTS

    g_w149 = float(event_params["blend_ratio_W149"])
    f_base = 10.0 ** (
        (
            float(SIMULATION_BAND_ZERO_POINTS["W149"])
            - float(event_params["W149"])
        )
        / 2.5
    )

    event_params.setdefault(
        "fsource_W149",
        f_base / (1.0 + g_w149),
    )

    event_params.setdefault(
        "ftotal_W149",
        f_base,
    )

    half_window = (
        float(time_window_te)
        * float(tE)
    )

    time_window = (
        float(t0) - half_window,
        float(t0) + half_window,
    )

    ephemerides = (
        root
        / "ephemerides"
        / "Roman_positions_2027_2032_periodic_v1.npy"
    )

    # roman_rubin main: sim_event (simulate_event_for_fit was a thin
    # wrapper around it and has been removed).
    model, py_params, decision = (
        frr.sim_event(
            int(event_index),
            dict(
                event_params
            ),
            str(
                ephemerides
            ),
            "PSPL",
            time_window=time_window,
            use_roman=True,
            use_rubin=False,
            truth_parallax=bool(
                truth_parallax
            ),
            apply_detection_criteria=False,
            apply_photometric_filter=bool(
                apply_photometric_filter
            ),
            truth_double_source=True,
            add_photometric_noise=(
                not bool(asimov)
            ),
        )
    )

    if model is None:
        raise RuntimeError(
            "roman_rubin returned model=None."
        )

    return {
        "model": model,
        "py_params": py_params,
        "decision": decision,
        "event_params": event_params,
        "xallarap": xal,
        "ephemerides": ephemerides,
        "asimov": bool(asimov),
    }


# ============================================================================
# PSPL / PSPL+parallax fits using roman_rubin fitter
# ============================================================================

def _chi2(fit):
    result = fit.fit_results

    if "chi2" in result:
        return float(
            result["chi2"]
        )

    loss = getattr(
        fit,
        "loss_function",
        None,
    )

    if (
        loss is not None
        and loss in result
    ):
        return float(
            result[loss]
        )

    raise KeyError(
        f"Cannot locate chi2: {list(result)}"
    )


def pylima_covariance(fit):
    """
    Parameter names, covariance and 1-sigma uncertainties exactly as
    returned by the pyLIMA TRF fit (fit_results['covariance_matrix'],
    in physical coordinates for the default fit coordinates).

    pyLIMA rescales (J^T J)^{-1} by chi2/(N - k). This is the intended
    estimate for light curves WITH photometric noise; it is not meaningful
    for noiseless (Asimov) light curves.
    """
    names = list(
        fit.fit_parameters.keys()
    )

    results = fit.fit_results

    cov = results.get(
        "covariance_matrix_physical",
        None,
    )

    if cov is None:
        cov = results["covariance_matrix"]

    cov = np.asarray(
        cov,
        dtype=float,
    )

    return {
        "names": names,
        "covariance": cov,
        "sigma": {
            name: float(np.sqrt(max(cov[k, k], 0.0)))
            for k, name in enumerate(names)
        },
    }


def fit_pspl_comparison(
    simulation,
    output_dir,
    seed=20261019,
):
    """
    Fit the SAME Roman realization with:

        A: PSPL, no parallax
        B: PSPL + annual parallax

    Both fits use roman_rubin -> pyLIMA TRFfit.
    """
    (
        _,
        frr,
        fit_lc,
        _,
    ) = load_roman_rubin()

    model_true = simulation[
        "model"
    ]

    params = simulation[
        "event_params"
    ]

    ephemerides = simulation[
        "ephemerides"
    ]

    curves, _ = (
        frr.extract_lightcurves_for_fit(
            model_true
        )
    )

    roman_lc = curves[
        "W149"
    ]

    event_ra, event_dec = (
        frr.get_event_coordinates(
            model_true
        )
    )

    # get_model_origin was removed from roman_rubin main; it was
    # getattr(model, "origin", None).
    origin = getattr(
        model_true,
        "origin",
        None,
    )

    t0 = float(
        params["t0"]
    )

    u0 = float(
        params["u0"]
    )

    tE = float(
        params["tE"]
    )

    # Bounds scale with tE: |t0 - t0_true| <= 2 tE and
    # 0.02 <= tE/tE_true <= 20 (fixed 10 d / 100 d bounds
    # cannot represent tE = 100-200 d events).
    common_bounds = {
        "t0": {
            "bounds": [
                t0 - 2.0 * tE,
                t0 + 2.0 * tE,
            ],
        },
        "u0": {
            "bounds": [
                -2.0,
                2.0,
            ],
        },
        "tE": {
            "bounds": [
                0.02 * tE,
                20.0 * tE,
            ],
        },
    }

    h1_bounds = {
        **common_bounds,
        "piEN": {
            "bounds": [
                -40.0,
                40.0,
            ],
        },
        "piEE": {
            "bounds": [
                -40.0,
                40.0,
            ],
        },
    }

    output_dir = Path(
        output_dir
    )

    h0_dir = (
        output_dir
        / "H0"
    )

    h1_dir = (
        output_dir
        / "H1"
    )

    h0_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    h1_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit_h0, _, _ = (
        fit_lc.fit_rubin_roman(
            Source=int(seed),
            event_params=params,
            path_save=h0_dir,
            path_ephemerides=str(
                ephemerides
            ),
            model="PSPL",
            algo="TRF",
            Origin=origin,
            rango=0,
            wfirst_lc=roman_lc,
            lsst_u=[],
            lsst_g=[],
            lsst_r=[],
            lsst_i=[],
            lsst_z=[],
            lsst_y=[],
            fit_model="PSPL",
            fit_parallax=False,
            fit_bounds=common_bounds,
            initial_guess={
                "t0": t0,
                "u0": u0,
                "tE": tE,
            },
            random_state=int(seed),
            event_ra=event_ra,
            event_dec=event_dec,
        )
    )

    best_h0 = np.asarray(
        fit_h0.fit_results[
            "best_model"
        ],
        dtype=float,
    )

    def run_h1(initial, flux_method=None, subdir="H1"):
        h1_path = output_dir / subdir
        h1_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        fit, _, _ = fit_lc.fit_rubin_roman(
            Source=int(seed),
            event_params=params,
            path_save=h1_path,
            path_ephemerides=str(
                ephemerides
            ),
            model="PSPL",
            algo="TRF",
            Origin=origin,
            rango=0,
            wfirst_lc=roman_lc,
            lsst_u=[],
            lsst_g=[],
            lsst_r=[],
            lsst_i=[],
            lsst_z=[],
            lsst_y=[],
            fit_model="PSPL",
            fit_parallax=True,
            fit_bounds=h1_bounds,
            initial_guess=initial,
            random_state=int(seed),
            event_ra=event_ra,
            event_dec=event_dec,
            flux_method=flux_method,
        )

        return fit

    # H1 starts exactly from the H0 optimum with pi_E = 0.
    h1_start = {
        "t0": float(best_h0[0]),
        "u0": float(best_h0[1]),
        "tE": float(best_h0[2]),
        "piEN": 0.0,
        "piEE": 0.0,
    }

    fit_h1 = run_h1(h1_start)

    chi2_h0 = _chi2(
        fit_h0
    )

    chi2_h1 = _chi2(
        fit_h1
    )

    # PSPL is nested in PSPL+parallax (pi_E = 0), so a converged H1 fit
    # must satisfy chi2_h1 <= chi2_h0. If not, the H1 optimizer stopped in
    # a worse local minimum (its flux start is chosen by pyLIMA, not taken
    # from H0). Retry with profiled fluxes (roman_rubin bounded_profile),
    # both signs of u0 and small pi_E offsets, and keep the best H1.
    nested_tol = 1e-6 * (1.0 + abs(chi2_h0))
    h1_attempts = 1

    if chi2_h1 > chi2_h0 + nested_tol:

        retry_starts = []

        for u0_start in (h1_start["u0"], -h1_start["u0"]):
            for piEN, piEE in (
                (0.0, 0.0),
                (0.05, 0.0),
                (-0.05, 0.0),
                (0.0, 0.05),
                (0.0, -0.05),
            ):
                retry_starts.append(
                    dict(
                        h1_start,
                        u0=float(u0_start),
                        piEN=piEN,
                        piEE=piEE,
                    )
                )

        for k, start in enumerate(retry_starts):
            try:
                fit_try = run_h1(
                    start,
                    flux_method="bounded_profile",
                    subdir=f"H1_retry_{k:02d}",
                )
            except Exception:
                continue

            h1_attempts += 1
            chi2_try = _chi2(
                fit_try
            )

            if chi2_try < chi2_h1:
                fit_h1 = fit_try
                chi2_h1 = chi2_try

            if chi2_h1 <= chi2_h0 + nested_tol:
                break

    h1_nested_ok = bool(
        chi2_h1 <= chi2_h0 + nested_tol
    )

    best_h1 = np.asarray(
        fit_h1.fit_results[
            "best_model"
        ],
        dtype=float,
    )

    return {
        "fit_h0": fit_h0,
        "fit_h1": fit_h1,

        "chi2_h0":
            chi2_h0,

        "chi2_h1":
            chi2_h1,

        "h1_nested_ok":
            h1_nested_ok,

        "h1_attempts":
            int(h1_attempts),

        "cov_h0":
            pylima_covariance(fit_h0),

        "cov_h1":
            pylima_covariance(fit_h1),

        "n_data":
            int(len(roman_lc)),

        "delta_chi2":
            chi2_h0
            - chi2_h1,

        "best_h0":
            best_h0,

        "best_h1":
            best_h1,
    }
