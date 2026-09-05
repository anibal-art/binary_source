#!/usr/bin/env python3
"""Generate two fixed BSPL--PSPL examples for the manuscript.

This is a non-interactive version of the exploratory widget used during the
project.  It generates exactly two dark-companion examples:

1. A clearly distinguishable BSPL event: P = 210 d.
2. A strongly confused BSPL/PSPL event: P = 6000 d.

The physical source masses are kept fixed at M1=2 Msun, M2=1 Msun
(Mtot=3 Msun, qM=0.5), so Kepler's law gives a_rel ~ 1 AU and ~9.3 AU,
respectively.  The projected Einstein radius at the source is chosen to be
approximately 5 AU through the lens geometry.

The intrinsic mismatch D_BSPL-PSPL is computed with exactly the normalized
trapezoidal objective used in the manuscript, over t0 +/- 3.5 tE.

Outputs
-------
figures/figures_lightcurves/example_1.png
figures/figures_lightcurves/example_1.pdf
figures/figures_lightcurves/example_2_confused.png
figures/figures_lightcurves/example_2_confused.pdf
"""

from pathlib import Path
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from scipy.signal import find_peaks
from matplotlib.ticker import AutoMinorLocator

from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.simulations import simulator


# =============================================================================
# Repository paths
# =============================================================================

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "figures" / "figures_lightcurves"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Paper configuration
# =============================================================================

T0_TRUE = 0.0
U0_TRUE = 0.1
TE_TRUE = 30.0

M1_SOURCE = 2.0
M2_SOURCE = 1.0
MTOT_SOURCE = M1_SOURCE + M2_SOURCE
Q_MASS = M2_SOURCE / M1_SOURCE
Q_FLUX = 0.0  # dark companion: only source 1 is luminous

# Lens/source geometry chosen so that R_E projected to the source plane is
# approximately 5 AU, matching the fiducial scale adopted in the manuscript.
M_LENS = 0.384
DL_KPC = 4.0
DS_KPC = 8.0

XI_SOURCE = "source1"
THETA = 0.0
PHI0 = np.pi
LAMBDA_XI = 0.0

METRIC_WINDOW_TE = 3.5
N_POINTS = 10_000
TRAJECTORY_LIMIT = 1.15
D_REFERENCE = 1.0e-2

# The confused example is fixed.  The distinguishable example is selected
# from a small targeted geometry search and then recomputed at full resolution.
CASES = [
    {
        "name": "Clearly distinguishable BSPL",
        "basename": "example_1",
        "expected": "distinguishable",
        "auto_search": True,
    },
    {
        "name": "BSPL confused with PSPL",
        "P_days": 6000.0,
        "u0_true": U0_TRUE,
        "theta": THETA,
        "phi0": PHI0,
        "lambda_xi": LAMBDA_XI,
        "basename": "example_2_confused",
        "expected": "confused",
        "auto_search": False,
    },
]

# Targeted search.  These are only used to choose an illustrative example,
# not as a scientific classification grid.
SEARCH_P_OVER_TE = [
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
]
SEARCH_U0 = [0.01, 0.03, 0.10]
SEARCH_THETA = [
    0.0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
]
SEARCH_PHI0 = [
    0.0,
    np.pi / 2,
    np.pi,
    3 * np.pi / 2,
]
SEARCH_LAMBDA = [
    0.0,
    np.pi / 2,
]

N_SEARCH_POINTS = 2000
D_DISTINGUISHABLE_MIN = 5.0e-3

# Morphological requirements for the illustrative example.
#
# These are NOT proposed as scientific detection criteria.  They only prevent
# the manuscript example from being a single central spike.
MIN_RESIDUAL_PEAKS = 2
MIN_SECOND_PEAK_RATIO = 0.20
MIN_TRAJECTORY_DEVIATION = 3.0e-3

# Peak-finding thresholds expressed relative to the strongest feature.
RESIDUAL_PEAK_PROMINENCE_FRAC = 0.05
RESIDUAL_PEAK_HEIGHT_FRAC = 0.15

# For local maxima in the BSPL light curve itself.
LIGHTCURVE_PROMINENCE_FRAC = 0.01



# =============================================================================
# Plot style
# =============================================================================

def set_paper_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "savefig.dpi": 300,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )


# =============================================================================
# Basic helpers
# =============================================================================

def _arr(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


def mag_to_flux(mag_value, zp=0.0):
    return 10.0 ** (-0.4 * (mag_value - zp))


def build_sim_event(t, mag0=19.0, emag=1.0e-9, filt="G"):
    ev = event.Event()
    ev.name = "Simulated"
    ev.ra = 170
    ev.dec = -70

    lc = np.c_[
        t,
        np.full_like(t, mag0),
        np.full_like(t, emag),
    ]

    tel = telescopes.Telescope(
        name="Simulation",
        camera_filter=filt,
        lightcurve=lc.astype(float),
        lightcurve_names=["time", "mag", "err_mag"],
        lightcurve_units=["JD", "mag", "mag"],
        location="Earth",
    )
    ev.telescopes.append(tel)
    return ev


# =============================================================================
# Physical scales
# =============================================================================

def thetaE_from_lens_mass(M_lens_Msun, Dl_kpc=4.0, Ds_kpc=8.0):
    if Dl_kpc <= 0 or Ds_kpc <= 0:
        raise ValueError("Dl and Ds must be positive.")
    if Dl_kpc >= Ds_kpc:
        raise ValueError("Require Dl < Ds.")

    kappa_mas_per_Msun = 8.144
    pi_rel_mas = 1.0 / Dl_kpc - 1.0 / Ds_kpc
    thetaE_mas = np.sqrt(kappa_mas_per_Msun * M_lens_Msun * pi_rel_mas)
    return thetaE_mas, pi_rel_mas


def kepler_binary_source_xiE(
    P_days,
    q_mass,
    Mtot_source_Msun,
    M_lens_Msun,
    Dl_kpc,
    Ds_kpc,
    tE_days,
    xi_source="source1",
):
    P_yr = P_days / 365.25

    M1_Msun = Mtot_source_Msun / (1.0 + q_mass)
    M2_Msun = q_mass * M1_Msun

    # Kepler in AU, yr, Msun: a^3 = Mtot P^2.
    a_rel_AU = (Mtot_source_Msun * P_yr**2) ** (1.0 / 3.0)
    a1_AU = a_rel_AU * M2_Msun / Mtot_source_Msun
    a2_AU = a_rel_AU * M1_Msun / Mtot_source_Msun

    thetaE_mas, pi_rel_mas = thetaE_from_lens_mass(
        M_lens_Msun=M_lens_Msun,
        Dl_kpc=Dl_kpc,
        Ds_kpc=Ds_kpc,
    )

    # 1 mas at 1 kpc = 1 AU.
    RE_source_AU = thetaE_mas * Ds_kpc

    if xi_source == "source1":
        a_source_AU = a1_AU
    elif xi_source == "source2":
        a_source_AU = a2_AU
    elif xi_source == "relative":
        a_source_AU = a_rel_AU
    else:
        raise ValueError("xi_source must be 'source1', 'source2', or 'relative'.")

    xiE = a_source_AU / RE_source_AU

    return xiE, {
        "P_days": P_days,
        "P_yr": P_yr,
        "Mtot_source_Msun": Mtot_source_Msun,
        "M1_Msun": M1_Msun,
        "M2_Msun": M2_Msun,
        "q_mass": q_mass,
        "a_rel_AU": a_rel_AU,
        "a1_AU": a1_AU,
        "a2_AU": a2_AU,
        "thetaE_mas": thetaE_mas,
        "pi_rel_mas": pi_rel_mas,
        "Dl_kpc": Dl_kpc,
        "Ds_kpc": Ds_kpc,
        "RE_source_AU": RE_source_AU,
        "tE_model_days": tE_days,
        "xi_source": xi_source,
        "xiE": xiE,
    }


# =============================================================================
# BSPL model
# =============================================================================

def build_binary_source_state(
    t,
    P_days,
    t0_true,
    u0_true,
    tE_true,
    Mtot_source_Msun,
    M_lens_Msun,
    Dl_kpc,
    Ds_kpc,
    xi_source,
    theta,
    phi0,
    lambda_xi,
    q_mass,
    qflux,
    ms=24.0,
    mtotal=24.0,
):
    omega = 2.0 * np.pi / float(P_days)

    xiE_use, kepler_info = kepler_binary_source_xiE(
        P_days=P_days,
        q_mass=q_mass,
        Mtot_source_Msun=Mtot_source_Msun,
        M_lens_Msun=M_lens_Msun,
        Dl_kpc=Dl_kpc,
        Ds_kpc=Ds_kpc,
        tE_days=tE_true,
        xi_source=xi_source,
    )

    xi_para = xiE_use * np.cos(theta)
    xi_perp = xiE_use * np.sin(theta)

    ev = build_sim_event(t, mag0=19.0, emag=1.0e-9, filt="G")

    model_bspl = PSPL_model.PSPLmodel(
        ev,
        parallax=["None", 0.0],
        double_source=["Circular", t0_true],
    )
    model_bspl.define_model_parameters()

    ZP = 27.615
    fs = mag_to_flux(ms, zp=ZP)
    ftotal = mag_to_flux(mtotal, zp=ZP)

    params_bspl = [
        t0_true,
        u0_true,
        tE_true,
        xi_para,
        xi_perp,
        omega,
        phi0,
        lambda_xi,
        q_mass,
        qflux,
        fs,
        ftotal,
    ]

    py_params = model_bspl.compute_pyLIMA_parameters(params_bspl)
    simulator.simulate_lightcurve(model_bspl, py_params)

    A_truth = model_bspl.model_magnification(ev.telescopes[0], py_params) / (
        1.0 + qflux
    )

    trajectories = model_bspl.sources_trajectory(
        model_bspl.event.telescopes[0],
        py_params,
        data_type="photometry",
    )

    source1_x, source1_y, source2_x, source2_y, dsep, dalpha = trajectories

    return {
        "t": np.asarray(t, dtype=float),
        "event": ev,
        "model": model_bspl,
        "py_params": py_params,
        "A": np.asarray(A_truth, dtype=float),
        "fs": fs,
        "ftotal": ftotal,
        "source1_x": _arr(source1_x),
        "source1_y": _arr(source1_y),
        "source2_x": _arr(source2_x),
        "source2_y": _arr(source2_y),
        "dseparation": _arr(dsep),
        "dalpha": _arr(dalpha),
        "omega": omega,
        "xiE_use": xiE_use,
        "xi_para": xi_para,
        "xi_perp": xi_perp,
        "kepler_info": kepler_info,
    }


# =============================================================================
# PSPL fit using the same trapezoidal objective as D
# =============================================================================

def objective_trapezoid(
    fit_params,
    target_model,
    A_true,
    t,
    fs_fixed,
    ftotal_fixed,
):
    fit_params = np.asarray(fit_params, dtype=float)
    full_params = np.concatenate([fit_params, [fs_fixed, ftotal_fixed]])
    py_params = target_model.compute_pyLIMA_parameters(full_params)
    telescope = target_model.event.telescopes[0]
    A_model = target_model.model_magnification(telescope, py_params)
    residual = A_true - A_model
    return float(np.trapezoid(residual**2, x=t))


def fit_pspl(ev, A_truth, t, t0_true, u0_true, tE_true, fs, ftotal):
    model_pspl = PSPL_model.PSPLmodel(
        ev,
        parallax=["None", 0.0],
        double_source=["None", 0.0],
    )
    model_pspl.define_model_parameters()

    x0 = np.array([t0_true, u0_true, tE_true], dtype=float)

    res = so.minimize(
        objective_trapezoid,
        x0=x0,
        args=(model_pspl, A_truth, t, fs, ftotal),
        method="Nelder-Mead",
        options={
            "maxiter": 200_000,
            "xatol": 1.0e-10,
            "fatol": 1.0e-12,
        },
    )

    if not res.success:
        raise RuntimeError(f"PSPL fit failed: {res.message}")

    best = np.asarray(res.x, dtype=float)
    best_full = np.concatenate([best, [fs, ftotal]])
    py_best = model_pspl.compute_pyLIMA_parameters(best_full)

    A_fit = model_pspl.model_magnification(ev.telescopes[0], py_best)
    traj = model_pspl.sources_trajectory(
        model_pspl.event.telescopes[0],
        py_best,
        data_type="photometry",
    )

    return {
        "A_fit": np.asarray(A_fit, dtype=float),
        "best_fit": best,
        "model": model_pspl,
        "py_params": py_best,
        "x": _arr(traj[0]),
        "y": _arr(traj[1]),
    }


# =============================================================================
# Metrics
# =============================================================================

def measure_residual_fwhm(t, residual):
    t = np.asarray(t, dtype=float)
    y = np.abs(np.asarray(residual, dtype=float))

    finite = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan

    safe = np.where(finite, y, -np.inf)
    i_max = int(np.argmax(safe))
    ymax = y[i_max]

    if not np.isfinite(ymax) or ymax <= 0:
        return 0.0

    half = 0.5 * ymax

    i_left = i_max
    while i_left > 0 and np.isfinite(y[i_left - 1]) and y[i_left - 1] >= half:
        i_left -= 1

    if i_left == 0:
        t_left = t[0]
    else:
        i0, i1 = i_left - 1, i_left
        if np.isclose(y[i1], y[i0]):
            t_left = t[i1]
        else:
            t_left = t[i0] + (half - y[i0]) * (t[i1] - t[i0]) / (y[i1] - y[i0])

    i_right = i_max
    while (
        i_right < len(t) - 1
        and np.isfinite(y[i_right + 1])
        and y[i_right + 1] >= half
    ):
        i_right += 1

    if i_right == len(t) - 1:
        t_right = t[-1]
    else:
        i0, i1 = i_right, i_right + 1
        if np.isclose(y[i1], y[i0]):
            t_right = t[i0]
        else:
            t_right = t[i0] + (half - y[i0]) * (t[i1] - t[i0]) / (y[i1] - y[i0])

    return float(t_right - t_left)


def intrinsic_distance(t, A_bspl, A_pspl):
    residual = A_bspl - A_pspl
    numerator = np.trapezoid(residual**2, x=t)
    denominator = np.trapezoid((A_bspl - 1.0) ** 2, x=t)

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan

    return float(np.sqrt(numerator / denominator))


# =============================================================================
# Full case
# =============================================================================

def compute_case(case, n_points=N_POINTS):

    P_days = float(case["P_days"])
    u0_true = float(case.get("u0_true", U0_TRUE))
    theta = float(case.get("theta", THETA))
    phi0 = float(case.get("phi0", PHI0))
    lambda_xi = float(case.get("lambda_xi", LAMBDA_XI))

    t = np.linspace(
        T0_TRUE - METRIC_WINDOW_TE * TE_TRUE,
        T0_TRUE + METRIC_WINDOW_TE * TE_TRUE,
        int(n_points),
    )

    state = build_binary_source_state(
        t=t,
        P_days=P_days,
        t0_true=T0_TRUE,
        u0_true=u0_true,
        tE_true=TE_TRUE,
        Mtot_source_Msun=MTOT_SOURCE,
        M_lens_Msun=M_LENS,
        Dl_kpc=DL_KPC,
        Ds_kpc=DS_KPC,
        xi_source=XI_SOURCE,
        theta=theta,
        phi0=phi0,
        lambda_xi=lambda_xi,
        q_mass=Q_MASS,
        qflux=Q_FLUX,
    )

    fit = fit_pspl(
        state["event"],
        state["A"],
        t,
        T0_TRUE,
        u0_true,
        TE_TRUE,
        state["fs"],
        state["ftotal"],
    )

    residual = state["A"] - fit["A_fit"]

    D = intrinsic_distance(
        t,
        state["A"],
        fit["A_fit"],
    )

    Rmax = float(
        np.nanmax(np.abs(residual))
    )

    t_dev = measure_residual_fwhm(
        t,
        residual,
    )

    best = fit["best_fit"]

    x1 = _arr(state["source1_x"])
    y1 = _arr(state["source1_y"])
    x2 = _arr(state["source2_x"])
    y2 = _arr(state["source2_y"])

    u1 = np.hypot(x1, y1)
    u2 = np.hypot(x2, y2)

    u1_min = float(np.nanmin(u1))
    u2_min = float(np.nanmin(u2))

    y_span_1 = float(np.nanmax(y1) - np.nanmin(y1))
    y_span_2 = float(np.nanmax(y2) - np.nanmin(y2))
    y_span = max(y_span_1, y_span_2)

    # ------------------------------------------------------------------------
    # 1. Significant residual features
    # ------------------------------------------------------------------------

    abs_residual = np.abs(residual)

    peak_distance = max(
        1,
        int(0.01 * len(t)),
    )

    residual_peaks, residual_props = find_peaks(
        abs_residual,
        prominence=max(
            1.0e-12,
            RESIDUAL_PEAK_PROMINENCE_FRAC * Rmax,
        ),
        distance=peak_distance,
    )

    if len(residual_peaks) > 0:

        residual_peak_heights = (
            abs_residual[residual_peaks]
        )

        significant = (
            residual_peak_heights
            >= RESIDUAL_PEAK_HEIGHT_FRAC * Rmax
        )

        significant_peaks = (
            residual_peaks[significant]
        )

        significant_heights = (
            residual_peak_heights[significant]
        )

    else:

        significant_peaks = np.array(
            [],
            dtype=int,
        )

        significant_heights = np.array(
            [],
            dtype=float,
        )

    n_residual_peaks = int(
        len(significant_peaks)
    )

    if n_residual_peaks >= 2:

        sorted_heights = np.sort(
            significant_heights
        )[::-1]

        second_peak_ratio = float(
            sorted_heights[1]
            / sorted_heights[0]
        )

        peak_time_span = float(
            np.max(t[significant_peaks])
            - np.min(t[significant_peaks])
        )

    else:

        second_peak_ratio = 0.0
        peak_time_span = 0.0


    # ------------------------------------------------------------------------
    # 2. Secondary structure directly in the BSPL light curve
    # ------------------------------------------------------------------------

    A_signal = np.asarray(
        state["A"],
        dtype=float,
    )

    event_amplitude = max(
        float(
            np.nanmax(A_signal)
            - np.nanmin(A_signal)
        ),
        1.0e-12,
    )

    lightcurve_peaks, lightcurve_props = find_peaks(
        A_signal,
        prominence=(
            LIGHTCURVE_PROMINENCE_FRAC
            * event_amplitude
        ),
        distance=peak_distance,
    )

    n_lightcurve_peaks = int(
        len(lightcurve_peaks)
    )

    if n_lightcurve_peaks >= 2:

        lc_prom = np.sort(
            lightcurve_props["prominences"]
        )[::-1]

        lightcurve_second_ratio = float(
            lc_prom[1] / lc_prom[0]
        )

    else:

        lightcurve_second_ratio = 0.0


    # ------------------------------------------------------------------------
    # 3. Difference between the luminous-source trajectory and best PSPL
    #
    # q_f = 0, therefore Source 1 is the relevant luminous trajectory.
    # ------------------------------------------------------------------------

    fit_x_arr = _arr(
        fit["x"]
    )

    fit_y_arr = _arr(
        fit["y"]
    )

    trajectory_offset = np.hypot(
        x1 - fit_x_arr,
        y1 - fit_y_arr,
    )

    trajectory_deviation_max = float(
        np.nanmax(
            trajectory_offset
        )
    )

    trajectory_deviation_rms = float(
        np.sqrt(
            np.nanmean(
                trajectory_offset**2
            )
        )
    )

    return {
        **state,

        "A_fit": fit["A_fit"],
        "fit_x": fit["x"],
        "fit_y": fit["y"],
        "best_fit": best,

        "residual": residual,
        "D": D,
        "Rmax": Rmax,
        "t_dev": t_dev,

        "P_days": P_days,
        "u0_true": u0_true,
        "theta": theta,
        "phi0": phi0,
        "lambda_xi": lambda_xi,

        "u1_min": u1_min,
        "u2_min": u2_min,
        "u_min": min(u1_min, u2_min),
        "y_span": y_span,

        "n_residual_peaks": n_residual_peaks,
        "second_peak_ratio": second_peak_ratio,
        "peak_time_span": peak_time_span,

        "n_lightcurve_peaks": n_lightcurve_peaks,
        "lightcurve_second_ratio": lightcurve_second_ratio,

        "trajectory_deviation_max": trajectory_deviation_max,
        "trajectory_deviation_rms": trajectory_deviation_rms,

        "significant_residual_peak_times": (
            np.asarray(t)[significant_peaks]
        ),

        "delta_t0": float(best[0] - T0_TRUE),
        "delta_u0": float(best[1] - u0_true),
        "delta_tE": float(best[2] - TE_TRUE),
    }


def distinguishable_score(out):
    """
    Ranking used ONLY to choose a pedagogically clear manuscript example.

    The scientific quantity D is still reported independently.  Here we favour
    morphology: multiple residual features and a visible orbital displacement
    of the luminous-source trajectory relative to the best PSPL trajectory.
    """

    D = max(
        float(out["D"]),
        1.0e-20,
    )

    Rmax = max(
        float(out["Rmax"]),
        1.0e-20,
    )

    trajectory = max(
        float(
            out["trajectory_deviation_max"]
        ),
        1.0e-20,
    )

    second_ratio = max(
        float(
            out["second_peak_ratio"]
        ),
        1.0e-3,
    )

    tspan = max(
        float(
            out["peak_time_span"]
        )
        / TE_TRUE,
        1.0e-3,
    )

    n_resid = int(
        out["n_residual_peaks"]
    )

    n_lc = int(
        out["n_lightcurve_peaks"]
    )

    lc_second = max(
        float(
            out["lightcurve_second_ratio"]
        ),
        1.0e-3,
    )

    # D and Rmax matter, but morphology now carries comparable weight.
    #
    # tspan rewards features separated in time instead of several numerical
    # maxima inside one narrow central spike.

    score = (
        1.5 * np.log10(D)
        + 0.6 * np.log10(Rmax)
        + 2.0 * np.log10(trajectory)
        + 1.2 * np.log10(second_ratio)
        + 0.8 * np.log10(tspan)
        + 0.35 * max(n_resid - 2, 0)
        + 0.60 * max(n_lc - 1, 0)
        + 0.50 * np.log10(lc_second)
    )

    return float(score)


def select_distinguishable_case():

    candidates = []

    n_total = (
        len(SEARCH_P_OVER_TE)
        * len(SEARCH_U0)
        * len(SEARCH_THETA)
        * len(SEARCH_PHI0)
        * len(SEARCH_LAMBDA)
    )

    print()
    print("=" * 80)
    print("SEARCHING FOR A MORPHOLOGICALLY CLEAR BSPL EXAMPLE")
    print(f"Candidates: {n_total}")
    print("=" * 80)

    i = 0

    for p_over_tE in SEARCH_P_OVER_TE:
        for u0 in SEARCH_U0:
            for theta in SEARCH_THETA:
                for phi0 in SEARCH_PHI0:
                    for lambda_xi in SEARCH_LAMBDA:

                        i += 1

                        case = {
                            "name": "Clearly distinguishable BSPL",
                            "basename": "example_1",
                            "expected": "distinguishable",
                            "auto_search": False,

                            "P_days": (
                                p_over_tE
                                * TE_TRUE
                            ),

                            "u0_true": u0,
                            "theta": theta,
                            "phi0": phi0,
                            "lambda_xi": lambda_xi,
                        }

                        try:

                            out = compute_case(
                                case,
                                n_points=N_SEARCH_POINTS,
                            )

                        except Exception as exc:

                            print(
                                f"[{i:03d}/{n_total}] "
                                f"failed: {exc}"
                            )

                            continue


                        if not (
                            np.isfinite(out["D"])
                            and
                            np.isfinite(out["Rmax"])
                            and
                            np.isfinite(
                                out[
                                    "trajectory_deviation_max"
                                ]
                            )
                        ):
                            continue


                        # -----------------------------------------------------
                        # Minimum intrinsic mismatch
                        # -----------------------------------------------------

                        if (
                            out["D"]
                            < D_DISTINGUISHABLE_MIN
                        ):
                            continue


                        # -----------------------------------------------------
                        # Require genuinely multiple residual structures
                        # -----------------------------------------------------

                        if (
                            out["n_residual_peaks"]
                            < MIN_RESIDUAL_PEAKS
                        ):
                            continue

                        if (
                            out["second_peak_ratio"]
                            < MIN_SECOND_PEAK_RATIO
                        ):
                            continue


                        # -----------------------------------------------------
                        # Require visible departure of Source 1 from PSPL
                        # -----------------------------------------------------

                        if (
                            out[
                                "trajectory_deviation_max"
                            ]
                            < MIN_TRAJECTORY_DEVIATION
                        ):
                            continue


                        score = (
                            distinguishable_score(
                                out
                            )
                        )

                        candidates.append(
                            (
                                score,
                                case,
                                out,
                            )
                        )

                        print(
                            f"[{i:03d}/{n_total}] "
                            f"P/tE={p_over_tE:4.2f} "
                            f"u0={u0:.3f} "
                            f"D={out['D']:.2e} "
                            f"Rmax={out['Rmax']:.2e} "
                            f"Nres={out['n_residual_peaks']} "
                            f"R2/R1={out['second_peak_ratio']:.2f} "
                            f"Nlc={out['n_lightcurve_peaks']} "
                            f"dtraj={out['trajectory_deviation_max']:.2e} "
                            f"Dtpeak/tE="
                            f"{out['peak_time_span']/TE_TRUE:.2f}"
                        )


    if not candidates:

        raise RuntimeError(
            "No configuration satisfied the morphological criteria. "
            "If needed, relax MIN_SECOND_PEAK_RATIO or "
            "MIN_TRAJECTORY_DEVIATION."
        )


    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    score, case, search_out = (
        candidates[0]
    )


    print()
    print("=" * 80)
    print("SELECTED MORPHOLOGICAL CONFIGURATION")
    print("=" * 80)

    print(
        f"P          = "
        f"{case['P_days']:.6g} d"
    )

    print(
        f"P/tE       = "
        f"{case['P_days']/TE_TRUE:.6g}"
    )

    print(
        f"u0         = "
        f"{case['u0_true']:.6g}"
    )

    print(
        f"theta/pi   = "
        f"{case['theta']/np.pi:.6g}"
    )

    print(
        f"phi0/pi    = "
        f"{case['phi0']/np.pi:.6g}"
    )

    print(
        f"lambda/pi  = "
        f"{case['lambda_xi']/np.pi:.6g}"
    )

    print(
        f"search D   = "
        f"{search_out['D']:.6e}"
    )

    print(
        f"search Rmax= "
        f"{search_out['Rmax']:.6e}"
    )

    print(
        f"N residual peaks = "
        f"{search_out['n_residual_peaks']}"
    )

    print(
        f"second/main peak  = "
        f"{search_out['second_peak_ratio']:.6f}"
    )

    print(
        f"peak time span    = "
        f"{search_out['peak_time_span']:.6f} d"
    )

    print(
        f"N lightcurve peaks= "
        f"{search_out['n_lightcurve_peaks']}"
    )

    print(
        f"LC second/main    = "
        f"{search_out['lightcurve_second_ratio']:.6f}"
    )

    print(
        f"trajectory max dev= "
        f"{search_out['trajectory_deviation_max']:.6e}"
    )

    print(
        f"trajectory rms dev= "
        f"{search_out['trajectory_deviation_rms']:.6e}"
    )

    print(
        f"score             = "
        f"{score:.6f}"
    )


    print()
    print(
        f"Recomputing with N={N_POINTS}..."
    )

    final_out = compute_case(
        case,
        n_points=N_POINTS,
    )


    print(
        f"final D      = "
        f"{final_out['D']:.6e}"
    )

    print(
        f"final Rmax   = "
        f"{final_out['Rmax']:.6e}"
    )

    print(
        f"final Nres   = "
        f"{final_out['n_residual_peaks']}"
    )

    print(
        f"final R2/R1  = "
        f"{final_out['second_peak_ratio']:.6f}"
    )

    print(
        f"final Nlc    = "
        f"{final_out['n_lightcurve_peaks']}"
    )

    print(
        f"final dtraj  = "
        f"{final_out['trajectory_deviation_max']:.6e}"
    )

    return (
        case,
        final_out,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def _time_offset_label(x):
    if np.isclose(x, 0.0):
        return r"$t_0$"

    sign = "+" if x > 0 else "-"
    f = Fraction(float(abs(x))).limit_denominator(12)
    num, den = f.numerator, f.denominator

    if den == 1:
        term = r"t_E" if num == 1 else rf"{num}t_E"
    else:
        term = rf"\frac{{1}}{{{den}}}t_E" if num == 1 else rf"\frac{{{num}}}{{{den}}}t_E"

    return rf"$t_0{sign}{term}$"


def set_time_ticks(ax, window_k=METRIC_WINDOW_TE, max_labels=7):
    nice_steps = np.array([0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    chosen = nice_steps[-1]

    for step in nice_steps:
        n_ticks = 2 * int(np.floor(window_k / step)) + 1
        if n_ticks <= max_labels:
            chosen = step
            break

    n_side = int(np.floor(window_k / chosen + 1.0e-10))
    offsets = np.arange(-n_side, n_side + 1) * chosen
    ticks = T0_TRUE + offsets * TE_TRUE
    labels = [_time_offset_label(x) for x in offsets]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def add_direction_arrows(ax, x, y, color, n_arrows=2, scale=0.045, lw=0.9):
    x = _arr(x)
    y = _arr(y)
    if len(x) < 3:
        return

    idxs = np.linspace(1, len(x) - 2, n_arrows, dtype=int)
    for idx in idxs:
        dx = x[idx + 1] - x[idx - 1]
        dy = y[idx + 1] - y[idx - 1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue

        ax.annotate(
            "",
            xy=(x[idx] + scale * dx / norm, y[idx] + scale * dy / norm),
            xytext=(x[idx], y[idx]),
            arrowprops={
                "arrowstyle": "->",
                "mutation_scale": 8,
                "color": color,
                "lw": lw,
            },
            zorder=12,
        )


def plot_case(case, out):
    set_paper_style()

    t = out["t"]
    A = out["A"]
    A_fit = out["A_fit"]
    residual = out["residual"]
    kinfo = out["kepler_info"]

    fig = plt.figure(figsize=(8.8, 6.4))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.2, 1.0],
        hspace=0.05,
    )

    axA = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[1, 0], sharex=axA)

    # Reserve the upper part of the canvas for the parameters.  They are
    # outside the plotting axes, rather than in a column to the right.
    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.12,
        top=0.74,
    )

    # -------------------------------------------------------------------------
    # Parameter/header block above the axes
    # -------------------------------------------------------------------------
    fig.text(
        0.5,
        0.965,
        case["name"],
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    line1 = (
        rf"$P={case['P_days']:.0f}\,\mathrm{{d}}$"
        rf"  $\; P/t_E={case['P_days']/TE_TRUE:.1f}$"
        rf"  $\; a_{{\rm rel}}={kinfo['a_rel_AU']:.3g}\,\mathrm{{AU}}$"
        rf"  $\; t_E={TE_TRUE:.0f}\,\mathrm{{d}}$"
        rf"  $\; u_0={out['u0_true']:g}$"
    )

    line2 = (
        rf"$M_1={M1_SOURCE:g}\,M_\odot$"
        rf"  $\; M_2={M2_SOURCE:g}\,M_\odot$"
        rf"  $\; q_M={Q_MASS:g}$"
        rf"  $\; q_f={Q_FLUX:g}$"
        rf"  $\; \hat r_E={kinfo['RE_source_AU']:.3g}\,\mathrm{{AU}}$"
        rf"  $\; \xi_{{E,1}}={out['xiE_use']:.3g}$"
    )

    line_geom = (
        rf"$\theta={out['theta']/np.pi:.2g}\pi$"
        rf"  $\; \phi_0={out['phi0']/np.pi:.2g}\pi$"
        rf"  $\; \lambda_\xi={out['lambda_xi']/np.pi:.2g}\pi$"
        rf"  $\; u_{{1,\min}}={out['u1_min']:.3g}$"
        rf"  $\; u_{{2,\min}}={out['u2_min']:.3g}$"
    )

    line3 = (
        rf"$D_{{\rm BSPL-PSPL}}={out['D']:.2e}$"
        rf"  $\; R_{{\rm max}}={out['Rmax']:.2e}$"
        rf"  $\; t_{{\rm dev}}={out['t_dev']:.3g}\,\mathrm{{d}}$"
        rf"  $\; (\Delta t_0,\Delta u_0,\Delta t_E)="
        rf"({out['delta_t0']:.2e},{out['delta_u0']:.2e},{out['delta_tE']:.2e})$"
    )

    fig.text(0.5, 0.925, line1, ha="center", va="top", fontsize=10.5)
    fig.text(0.5, 0.892, line2, ha="center", va="top", fontsize=10.5)
    fig.text(0.5, 0.859, line_geom, ha="center", va="top", fontsize=9.8)
    fig.text(0.5, 0.826, line3, ha="center", va="top", fontsize=10.0)

    # -------------------------------------------------------------------------
    # Light curve
    # -------------------------------------------------------------------------
    axA.plot(
        t,
        A,
        color="C0",
        lw=2.0,
        label="BSPL truth",
        zorder=3,
    )
    axA.plot(
        t,
        A_fit,
        color="red",
        lw=1.0,
        label="best PSPL",
        zorder=6,
    )

    axA.axvline(T0_TRUE, color="0.35", ls=":", lw=0.8, alpha=0.7)
    axR.axvline(T0_TRUE, color="0.35", ls=":", lw=0.8, alpha=0.7)

    axA.set_ylabel(r"$A$")
    axA.legend(loc="upper left", ncol=2, frameon=False)
    axA.grid(which="major", alpha=0.10, linewidth=0.5)

    # -------------------------------------------------------------------------
    # Residuals: manuscript sign convention Delta A = A_BSPL - A_PSPL
    # -------------------------------------------------------------------------
    axR.plot(t, residual, color="C0", lw=1.2)
    axR.axhline(0.0, color="0.3", ls="--", lw=0.7, alpha=0.7)
    axR.set_ylabel(r"$\Delta A$")
    axR.set_xlabel(r"$t$")
    axR.grid(which="major", alpha=0.10, linewidth=0.5)

    axA.yaxis.set_minor_locator(AutoMinorLocator())
    axR.yaxis.set_minor_locator(AutoMinorLocator())
    axA.tick_params(labelbottom=False)
    set_time_ticks(axR)

    # -------------------------------------------------------------------------
    # Source trajectories inset
    # -------------------------------------------------------------------------
    axT = axA.inset_axes([0.57, 0.25, 0.40, 0.69])

    phase = np.linspace(0.0, 2.0 * np.pi, 400)
    axT.plot(
        np.cos(phase),
        np.sin(phase),
        color="0.3",
        ls=":",
        lw=0.9,
        alpha=0.8,
        label=r"$\theta_E$",
        zorder=1,
    )

    axT.plot(
        out["source1_x"],
        out["source1_y"],
        color="C0",
        ls="-",
        lw=2.0,
        label="Source 1",
        zorder=6,
    )
    axT.plot(
        out["source2_x"],
        out["source2_y"],
        color="C0",
        ls="--",
        lw=2.0,
        label="Source 2",
        zorder=6,
    )
    axT.plot(
        out["fit_x"],
        out["fit_y"],
        color="red",
        ls="-",
        lw=1.0,
        zorder=10,
    )

    add_direction_arrows(axT, out["source1_x"], out["source1_y"], "C0")
    add_direction_arrows(axT, out["source2_x"], out["source2_y"], "C0")
    add_direction_arrows(
        axT,
        out["fit_x"],
        out["fit_y"],
        "red",
        n_arrows=1,
        lw=0.7,
    )

    axT.scatter([0], [0], marker="+", s=55, linewidth=1.2, color="k", zorder=20, label="Lens")
    axT.axhline(0.0, color="0.4", lw=0.45, alpha=0.20)
    axT.axvline(0.0, color="0.4", lw=0.45, alpha=0.20)
    # ---------------------------------------------------------------------
    # Trajectory zoom
    #
    # For the distinguishable example we zoom onto the lens-encounter region
    # instead of forcing the full Einstein ring into the inset.  Otherwise
    # orbital deviations of order 1e-2 are visually compressed.
    # ---------------------------------------------------------------------

    if case["expected"] == "distinguishable":

        tt = np.asarray(
            out["t"],
            dtype=float,
        )

        local = (
            np.abs(tt - T0_TRUE)
            <= 0.35 * TE_TRUE
        )

        xx = np.concatenate([
            np.asarray(out["source1_x"])[local],
            np.asarray(out["source2_x"])[local],
            np.asarray(out["fit_x"])[local],
            np.array([0.0]),
        ])

        yy = np.concatenate([
            np.asarray(out["source1_y"])[local],
            np.asarray(out["source2_y"])[local],
            np.asarray(out["fit_y"])[local],
            np.array([0.0]),
        ])

        trajectory_limit = float(
            np.nanmax(
                np.abs(
                    np.concatenate([
                        xx,
                        yy,
                    ])
                )
            )
        )

        trajectory_limit = float(
            np.clip(
                1.20 * trajectory_limit,
                0.05,
                0.30,
            )
        )

    else:

        trajectory_limit = (
            TRAJECTORY_LIMIT
        )

    axT.set_xlim(
        -trajectory_limit,
        trajectory_limit,
    )

    axT.set_ylim(
        -trajectory_limit,
        trajectory_limit,
    )
    axT.set_aspect("equal", adjustable="box")
    axT.set_xlabel(r"$u_x$", fontsize=10, labelpad=1)
    axT.set_ylabel(r"$u_y$", fontsize=10, labelpad=1)
    axT.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
    axT.xaxis.set_minor_locator(AutoMinorLocator())
    axT.yaxis.set_minor_locator(AutoMinorLocator())
    axT.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=2.0)

    for ax in (axA, axR, axT):
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axA.set_xlim(
        T0_TRUE - METRIC_WINDOW_TE * TE_TRUE,
        T0_TRUE + METRIC_WINDOW_TE * TE_TRUE,
    )

    png = OUTDIR / f"{case['basename']}.png"
    pdf = OUTDIR / f"{case['basename']}.pdf"

    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("FIXED BSPL--PSPL LIGHT-CURVE EXAMPLES")
    print("=" * 80)
    print(f"Output: {OUTDIR}")
    print()

    for original_case in CASES:

        if original_case.get("auto_search", False):
            case, out = select_distinguishable_case()

        else:
            case = dict(original_case)
            out = compute_case(case)

        print()
        print("-" * 80)
        print(case["name"])

        kinfo = out["kepler_info"]

        print(f"P           = {case['P_days']:.6f} d")
        print(f"P/tE        = {case['P_days']/TE_TRUE:.6f}")
        print(f"u0          = {out['u0_true']:.6e}")
        print(f"theta/pi    = {out['theta']/np.pi:.6f}")
        print(f"phi0/pi     = {out['phi0']/np.pi:.6f}")
        print(f"lambda/pi   = {out['lambda_xi']/np.pi:.6f}")
        print(f"a_rel       = {kinfo['a_rel_AU']:.6f} AU")
        print(f"R_E,source  = {kinfo['RE_source_AU']:.6f} AU")
        print(f"xi_E,1      = {out['xiE_use']:.6e}")
        print(f"u1_min      = {out['u1_min']:.6e}")
        print(f"u2_min      = {out['u2_min']:.6e}")
        print(f"trajectory dy = {out['y_span']:.6e}")
        print(
            f"trajectory dev = "
            f"{out['trajectory_deviation_max']:.6e}"
        )
        print(
            f"N residual peaks = "
            f"{out['n_residual_peaks']}"
        )
        print(
            f"second/main peak = "
            f"{out['second_peak_ratio']:.6f}"
        )
        print(
            f"N lightcurve peaks = "
            f"{out['n_lightcurve_peaks']}"
        )
        print(f"D           = {out['D']:.6e}")
        print(f"Rmax        = {out['Rmax']:.6e}")
        print(f"t_dev       = {out['t_dev']:.6e} d")

        print(
            "best PSPL   = "
            f"[{out['best_fit'][0]:.8f}, "
            f"{out['best_fit'][1]:.8f}, "
            f"{out['best_fit'][2]:.8f}]"
        )

        if (
            case["expected"] == "distinguishable"
            and
            out["D"] < D_REFERENCE
        ):
            print(
                f"WARNING: expected D >= {D_REFERENCE:g}, "
                f"but obtained D={out['D']:.3e}."
            )

        if (
            case["expected"] == "confused"
            and
            out["D"] > D_REFERENCE
        ):
            print(
                f"WARNING: expected D <= {D_REFERENCE:g}, "
                f"but obtained D={out['D']:.3e}."
            )

        png, pdf = plot_case(case, out)

        print(f"PNG: {png}")
        print(f"PDF: {pdf}")
        print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
