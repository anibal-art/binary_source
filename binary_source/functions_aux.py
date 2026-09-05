import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os, sys
current_path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_path, os.pardir))
# diagnostic print disabled for production runs
sys.path.append(parent_directory)

from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
# from pyLIMA.simulations import simulator
# from pyLIMA.fits import TRF_fit
from astropy import units as u
# from astropy import constants as C
# from pyLIMA.xallarap.xallarap import xallarap_shifts, compute_xallarap_curvature

import scipy.optimize as so
from scipy.interpolate import interp1d

mag = np.array([
    12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
    15.0, 15.5, 16.0, 16.5, 17.0, 17.5,
    18.0, 18.5, 19.0, 19.5, 20.0, 20.5,
    21.0, 21.5, 22.0, 22.5, 23.0, 23.5,
    24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0
])

sigma_W149 = np.array([
    0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
    0.0010475, 0.0011229, 0.0012038, 0.0013207,
    0.0015178, 0.0017852, 0.002139, 0.0025867,
    0.0032066, 0.0039261, 0.0050352, 0.0063881,
    0.0082817, 0.01087, 0.014603, 0.019616,
    0.027602, 0.039749, 0.058585, 0.088371,
    0.13351, 0.20964, 0.32165, 0.51373, 1.0
])

# Interpolation in log10(sigma)
sigma_interp_log = interp1d(
    mag,
    np.log10(sigma_W149),
    kind="linear",
    bounds_error=True
)

def mag(zp, Flux):
    return zp - 2.5 * np.log10(np.abs(Flux))

def flux_to_mag(flux, zp=27.615):
    flux = np.asarray(flux)
    return zp - 2.5 * np.log10(flux)

def sigma_W149_func(W149):
    """
    Return sigma_W149 [mag / 15 min] for a W149 magnitude.

    Valid for 12 <= W149 <= 27.
    """
    W149 = np.asarray(W149)

    log_sigma = sigma_interp_log(W149)
    sigma = 10**log_sigma

    return sigma



# ============================================================
# Roman F146 photometric precision
# ============================================================
#
# Current GBTDS prescription:
#
#   primary bandpass       : F146
#   exposure time          : ~66 s
#   reference requirement  : S/N ~ 100 at F146_AB = 21.2
#
# We use a simple source+effective-background noise model.
# The effective background is calibrated so that the model
# reproduces S/N=100 at F146=21.2.
#
# This is intentionally a survey-level precision prescription,
# not a full detector/crowding simulation.
# ============================================================

F146_ZP_AB = 27.648
F146_EXPOSURE_S = 66.0

F146_REFERENCE_MAG = 21.2
F146_REFERENCE_SNR = 100.0

F146_BRIGHT_FLOOR_MAG = 1.0e-3


def sigma_F146_func(F146):
    """
    Approximate single-epoch Roman/F146 photometric precision.

    Parameters
    ----------
    F146 : array_like
        F146 AB magnitude.

    Returns
    -------
    sigma_mag : ndarray
        1-sigma magnitude uncertainty per F146 epoch.

    Notes
    -----
    The model is normalized to S/N = 100 at F146_AB = 21.2
    for the current GBTDS F146 exposure prescription.

    We model

        variance = N_source + B_eff,

    where B_eff is calibrated at the reference magnitude.
    A 1 mmag bright-source floor is retained.
    """

    F146 = np.asarray(
        F146,
        dtype=float,
    )

    if np.any(~np.isfinite(F146)):
        raise ValueError(
            "Non-finite F146 magnitude."
        )

    # Source electrons in one exposure.
    n_source = (
        10.0
        ** (
            -0.4
            * (
                F146
                - F146_ZP_AB
            )
        )
        * F146_EXPOSURE_S
    )

    # Reference source electrons.
    n_ref = (
        10.0
        ** (
            -0.4
            * (
                F146_REFERENCE_MAG
                - F146_ZP_AB
            )
        )
        * F146_EXPOSURE_S
    )

    # Effective additional variance required to obtain
    # S/N = 100 at F146 = 21.2.
    b_eff = (
        (
            n_ref
            / F146_REFERENCE_SNR
        ) ** 2
        - n_ref
    )

    b_eff = max(
        float(b_eff),
        0.0,
    )

    sigma_e = np.sqrt(
        n_source
        + b_eff
    )

    snr = (
        n_source
        / sigma_e
    )

    sigma_mag = (
        2.5
        / np.log(10.0)
        / snr
    )

    sigma_mag = np.maximum(
        sigma_mag,
        F146_BRIGHT_FLOOR_MAG,
    )

    return np.asarray(
        sigma_mag,
        dtype=float,
    )



def mag_to_flux(mag, zp=27.615):
    return 10**(-0.4 * (mag - zp))

def sigma_flux_from_sigma_mag(mag, sigma_mag, zp=0.0):
    F = mag_to_flux(mag, zp)
    sigma_F = 0.921034 * F * sigma_mag
    return sigma_F




def sigma_flux_from_flux(flux, zp=27.615):
    """
    Return the absolute flux uncertainty using the sigma_W149(mag) relation.
    
    Requires sigma_W149_func(mag) to be defined.
    """
    flux = np.asarray(flux)

    mag = flux_to_mag(flux, zp=zp)
    sigma_mag = sigma_W149_func(mag)

    sigma_flux = (np.log(10) / 2.5) * flux * sigma_mag

    return sigma_flux

def orbital_period_kepler(a_au, M_tot_Msun):
    """
    Compute the orbital period of a binary system using Kepler's third law
    in astronomical units.

    Parameters
    ----------
    a_au : float or array-like
        Semimajor axis in astronomical units (AU).
    M_tot_Msun : float or array-like
        Total mass of the system in solar masses (M_sun).

    Returns
    -------
    P_yr : float or ndarray
        Orbital period in years.
    """
    a_au = np.asarray(a_au, dtype=float)
    M_tot_Msun = np.asarray(M_tot_Msun, dtype=float)
    # diagnostic print disabled for production runs
    # diagnostic print disabled for production runs
    return np.sqrt(a_au**3 / M_tot_Msun)*365.25*(1/u.day)

def build_case(case_name, DS, DL, rEhat, v_perp, a, M1, M2,
               t0=50, u0=0.1, xi_phase=0, xi_inclination=np.pi/2, flux_ratio=0.0):
    """
    Build a dictionary with the parameters of a xallarap configuration.
    """
    q_xi = (M1 / M2).decompose().value
    P = orbital_period_kepler(a, M1 + M2)

    # tE = (rEhat * DL / DS) / v_perp
    tE = (rEhat) / v_perp
    return {
        "case": case_name,
        "DS_kpc": DS.to(u.kpc).value,
        "DL_kpc": DL.to(u.kpc).value,
        "rEhat_AU": rEhat.to(u.AU).value,
        "v_perp_kms": v_perp.to(u.km/u.s).value,
        "a_AU": a.to(u.AU).value,
        "M1_Msun": M1.to(u.M_sun).value,
        "M2_Msun": M2.to(u.M_sun).value,
        "xi_mass_ratio": q_xi,
        "tE": tE.to(u.day).value,
        "t0": t0,
        "u0": u0,
        "xiE": (a / rEhat).decompose().value,
        "omega_xi_1_per_day": (2*np.pi / P).value,
        "xi_phase": xi_phase,
        "xi_inclination": xi_inclination,
        "flux_ratio": flux_ratio,
        "P": P.value,
    }

DS = 8 * u.kpc
DL = 4 * u.kpc
v_perp = 50 * u.km / u.s
a = 2 * u.AU

rows = []

# =========================
# rEhat = 5 AU
# =========================

rEhat = 5 * u.AU

# Case 1: face-on, P > tE
rows.append(build_case(
    "case1", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun
))

# Case 2: face-on, P < tE
rows.append(build_case(
    "case2", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun
))

# Case 3a: edge-on, low mass ratio
rows.append(build_case(
    "case3a", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun,
    xi_inclination=np.pi/2
))

# Case 3b: edge-on, high mass ratio
rows.append(build_case(
    "case3b", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun,
    xi_inclination=np.pi/2
))

# =========================
# rEhat = 2 AU  (Case 4)
# =========================
rEhat = 2 * u.AU

rows.append(build_case(
    "case4-1", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun,
    xi_inclination=0
))

rows.append(build_case(
    "case4-2", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun,
    xi_inclination=0
))

rows.append(build_case(
    "case4-3a", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun,
    xi_inclination=np.pi/2
))

rows.append(build_case(
    "case4-3b", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun,
    xi_inclination=np.pi/2
))

df_cases = pd.DataFrame(rows).set_index("case")



def chi2_theoretical(fit_params, your_model, use_magnification=False,
                     fs_fixed=1.0, ftotal_fixed=1.0):
    """
    SSE (sin pesos): compara data vs modelo.
    fit_params = [t0,u0,tE]
    Fija flujos: fsource=fs_fixed, ftotal=ftotal_fixed (porque blend_flux_parameter='ftotal' default).
    """
    fit_params = np.asarray(fit_params, dtype=float)
    full_params = np.concatenate([fit_params, [fs_fixed, ftotal_fixed]])

    py_params = your_model.compute_pyLIMA_parameters(full_params)

    sse = 0.0
    for telescope in your_model.event.telescopes:
        if telescope.lightcurve is None:
            continue

        data = telescope.lightcurve['flux'].value

        if use_magnification:
            model_pred = your_model.model_magnification(telescope, py_params)
        else:
            model_pred = your_model.compute_the_microlensing_model(
                telescope, py_params
            )['photometry']

        resid = data - model_pred
        sse += np.sum(resid**2)

    return float(sse)



def build_sim_event(time, mag0=19.0, emag=0.01, filt="G"):
    """
    Crea un Event con un Telescope con columnas time/mag/err_mag.
    simulator.simulate_lightcurve(...) will populate flux/err_flux if used.
    """
    ev = event.Event()
    ev.name = "Simulated"
    ev.ra = 170
    ev.dec = -70

    lightcurve_sim = np.c_[time, np.full_like(time, mag0), np.full_like(time, emag)]
    tel = telescopes.Telescope(
        name="Simulation",
        camera_filter=filt,
        lightcurve=lightcurve_sim.astype(float),
        lightcurve_names=["time", "mag", "err_mag"],
        lightcurve_units=["JD", "mag", "mag"],
        location="Earth",
    )
    ev.telescopes.append(tel)
    return ev




def a_from_P_kepler_days(P_days, Mtot_Msun):
    """
    Kepler: a^3 = Mtot * P^2, with P in years, a in AU, and Mtot in Msun.
    Return a in AU as a float.
    """
    P_yr = np.asarray(P_days, dtype=float) / 365.25
    return (Mtot_Msun * P_yr**2)**(1.0/3.0)
