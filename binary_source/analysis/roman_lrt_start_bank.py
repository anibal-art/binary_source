"""
Data-derived fit references and start banks for the Roman BSPL LRT.

Canonical start policy for binary_source/analysis/roman_bspl_lrt_catalog_runner.py.
Only the observed W149 light curve (time, mag, err_mag) and the best observed
PSPL fit are used. Catalog truth is never read here.

The same rules are applied to H0 and H1 datasets, so the calibrated null
distribution refers to exactly the statistic used for the power estimate.
"""

from __future__ import annotations

import numpy as np


def u_of_magnification(A):
    """Inverse of the PSPL magnification A(u) = (u^2+2)/(u sqrt(u^2+4))."""
    A = max(float(A), 1.0 + 1.0e-9)
    return float(np.sqrt(2.0 * A / np.sqrt(A * A - 1.0) - 2.0))


def clip_to_bounds(value, bounds, rel_margin=1.0e-6):
    """Move a start strictly inside [lo, hi] (TRF needs interior starts)."""
    lo, hi = float(bounds[0]), float(bounds[1])
    eps = rel_margin * (hi - lo)
    return float(min(max(float(value), lo + eps), hi - eps))


def light_curve_features(lc, bin_days=0.25, min_points_per_bin=3,
                         tE_fallback=30.0):
    """
    Peak time, peak magnification, and first-guess u0 and tE from W149 data.

    Assumes no blending (A_peak is a lower limit, so u0 is an upper limit);
    the PSPL start grid spans factors around these guesses.
    """
    lc = np.asarray(lc, dtype=float)
    t, mag, err = lc[:, 0], lc[:, 1], lc[:, 2]
    m_base = float(np.median(mag))
    A = 10.0 ** (-0.4 * (mag - m_base))
    w = 1.0 / (0.4 * np.log(10.0) * A * err) ** 2

    edges = np.arange(t.min(), t.max() + bin_days, bin_days)
    idx = np.digitize(t, edges) - 1
    n = np.bincount(idx, minlength=len(edges))
    sw = np.bincount(idx, weights=w, minlength=len(edges))
    swa = np.bincount(idx, weights=w * A, minlength=len(edges))
    ok = n >= min_points_per_bin
    tb = (edges + 0.5 * bin_days)[ok]
    Ab = swa[ok] / sw[ok]

    k = int(np.argmax(Ab))
    t_peak, A_peak = float(tb[k]), float(Ab[k])
    u0 = u_of_magnification(A_peak)

    # Half-maximum duration from the contiguous binned run around the peak.
    A_half = 0.5 * (A_peak + 1.0)
    lo = k
    while lo > 0 and Ab[lo - 1] > A_half:
        lo -= 1
    hi = k
    while hi < len(Ab) - 1 and Ab[hi + 1] > A_half:
        hi += 1
    dt_half = 0.5 * float(tb[hi] - tb[lo]) + 0.5 * bin_days
    du2 = u_of_magnification(A_half) ** 2 - u0 ** 2
    if A_peak > 1.05 and du2 > 0 and hi > lo:
        tE = dt_half / np.sqrt(du2)
        tE_source = "half_maximum"
    else:
        tE = float(tE_fallback)
        tE_source = "fallback"

    return {
        "t_peak": t_peak,
        "A_peak_unblended": A_peak,
        "u0_guess": float(u0),
        "tE_guess": float(tE),
        "tE_source": tE_source,
        "m_base": m_base,
        "bin_days": float(bin_days),
    }


def fit_reference_from_features(features):
    """Common reference (t0, u0, tE) for every fit family of one dataset."""
    return {
        "t0": float(features["t_peak"]),
        "u0": float(features["u0_guess"]),
        "tE": float(features["tE_guess"]),
    }


def pspl_starts(features, rule, bounds):
    """u0 x tE grid centred on the light-curve guesses; t0 at the peak."""
    starts = []
    for fu in rule["u0_factors"]:
        for ft in rule["tE_factors"]:
            starts.append({
                "label": f"pspl_u0x{fu:g}_tEx{ft:g}",
                "initial_guess": {
                    "t0": clip_to_bounds(features["t_peak"], bounds["t0"]),
                    "u0": clip_to_bounds(fu * features["u0_guess"], bounds["u0"]),
                    "tE": clip_to_bounds(ft * features["tE_guess"], bounds["tE"]),
                },
            })
    return starts


def bspl_starts(pspl_best, rule, bounds):
    """
    BSPL start bank built on the best observed PSPL (t0, u0, tE).

    periods  : log grid [period_min_days, period_max_days], n_periods nodes
    phases   : rule["phases"]
    amplitude: xi_rel chosen so that the primary displacement is delta_u for
               P <= 2 pi tE, growing as (P / 2 pi tE)^2 for longer periods
               (the acceleration term then sets the signal), capped.
    nested   : one start at xi_rel = nested_xi from the PSPL optimum, so that
               chi2_BSPL <= chi2_PSPL up to optimizer failure.
    """
    t0, u0, tE = (float(v) for v in pspl_best[:3])
    q = float(rule["xi_mass_ratio"])
    frac = q / (1.0 + q)
    common = {
        "t0": clip_to_bounds(t0, bounds["t0"]),
        "u0": clip_to_bounds(u0, bounds["u0"]),
        "tE": clip_to_bounds(tE, bounds["tE"]),
        "xi_perp": 0.0,
        "xi_inclination": clip_to_bounds(rule["xi_inclination"],
                                         bounds["xi_inclination"]),
        "xi_mass_ratio": clip_to_bounds(q, bounds["xi_mass_ratio"]),
        "q_flux_W149": clip_to_bounds(rule["q_flux"], bounds["q_flux_W149"]),
    }

    periods = np.geomspace(rule["period_min_days"], rule["period_max_days"],
                           int(rule["n_periods"]))
    starts = []
    if rule.get("nested_start", True):
        g = dict(common)
        g.update({
            "xi_para": float(rule["nested_xi"]),
            "xi_angular_velocity": clip_to_bounds(
                2.0 * np.pi / max(tE, rule["period_min_days"]),
                bounds["xi_angular_velocity"]),
            "xi_phase": clip_to_bounds(rule["phases"][0], bounds["xi_phase"]),
        })
        starts.append({"label": "bspl_nested", "initial_guess": g})

    for P in periods:
        growth = max(1.0, (P / (2.0 * np.pi * tE)) ** 2)
        for du in rule["delta_u"]:
            xi = min(du * growth / frac, rule["xi_cap"])
            for ph in rule["phases"]:
                g = dict(common)
                g.update({
                    "xi_para": clip_to_bounds(xi, bounds["xi_para"]),
                    "xi_angular_velocity": clip_to_bounds(
                        2.0 * np.pi / P, bounds["xi_angular_velocity"]),
                    "xi_phase": clip_to_bounds(ph, bounds["xi_phase"]),
                })
                starts.append({
                    "label": f"bspl_P{P:.4g}_du{du:g}_ph{ph:.3g}",
                    "initial_guess": g,
                })
    return starts
