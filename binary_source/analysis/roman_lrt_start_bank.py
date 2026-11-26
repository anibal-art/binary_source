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


# Start pairs with |B - eta| below this are skipped: the linear amplitude
# divided by B - eta would put the start at the xi bounds.
MIN_CENTROID_FACTOR = 0.05
# q_flux below this is a dark companion (the dark policies hold q_flux in
# [0, 1e-8] through bounds); the conversions then reduce exactly to the
# dark-companion ones.
DARK_Q_FLUX = 1e-6


def start_pairs(rule):
    """
    (q_M, q_f) start pairs of the BSPL fit.  rule["start_pairs"] (luminous
    companion policies) or, by default, the single pair
    (rule["xi_mass_ratio"], rule["q_flux"]) of the dark-companion policies.
    """
    if "start_pairs" in rule:
        return [(float(qm), float(qf)) for qm, qf in rule["start_pairs"]]
    return [(float(rule["xi_mass_ratio"]), float(rule["q_flux"]))]


def centroid_factor(mass_ratio, q_flux=0.0):
    """
    B - eta: displacement of the light centroid in units of the source-1
    displacement amplitude xi (B = q_M/(1+q_M), eta = q_f/(1+q_f)).  The
    linearized model sees the centroid, so the source-1 amplitude is the
    linear one divided by B - eta (negative: opposite direction).
    """
    q, qf = float(mass_ratio), float(q_flux)
    if qf < DARK_Q_FLUX:                      # dark policies: q_flux ~ 1e-8 by bounds
        return q / (1.0 + q)
    return q / (1.0 + q) - qf / (1.0 + qf)


def second_order_candidates(names, pspl_best, omega, rule, bounds):
    """
    BSPL starts that do not use the linearized periodogram, for companions on
    the cancellation line q_f ~ q_M (Section 5): there the light centroid does
    not move, the first-order (linear) term vanishes, and the light curve
    depends on the separation term eta(1-eta) xi_rel^2.  rule["second_order_starts"]:

        pairs        : (q_M, q_f) start pairs, e.g. [[1, 1]]
        xi_over_u0   : xi_rel = factor * |u0 of the best PSPL| (the separation
                       term matters when xi_rel ~ u0)
        n_orientation: angles of (xi_para, xi_perp) in [0, pi)
        n_phase      : orbital phases in [0, 2 pi)
        inclinations : xi_inclination values

    t0, u0, tE from the best PSPL.  Returns [] without the key (dark policies).
    Every candidate has family "second_order"; the omega profile fits the best
    one of each family (lrt_omega_profile).
    """
    so = rule.get("second_order_starts")
    if not so:
        return []
    i_t0, i_u0, i_tE = names.index("t0"), names.index("u0"), names.index("tE")
    t0, u0, tE = (float(pspl_best[i]) for i in (i_t0, i_u0, i_tE))
    angles = np.pi * np.arange(int(so["n_orientation"])) / int(so["n_orientation"])
    phases = 2.0 * np.pi * np.arange(int(so["n_phase"])) / int(so["n_phase"])
    out = []
    for q, qf in so["pairs"]:
        for fac in so["xi_over_u0"]:
            xi = float(fac) * max(abs(u0), 1e-4)
            for th in angles:
                for ph in phases:
                    for inc in so["inclinations"]:
                        g = {
                            "t0": clip_to_bounds(t0, bounds["t0"]),
                            "u0": clip_to_bounds(u0, bounds["u0"]),
                            "tE": clip_to_bounds(tE, bounds["tE"]),
                            "xi_para": clip_to_bounds(xi * np.cos(th), bounds["xi_para"]),
                            "xi_perp": clip_to_bounds(xi * np.sin(th), bounds["xi_perp"]),
                            "xi_angular_velocity": clip_to_bounds(omega, bounds["xi_angular_velocity"]),
                            "xi_phase": clip_to_bounds(ph, bounds["xi_phase"]),
                            "xi_inclination": clip_to_bounds(float(inc), bounds["xi_inclination"]),
                            "xi_mass_ratio": clip_to_bounds(float(q), bounds["xi_mass_ratio"]),
                            "q_flux_W149": clip_to_bounds(float(qf), bounds["q_flux_W149"]),
                        }
                        out.append({
                            "label": (f"bspl_second_order_P{2 * np.pi / omega:.4g}_qM{q:g}_qf{qf:g}"
                                      f"_x{fac:g}_th{th:.3g}_ph{ph:.3g}_i{inc:.3g}"),
                            "alpha": None, "family": "second_order", "initial_guess": g})
    return out


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
    frac = abs(centroid_factor(q, rule["q_flux"]))
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


# =============================================================================
# One data-derived BSPL start: linearized xallarap periodogram
# =============================================================================
#
# pyLIMA circular xallarap (pyLIMA.xallarap.xallarap and
# ML_model.xallarap_trajectory_shifts) moves source 1 by
#
#   (dtau, dbeta) = [[xi_para, xi_perp], [-xi_perp, xi_para]]
#                   . m1 (cos W - cos phi, sin(i) (sin W - sin phi)),
#   W = omega (t - t0_xi) + phi,   m1 = q_M / (1 + q_M),
#
# with t0_xi = the t0 of the fit reference (roman_rubin build_fit_pyLIMA_model).
# To first order in xi the magnitude changes by dm = G_tau dtau + G_beta dbeta,
# G_tau = dm/dtau = -tE dm/dt0 and G_beta = dm/du0 of the best PSPL. For each
# omega the time-dependent part is linear in four coefficients,
#
#   dtau  = c1 cos x + c2 sin x,   dbeta = c3 cos x + c4 sin x,   x = omega (t - t0_xi),
#
# and the constant part is absorbed by the PSPL parameters (nuisance). The
# chi2 reduction Delta chi2(omega) is the BSPL likelihood in the small-xi limit.
# At the peak C = [[c1, c2], [c3, c4]] = m1 xi Rot(-theta) diag(1, sin i) Rot(phi),
# so its SVD gives xi, theta (xi_para = xi cos theta, xi_perp = xi sin theta),
# the inclination and the phase.


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def xallarap_design(t, err, jac, i_t0, i_u0, tE, t0_xi, window_tE=3.0):
    """
    Ingredients of the linearized xallarap model inside |t - t0_xi| < window_tE tE:
    selection mask, whitening weights sw = 1/|err|, orthonormal basis Q of the
    whitened PSPL Jacobian (nuisance directions), and the two PSPL derivative
    columns g_tau = -tE dm/dt0, g_beta = dm/du0 that the orbit modulates.
    """
    sel = np.abs(t - t0_xi) < window_tE * abs(tE)
    J = jac[sel]
    sw = np.sqrt(1.0 / err[sel] ** 2)          # as before the refactor (bitwise identical)
    Q, _ = np.linalg.qr(J * sw[:, None])
    return {"sel": sel, "t": t[sel] - t0_xi, "sw": sw, "Q": Q,
            "g_tau": -abs(tE) * J[:, i_t0], "g_beta": J[:, i_u0]}


def xallarap_columns(design, omega):
    """Whitened xallarap basis at omega with the PSPL directions projected out (n x 4)."""
    x = omega * design["t"]
    c, s = np.cos(x), np.sin(x)
    g_tau, g_beta, Q = design["g_tau"], design["g_beta"], design["Q"]
    X = np.column_stack([g_tau * c, g_tau * s, g_beta * c, g_beta * s]) * design["sw"][:, None]
    return X - Q @ (Q.T @ X)


def linearized_xallarap_periodogram(t, resid, err, jac, i_t0, i_u0, tE, t0_xi,
                                    omegas, window_tE=3.0):
    """
    Delta chi2(omega) of the four xallarap basis functions on top of the PSPL
    Jacobian `jac` (N x K, dm/dp for every fitted PSPL parameter).

    Returns (dchi2, coefficients): coefficients[j] = (c1, c2, c3, c4) at omegas[j].
    """
    design = xallarap_design(t, err, jac, i_t0, i_u0, tE, t0_xi, window_tE)
    Q = design["Q"]
    # Whiten and project out the PSPL directions (nuisance).
    rw = resid[design["sel"]] * design["sw"]
    rw = rw - Q @ (Q.T @ rw)

    dchi2 = np.empty(len(omegas))
    coef = np.empty((len(omegas), 4))
    for j, om in enumerate(omegas):
        X = xallarap_columns(design, om)
        b, *_ = np.linalg.lstsq(X, rw, rcond=None)
        fit = X @ b
        dchi2[j] = 2.0 * rw @ fit - fit @ fit
        coef[j] = b
    return dchi2, coef


def xallarap_from_coefficients(coef, mass_ratio, q_flux=0.0):
    """
    (c1, c2, c3, c4) -> pyLIMA xi_para, xi_perp, xi_phase, xi_inclination.

    The coefficients describe the light centroid, which moves by (B - eta)
    times the source-1 displacement (centroid_factor); q_flux = 0 gives the
    dark-companion conversion (factor B = q_M/(1+q_M)).
    """
    C = np.array([[coef[0], coef[1]], [coef[2], coef[3]]])
    U, S, Vt = np.linalg.svd(C)
    S = np.array(S, dtype=float)
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1.0
        S[1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[1, :] *= -1.0
        S[1] *= -1.0
    m1 = centroid_factor(mass_ratio, q_flux)
    xi = S[0] / m1                              # signed: B < eta flips the orbit
    sin_i = S[1] / S[0] if S[0] > 0 else 0.0
    theta = np.arctan2(U[0, 1], U[0, 0])        # U = Rot(-theta)
    phi = np.arctan2(Vt[1, 0], Vt[0, 0])        # Vt = Rot(phi)
    return {
        "xi_para": float(xi * np.cos(theta)),
        "xi_perp": float(xi * np.sin(theta)),
        "xi_phase": float(phi),
        "xi_inclination": float(np.arcsin(np.clip(sin_i, -1.0, 1.0))),
    }


def joint_coefficients(t, resid, err, jac, i_t0, i_u0, tE, t0_xi, omega, window_tE=3.0):
    """
    Weighted least squares of the residuals on [jac, xallarap basis] at one omega.
    Returns (a, c): corrections to every PSPL parameter (columns of jac) and the
    four xallarap coefficients. The PSPL corrections are part of the linear
    optimum: along nearly degenerate directions (e.g. dbeta oscillating slowly
    across a narrow peak, which looks like a change of u0) the solution pairs a
    large orbit with an opposite change of u0, t0 or tE.
    """
    sel = np.abs(t - t0_xi) < window_tE * abs(tE)
    t, r, sw = t[sel], resid[sel], 1.0 / err[sel]
    J = jac[sel]
    x = omega * (t - t0_xi)
    g_tau, g_beta = -abs(tE) * J[:, i_t0], J[:, i_u0]
    X = np.column_stack([J, g_tau * np.cos(x), g_tau * np.sin(x),
                         g_beta * np.cos(x), g_beta * np.sin(x)])
    sol, *_ = np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)
    return sol[:J.shape[1]], sol[J.shape[1]:]


def candidates_at_omega(names, pspl_best, a, c, omega, rule, bounds):
    """
    BSPL starts on the linear-optimum direction at one omega (PSPL corrections
    a, xallarap coefficients c from joint_coefficients), scaled by
    rule["amplitude_scales"]. pyLIMA's shift is L(t) - L(t0_xi), with
    L(t0_xi) = (c1, c3); the start that reproduces the linear trajectory is
    tE' = tE + a_tE,  t0' = t0 + a_t0 - tE' c1,  u0' = u0 + a_u0 + c3.
    """
    i_t0, i_u0, i_tE = names.index("t0"), names.index("u0"), names.index("tE")
    tE = float(pspl_best[i_tE])
    pairs = start_pairs(rule)
    candidates = []
    for q, qf in pairs:
        m = centroid_factor(q, qf)
        if abs(m) < MIN_CENTROID_FACTOR:              # on q_f ~ q_M: no linear start
            continue
        # The linear fit describes the light centroid.  pyLIMA anchors source 1
        # at t0_xi; the centroid sits at source 1 - eta xi, so the constant
        # term L(t0_xi) = (c1, c3) maps to the source-1 anchor with a factor
        # B / (B - eta) if the companion keeps its offset at t0_xi, or 1 if both
        # sources are anchored.  Both are tried (identical for q_flux = 0); the
        # caller ranks the candidates by their full-model chi2.
        eta = 0.0 if qf < DARK_Q_FLUX else qf / (1.0 + qf)
        f_offset = (m + eta) / m                       # = B / (B - eta)
        anchors = [1.0] if abs(f_offset - 1.0) < 1e-6 else [1.0, f_offset]
        for f in anchors:
            for alpha in rule.get("amplitude_scales", [1.0]):
                xi = xallarap_from_coefficients(alpha * c, q, qf)
                tE_start = tE + alpha * a[i_tE]
                t0_start = pspl_best[i_t0] + alpha * a[i_t0] - tE_start * f * alpha * c[0]
                u0_start = pspl_best[i_u0] + alpha * a[i_u0] + f * alpha * c[2]
                guess = {
                    "t0": clip_to_bounds(t0_start, bounds["t0"]),
                    "u0": clip_to_bounds(u0_start, bounds["u0"]),
                    "tE": clip_to_bounds(tE_start, bounds["tE"]),
                    "xi_para": clip_to_bounds(xi["xi_para"], bounds["xi_para"]),
                    "xi_perp": clip_to_bounds(xi["xi_perp"], bounds["xi_perp"]),
                    "xi_angular_velocity": clip_to_bounds(omega, bounds["xi_angular_velocity"]),
                    "xi_phase": clip_to_bounds(xi["xi_phase"], bounds["xi_phase"]),
                    "xi_inclination": clip_to_bounds(xi["xi_inclination"],
                                                     bounds["xi_inclination"]),
                    "xi_mass_ratio": clip_to_bounds(q, bounds["xi_mass_ratio"]),
                    "q_flux_W149": clip_to_bounds(qf, bounds["q_flux_W149"]),
                }
                # Dark policies (one pair, one anchor) keep their original labels.
                tag = "" if len(pairs) == 1 else f"_qM{q:g}_qf{qf:g}_f{f:.3g}"
                candidates.append({
                    "label": f"bspl_periodogram_P{2 * np.pi / omega:.4g}_a{alpha:g}{tag}",
                    "alpha": float(alpha),
                    "family": "linear",
                    "initial_guess": guess,
                })
    return candidates + second_order_candidates(names, pspl_best, omega, rule, bounds)


def bspl_start_candidates(t, resid, err, jac, names, pspl_best, t0_xi, rule, bounds):
    """
    Linearized periodogram, then BSPL start candidates at its peak.

    The frequency grid is uniform in f = 1/P on [1/P_max, 1/P_min] with
    df = 1 / (oversample * 2 * window_tE * tE): `oversample` nodes per
    independent frequency of the data window around the peak.

    pyLIMA's shift is L(t) - L(t0_xi), with L the four-coefficient basis and
    L(t0_xi) = (c1, c3); the linear fit puts that constant into the PSPL
    corrections a. The start that reproduces the linear trajectory exactly is

        tE' = tE + a_tE,   t0' = t0 + a_t0 - tE' c1,   u0' = u0 + a_u0 + c3.

    Candidate k scales the linear optimum by alpha_k (rule["amplitude_scales"]):
    orbit coefficients and PSPL corrections (t0, u0, tE) are multiplied by
    alpha_k together, so every candidate lies on the linear-optimum direction.
    The caller evaluates the full model on each candidate and keeps the best;
    this guards against orbits too large for the linear approximation.
    Returns (candidates, periodogram).
    """
    i_t0, i_u0, i_tE = names.index("t0"), names.index("u0"), names.index("tE")
    tE = float(pspl_best[i_tE])
    window = float(rule.get("window_tE", 3.0))
    df = 1.0 / (float(rule["oversample"]) * 2.0 * window * abs(tE))
    f = np.arange(1.0 / rule["period_max_days"], 1.0 / rule["period_min_days"] + df, df)
    omegas = 2.0 * np.pi * f

    dchi2, _ = linearized_xallarap_periodogram(
        t, resid, err, jac, i_t0, i_u0, tE, t0_xi, omegas, window)
    k = int(np.argmax(dchi2))
    a, c = joint_coefficients(t, resid, err, jac, i_t0, i_u0, tE, t0_xi, omegas[k], window)
    q = float(rule["xi_mass_ratio"])

    candidates = candidates_at_omega(names, pspl_best, a, c, omegas[k], rule, bounds)
    periodogram = {
        "period_days": (2.0 * np.pi / omegas).tolist(),
        "delta_chi2": dchi2.tolist(),
        "peak_period_days": float(2.0 * np.pi / omegas[k]),
        "peak_delta_chi2": float(dchi2[k]),
        "peak_coefficients": c.tolist(),
        "peak_pspl_corrections": dict(zip(names, a.tolist())),
        "t0_xi": float(t0_xi),
    }
    return candidates, periodogram
