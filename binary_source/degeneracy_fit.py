import numpy as np
import pandas as pd
import os
import sys
import subprocess
from pathlib import Path

from pyLIMA.models import PSPL_model

import scipy.optimize as so


from functions_aux import (
    mag,
    flux_to_mag,
    mag_to_flux,
    sigma_flux_from_sigma_mag,
    sigma_flux_from_flux,
    orbital_period_kepler,
    build_sim_event,
    a_from_P_kepler_days,
)


# ============================================================
# Numerical-method provenance
# ============================================================

FIT_OBJECTIVE = "intrinsic_magnification_trapezoid"

# np.trapezoid exists from numpy 2.0; np.trapz is the same rule in older numpy.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _get_git_state():
    """
    Return repository commit and dirty-state information.

    These values are stored in every final NPZ file so that each
    numerical result can be traced back to the exact code version.
    """

    repo_root = Path(__file__).resolve().parents[1]

    try:

        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short=12",
                "HEAD",
            ],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        dirty = bool(
            status.strip()
        )

    except Exception:

        commit = "unknown"
        dirty = True

    return commit, dirty


CODE_COMMIT, CODE_DIRTY = _get_git_state()


def pspl_magnification(
    t,
    t0,
    u0,
    tE,
):
    """
    Standard PSPL magnification.

    Parameters
    ----------
    t : array-like
        Times.
    t0 : float
        Time of closest approach.
    u0 : float
        Impact parameter.
    tE : float
        Einstein timescale.

    Returns
    -------
    A : ndarray
        PSPL magnification.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    if (
        not np.isfinite(t0)
        or not np.isfinite(u0)
        or not np.isfinite(tE)
        or tE <= 0.0
        or u0 < 0.0
    ):
        return None

    tau = (
        t - t0
    ) / tE

    u = np.sqrt(
        u0**2
        + tau**2
    )

    A = (
        u**2 + 2.0
    ) / (
        u
        * np.sqrt(
            u**2 + 4.0
        )
    )

    return A


def intrinsic_magnification_objective(
    fit_params,
    t,
    A_truth,
):
    """
    Intrinsic BSPL--PSPL objective.

    Minimizes exactly the squared mismatch in magnification:

        S_A = integral [A_BSPL(t) - A_PSPL(t)]^2 dt

    This is the numerator of D^2, apart from its normalization.
    """

    t0, u0, tE = np.asarray(
        fit_params,
        dtype=float,
    )

    A_fit = pspl_magnification(
        t=t,
        t0=t0,
        u0=u0,
        tE=tE,
    )

    if A_fit is None:
        return 1e100

    residual = (
        np.asarray(
            A_truth,
            dtype=float,
        )
        - A_fit
    )

    if not np.all(
        np.isfinite(residual)
    ):
        return 1e100

    return float(
        _trapezoid(
            residual**2,
            x=t,
        )
    )


# ============================================================
# Optional PSPL + annual-parallax fit (same objective, same
# Nelder-Mead settings as the plain PSPL fit)
# ============================================================

class _ShiftedParallaxEvaluator:
    """
    PSPL + parallax evaluated on t + offset (JD) while the fit works in the
    same relative time as the plain PSPL fit. t0 is passed relative and
    shifted here, so the plain fit is exactly the one used without parallax.
    """

    def __init__(self, evaluator, offset):
        self.evaluator = evaluator
        self.offset = float(offset)
        self.ephemerides_path = evaluator.ephemerides_path

    def magnification(self, t0, **kwargs):
        return self.evaluator.magnification(t0=t0 + self.offset, **kwargs)


def intrinsic_parallax_objective(
    fit_params,
    t,
    A_truth,
    evaluator,
):
    """
    Same objective as intrinsic_magnification_objective, with the PSPL
    magnification replaced by PSPL + parallax:

        S_A = integral [A_BSPL(t) - A_PSPL+pi(t)]^2 dt

    fit_params = (t0, u0, tE, piEN, piEE). u0 may be negative, because
    with parallax the side on which the lens passes matters.
    """

    t0, u0, tE, piEN, piEE = np.asarray(
        fit_params,
        dtype=float,
    )

    if (
        not np.all(np.isfinite(fit_params))
        or tE <= 0.0
    ):
        return 1e100

    try:
        A_fit = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=piEN,
            piEE=piEE,
        )
    except Exception:
        return 1e100

    residual = (
        np.asarray(A_truth, dtype=float)
        - np.asarray(A_fit, dtype=float)
    )

    if not np.all(np.isfinite(residual)):
        return 1e100

    return float(
        _trapezoid(
            residual**2,
            x=t,
        )
    )


PIE_START_NORM = 10.0


def fit_pspl_parallax(
    t,
    A_truth,
    best_pspl,
    evaluator,
    previous_solution=None,
):
    """
    Fit PSPL + parallax to A_truth with several starting points
    (the objective has several local minima where parallax absorbs much
    of the binary-source signal):

      0-1  (t0, +u0, tE, 0, 0) and (t0, -u0, tE, 0, 0) from the plain fit;
      2-5  (t0, +u0, tE, piE) with |piE| = PIE_START_NORM towards
           N, E, S and W;
      6    previous_solution, the best parallax fit of the previous
           period of the grid (continuation in P), when given.

    Each start is followed by one Nelder-Mead restart from its own result.
    The plain PSPL is the piE = 0 member of this model, so if no start
    improves on it the plain fit is returned (nested=True, start=-1).

    Returns dict(x, S, nested, start).
    """

    t0, u0, tE = np.asarray(best_pspl, dtype=float)

    S_plain = intrinsic_magnification_objective(
        [t0, abs(u0), tE],
        t,
        A_truth,
    )

    best = dict(
        x=np.array([t0, abs(u0), tE, 0.0, 0.0]),
        S=S_plain,
        nested=True,
        start=-1,
    )

    starts = [
        np.array([t0, +abs(u0), tE, 0.0, 0.0]),
        np.array([t0, -abs(u0), tE, 0.0, 0.0]),
    ]

    for piEN0, piEE0 in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
        starts.append(
            np.array([t0, abs(u0), tE,
                      PIE_START_NORM * piEN0, PIE_START_NORM * piEE0])
        )

    if previous_solution is not None and np.all(np.isfinite(previous_solution)):
        starts.append(np.asarray(previous_solution, dtype=float).copy())

    steps = np.array(
        [
            0.01 * tE,
            0.05 * max(abs(u0), 1e-3),
            0.05 * tE,
            0.05,
            0.05,
        ]
    )

    # Tolerances relative to the scale of the problem: S ~ S_plain, and
    # 1e-9 relative in S is far below what D needs (D ~ sqrt(S)). maxfev
    # caps the rare long runs; every start is restarted once and there are
    # seven starts, so a capped run does not decide the result alone.
    options = dict(
        maxiter=5000,
        maxfev=3000,
        xatol=1e-6,
        fatol=1e-9 * max(S_plain, 1e-300),
    )

    for i_start, x_start in enumerate(starts):

        x0 = x_start.copy()

        for _ in range(2):          # start + one restart

            simplex = np.vstack(
                [x0] + [x0 + steps[k] * np.eye(5)[k] for k in range(5)]
            )

            res = so.minimize(
                intrinsic_parallax_objective,
                x0=x0,
                args=(t, A_truth, evaluator),
                method="Nelder-Mead",
                options=dict(options, initial_simplex=simplex),
            )

            x0 = np.asarray(res.x, dtype=float)

        S = float(res.fun)

        if np.isfinite(S) and S < best["S"]:
            best = dict(x=x0.copy(), S=S, nested=False, start=i_start)

    return best




# ============================================================
# Helper function:
# t_dev = FWHM de la feature dominante de |residual|
# ============================================================

def compute_tdev_fwhm(t, residual):
    """
    Compute t_dev, defined as the temporal width of the interval
    CONTIGUOUS interval containing the global maximum of |residual| and satisfying

        |residual(t)| >= 0.5 * max(|residual|)

    The half-maximum crossings are linearly interpolated between
    puntos consecutivos.

    Parameters
    ----------
    t : array-like
        Vector temporal.

    residual : array-like
        Residuales, por ejemplo A_truth - A_fit.

    Returns
    -------
    t_dev : float
        Full width at half maximum de la feature dominante,
        en las mismas unidades que t.

    censored : bool
        True si la feature alcanza alguno de los extremos de la
        time window, so the full FWHM may
        extenderse fuera del intervalo simulado.
    """

    t = np.asarray(t, dtype=float)
    residual = np.asarray(residual, dtype=float)

    if len(t) != len(residual):

        raise ValueError(
            "t and residual must have the same length."
        )

    if len(t) < 2:

        return np.nan, True

    abs_resid = np.abs(residual)

    valid = (
        np.isfinite(t)
        & np.isfinite(abs_resid)
    )

    if not np.any(valid):

        return np.nan, True

    # In principle, t and resid should already be fully valid,
    # but we retain this treatment for robustness.

    t_valid = t[valid]
    y_valid = abs_resid[valid]

    if len(t_valid) < 2:

        return np.nan, True

    # Ordenar temporalmente por seguridad

    order = np.argsort(t_valid)

    t_valid = t_valid[order]
    y_valid = y_valid[order]

    # Global maximum of |residual|

    i_max = int(
        np.argmax(y_valid)
    )

    r_max = float(
        y_valid[i_max]
    )

    if (
        not np.isfinite(r_max)
        or r_max <= 0
    ):

        return np.nan, False

    half_max = 0.5 * r_max

    # ========================================================
    # Find the CONTIGUOUS interval around the maximum
    # ========================================================

    i_left = i_max

    while (
        i_left > 0
        and y_valid[i_left - 1] >= half_max
    ):

        i_left -= 1

    i_right = i_max

    while (
        i_right < len(y_valid) - 1
        and y_valid[i_right + 1] >= half_max
    ):

        i_right += 1

    # ========================================================
    # Cruce izquierdo
    # ========================================================

    left_censored = False

    if i_left == 0:

        # The signal is still above half maximum
        # en el borde izquierdo.

        t_left = t_valid[0]
        left_censored = True

    else:

        t1 = t_valid[i_left - 1]
        t2 = t_valid[i_left]

        y1 = y_valid[i_left - 1]
        y2 = y_valid[i_left]

        # Linear interpolation:
        #
        # y(t_cross) = half_max

        if np.isclose(
            y2,
            y1,
        ):

            t_left = 0.5 * (
                t1 + t2
            )

        else:

            frac = (
                (half_max - y1)
                / (y2 - y1)
            )

            t_left = (
                t1
                + frac * (t2 - t1)
            )

    # ========================================================
    # Cruce derecho
    # ========================================================

    right_censored = False

    if i_right == len(y_valid) - 1:

        # The signal is still above half maximum
        # en el borde derecho.

        t_right = t_valid[-1]
        right_censored = True

    else:

        t1 = t_valid[i_right]
        t2 = t_valid[i_right + 1]

        y1 = y_valid[i_right]
        y2 = y_valid[i_right + 1]

        if np.isclose(
            y2,
            y1,
        ):

            t_right = 0.5 * (
                t1 + t2
            )

        else:

            frac = (
                (half_max - y1)
                / (y2 - y1)
            )

            t_right = (
                t1
                + frac * (t2 - t1)
            )

    # ========================================================
    # FWHM
    # ========================================================

    t_dev = float(
        t_right - t_left
    )

    censored = (
        left_censored
        or right_censored
    )

    if (
        not np.isfinite(t_dev)
        or t_dev < 0
    ):

        return np.nan, censored

    return t_dev, censored


# ============================================================
# NUEVO
# Normalized BSPL--PSPL degeneracy metric
# ============================================================

def compute_D_metric(
    t,
    A_truth,
    A_fit,
):
    """
    Compute the dimensionless metric

                    integral [A_truth(t) - A_fit(t)]^2 dt
        D^2 =      ------------------------------------------
                    integral [A_truth(t) - 1]^2 dt

    es decir,

        D = ||A_truth - A_fit||_2 / ||A_truth - 1||_2.

    The metric compares the irreducible residual of the best fit
    PSPL relative to the total scale of the BSPL microlensing signal
    respecto del baseline A = 1.

    Parameters
    ----------
    t : array-like
        Vector temporal.

    A_truth : array-like
        True magnification of the BSPL model.

    A_fit : array-like
        Magnification of the best-fitting PSPL model.

    Returns
    -------
    D : float
        Dimensionless mismatch metric.

        D -> 0:
            very strong BSPL--PSPL degeneracy.

        D grande:
            a significant fraction of the light-curve structure
            BSPL cannot be reproduced by the PSPL model.

    Notes
    -----
    D does not depend on the time units because the same
    diferencial dt aparece en numerador y denominador.

    It also does not require assuming a particular photometric precision.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    A_truth = np.asarray(
        A_truth,
        dtype=float,
    )

    A_fit = np.asarray(
        A_fit,
        dtype=float,
    )

    # --------------------------------------------------------
    # Check array sizes
    # --------------------------------------------------------

    if not (
        len(t)
        == len(A_truth)
        == len(A_fit)
    ):

        raise ValueError(
            "t, A_truth, and A_fit must have the same length."
        )

    # --------------------------------------------------------
    # Mask of valid values
    # --------------------------------------------------------

    valid = (
        np.isfinite(t)
        & np.isfinite(A_truth)
        & np.isfinite(A_fit)
    )

    if np.count_nonzero(valid) < 2:

        return np.nan

    t_valid = t[valid]
    A_truth_valid = A_truth[valid]
    A_fit_valid = A_fit[valid]

    # --------------------------------------------------------
    # Ordenar temporalmente por seguridad
    # --------------------------------------------------------

    order = np.argsort(
        t_valid
    )

    t_valid = t_valid[order]

    A_truth_valid = (
        A_truth_valid[order]
    )

    A_fit_valid = (
        A_fit_valid[order]
    )

    # --------------------------------------------------------
    # Residual irreducible BSPL - PSPL
    # --------------------------------------------------------

    delta_A = (
        A_truth_valid
        - A_fit_valid
    )

    # --------------------------------------------------------
    # Total microlensing signal relative to the A = 1 baseline
    # --------------------------------------------------------

    signal_A = (
        A_truth_valid
        - 1.0
    )

    # --------------------------------------------------------
    # Norma L2 al cuadrado
    # --------------------------------------------------------

    numerator = _trapezoid(
        delta_A**2,
        x=t_valid,
    )

    denominator = _trapezoid(
        signal_A**2,
        x=t_valid,
    )

    # --------------------------------------------------------
    # Robustez
    # --------------------------------------------------------

    if (
        not np.isfinite(numerator)
        or numerator < 0
    ):

        return np.nan

    if (
        not np.isfinite(denominator)
        or denominator <= 0
    ):

        return np.nan

    # --------------------------------------------------------
    # D
    # --------------------------------------------------------

    D = np.sqrt(
        numerator / denominator
    )

    if not np.isfinite(D):

        return np.nan

    return float(D)


# ============================================================
# Main function
# ============================================================

def run_grid_and_save_npz_kepler(
    out_npz_path: str,
    t: np.ndarray,

    # truth PSPL-like trajectory params
    t0_true: float,
    u0_true: float,
    tE_true: float,

    # xallarap angular params
    phi_true: float,
    i_true: float,
    qflux_true: float,
    theta_true: float,

    # physical params for Kepler consistency
    M1_Msun: float,
    M2_Msun: float,
    rEhat_AU: float,

    # grid (P is scanned)
    P_grid: np.ndarray,

    # fixed photometric wrapper
    msource_true: float = 1.0,
    mtotal_true: float = 0.0,

    # legacy wrapper configuration; the intrinsic PSPL fit
    # is always performed in magnification space
    use_magnification_fit: bool = False,

    # storage
    store_curves: bool = True,

    # ========================================================
    # Additional configuration
    # ========================================================

    override_xiE: float | None = None,

    set_flux_from_truth_photometry: bool = True,

    rms_on_magnification: bool = True,

    # ========================================================
    # Optional second fit: PSPL + annual parallax seen from
    # Roman (L2). The parallax model is evaluated at
    # t + parallax_time_offset (JD), with reference time
    # t0_true + parallax_time_offset; t, t0_true and the plain
    # PSPL fit stay exactly as without parallax.
    # ========================================================

    fit_parallax: bool = False,

    parallax_time_offset: float = 0.0,

    parallax_ephemerides_path: str | None = None,
):

    """
    For each period:

    1) Generate the true BSPL/xallarap model.
    2) Build the theoretical photometry.
    3) Ajusta un PSPL variando [t0, u0, tE].
    4) Compute the residual metrics.
    5) Save the results to an NPZ file.

    Stored metrics
    ------------------
    RMS
        RMS del residual.

    MAXABS
        Maximum absolute residual.

    D
        Normalized distance between the BSPL light curve and its best
        PSPL fit:

            D = sqrt[
                    integral (A_BSPL - A_PSPL)^2 dt
                    --------------------------------
                    integral (A_BSPL - 1)^2 dt
                ]

        This metric is always computed in magnification space.

    TDEV
        FWHM temporal de la feature dominante de |residual|.

    TFWHM
        Alias de TDEV.

    TDEV_CENSORED
        True when the region above half maximum reaches
        either edge of the time window.

    Q_A
        sqrt(chi2/N), following the original implementation.
    """

    P_grid = np.asarray(
        P_grid,
        dtype=float,
    )

    t = np.asarray(
        t,
        dtype=float,
    )

    # ========================================================
    # Mass ratio y masa total
    # ========================================================

    q_mass_true = float(
        M2_Msun / M1_Msun
    )

    Mtot_Msun = float(
        M1_Msun + M2_Msun
    )

    n_P = len(P_grid)
    n_t = len(t)

    # ========================================================
    # Result arrays
    # ========================================================

    RMS = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    MAXABS = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # NUEVO:
    # Normalized degeneracy metric
    # ========================================================

    D = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # FWHM duration of the dominant feature
    # ========================================================

    TDEV = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # indica si el FWHM toca el borde temporal
    # ========================================================

    TDEV_CENSORED = np.zeros(
        n_P,
        dtype=bool,
    )

    DT0 = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    DU0 = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    DTE = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    SUCCESS = np.zeros(
        n_P,
        dtype=bool,
    )

    Q_A = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    BEST_T0U0TE = np.full(
        (n_P, 3),
        np.nan,
        dtype=float,
    )

    XI_E = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # Optional PSPL + parallax fit
    # ========================================================

    if fit_parallax:

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from binary_source.analysis.intrinsic_parallax import (
            PSPLParallaxEvaluator,
            file_sha256,
        )

        pi_evaluator = _ShiftedParallaxEvaluator(
            PSPLParallaxEvaluator(
                t + float(parallax_time_offset),
                t0_reference=t0_true + float(parallax_time_offset),
                ephemerides_path=parallax_ephemerides_path,
            ),
            parallax_time_offset,
        )

        eph_path = str(pi_evaluator.ephemerides_path)
        eph_sha256 = file_sha256(pi_evaluator.ephemerides_path)

        D_PI = np.full(n_P, np.nan, dtype=float)
        BEST_PI = np.full((n_P, 5), np.nan, dtype=float)
        PI_NESTED = np.zeros(n_P, dtype=bool)
        SUCCESS_PI = np.zeros(n_P, dtype=bool)
        PI_START = np.full(n_P, -1, dtype=np.int64)
        previous_pi_solution = None

    A_AU = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # Save complete light curves
    # ========================================================

    if store_curves:

        A_truth_grid = np.full(
            (n_P, n_t),
            np.nan,
            dtype=np.float32,
        )

        A_fit_grid = np.full(
            (n_P, n_t),
            np.nan,
            dtype=np.float32,
        )

        F_truth_grid = np.full(
            (n_P, n_t),
            np.nan,
            dtype=np.float32,
        )

        F_fit_grid = np.full(
            (n_P, n_t),
            np.nan,
            dtype=np.float32,
        )

    # ========================================================
    # Loop over period
    # ========================================================

    for j_P, P in enumerate(P_grid):

        try:

            omega = (
                2.0 * np.pi / P
            )

            # =================================================
            # xiE: Kepler o override
            # =================================================

            if override_xiE is None:

                a_AU = a_from_P_kepler_days(
                    P,
                    Mtot_Msun,
                )

                xiE = (
                    a_AU
                    / float(rEhat_AU)
                )

            else:

                xiE = float(
                    override_xiE
                )

                a_AU = (
                    xiE
                    * float(rEhat_AU)
                )

            A_AU[j_P] = a_AU
            XI_E[j_P] = xiE

            xi_para = (
                xiE
                * np.cos(theta_true)
            )

            xi_perp = (
                xiE
                * np.sin(theta_true)
            )

            # =================================================
            # Evento
            # =================================================

            ev = build_sim_event(
                t,
                mag0=19.0,
                emag=0.01,
                filt="G",
            )

            # =================================================
            # True BSPL/xallarap model
            # =================================================

            model_xal = PSPL_model.PSPLmodel(
                ev,
                parallax=["None", 0.0],
                double_source=[
                    "Circular",
                    t0_true,
                ],
            )

            model_xal.define_model_parameters()

            ZP = 27.615

            fsource_true = mag_to_flux(
                msource_true,
                zp=ZP,
            )

            ftotal_true = mag_to_flux(
                mtotal_true,
                zp=ZP,
            )

            fblend_true = 0.0

            params_xal = [
                t0_true,
                u0_true,
                tE_true,
                xi_para,
                xi_perp,
                omega,
                phi_true,
                i_true,
                q_mass_true,
                qflux_true,
                fsource_true,
                ftotal_true,
            ]

            py_params_xal = (
                model_xal.compute_pyLIMA_parameters(
                    params_xal
                )
            )

            # =================================================
            # True magnification light curve
            # =================================================

            A_truth = (
                model_xal.model_magnification(
                    ev.telescopes[0],
                    py_params_xal,
                )
                / (1.0 + qflux_true)
            )

            A_truth = np.asarray(
                A_truth,
                dtype=float,
            )

            # =================================================
            # True light curve in flux
            # =================================================

            F_truth = (
                model_xal.compute_the_microlensing_model(
                    ev.telescopes[0],
                    py_params_xal,
                )["photometry"]
            )

            F_truth = np.asarray(
                F_truth,
                dtype=float,
            )

            # =================================================
            # Data construction
            # =================================================

            if set_flux_from_truth_photometry:

                ev.telescopes[0].lightcurve[
                    "flux"
                ] = F_truth

            else:

                if use_magnification_fit:

                    ev.telescopes[0].lightcurve[
                        "flux"
                    ] = A_truth

                else:

                    ev.telescopes[0].lightcurve[
                        "flux"
                    ] = F_truth

            # =================================================
            # Modelo PSPL
            # =================================================

            model_pspl = PSPL_model.PSPLmodel(
                ev,
                parallax=["None", 0.0],
                double_source=["None", 0.0],
            )

            model_pspl.define_model_parameters()

            # =================================================
            # Fit [t0, u0, tE]
            # =================================================

            x0 = np.array(
                [
                    t0_true,
                    u0_true,
                    tE_true,
                ],
                dtype=float,
            )

            res = so.minimize(
                intrinsic_magnification_objective,
                x0=x0,
                args=(
                    t,
                    A_truth,
                ),
                method="Nelder-Mead",
                options=dict(
                    maxiter=20000,
                    xatol=1e-10,
                    fatol=1e-12,
                ),
            )

            if not res.success:

                SUCCESS[j_P] = False
                continue

            best = np.asarray(
                res.x,
                dtype=float,
            )

            BEST_T0U0TE[
                j_P,
                :
            ] = best

            # =================================================
            # PSPL reconstruction
            # =================================================

            best_full = np.concatenate(
                [
                    best,
                    [
                        fsource_true,
                        ftotal_true,
                    ],
                ]
            )

            py_params_best = (
                model_pspl.compute_pyLIMA_parameters(
                    best_full
                )
            )

            A_fit = (
                model_pspl.model_magnification(
                    ev.telescopes[0],
                    py_params_best,
                )
            )

            A_fit = np.asarray(
                A_fit,
                dtype=float,
            )

            F_fit = (
                model_pspl.compute_the_microlensing_model(
                    ev.telescopes[0],
                    py_params_best,
                )["photometry"]
            )

            F_fit = np.asarray(
                F_fit,
                dtype=float,
            )

            # =================================================
            # Residual used for RMS/MAXABS/TDEV
            # =================================================

            if rms_on_magnification:

                resid = (
                    A_truth
                    - A_fit
                )

            else:

                resid = (
                    np.asarray(
                        ev.telescopes[0]
                        .lightcurve["flux"]
                        .value,
                        dtype=float,
                    )
                    -
                    F_fit
                )

            # =================================================
            # Q_A
            # =================================================

            mag_truth = flux_to_mag(
                F_truth
            )

            mag_fit = flux_to_mag(
                F_fit
            )

            mask = (
                (mag_truth >= 12.0)
                & (mag_truth <= 27.0)
                & (mag_fit >= 12.0)
                & (mag_fit <= 27.0)
            )

            F_truth_m = F_truth[
                mask
            ]

            F_fit_m = F_fit[
                mask
            ]

            err_truth = sigma_flux_from_flux(
                F_truth_m
            )

            err_fit = sigma_flux_from_flux(
                F_fit_m
            )

            residuals = (
                F_truth_m
                - F_fit_m
            )

            if len(residuals) > 0:

                q_A = (
                    np.sum(
                        (
                            residuals
                            / err_truth
                        ) ** 2
                    )
                    / len(residuals)
                )

                Q_A[j_P] = q_A

            # =================================================
            # RMS
            # =================================================

            RMS[j_P] = float(
                np.sqrt(
                    np.mean(
                        resid**2
                    )
                )
            )

            # =================================================
            # Maximum absolute residual
            # =================================================

            MAXABS[j_P] = float(
                np.max(
                    np.abs(resid)
                )
            )

            # =================================================
            # NUEVO:
            # D = mismatch BSPL--PSPL normalizado
            #
            # IMPORTANTE:
            # is ALWAYS computed in magnification space:
            #
            #             ∫(A_truth - A_fit)^2 dt
            # D^2 =      -------------------------
            #             ∫(A_truth - 1)^2 dt
            #
            # No depende de rms_on_magnification.
            # =================================================

            D[j_P] = compute_D_metric(
                t=t,
                A_truth=A_truth,
                A_fit=A_fit,
            )

            # =================================================
            # t_dev / t_FWHM
            # =================================================

            t_dev, censored = compute_tdev_fwhm(
                t,
                resid,
            )

            TDEV[j_P] = t_dev

            TDEV_CENSORED[
                j_P
            ] = censored

            # =================================================
            # Parameter biases
            # =================================================

            DT0[j_P] = (
                best[0]
                - t0_true
            )

            DU0[j_P] = (
                best[1]
                - u0_true
            )

            DTE[j_P] = (
                best[2]
                - tE_true
            )

            # =================================================
            # Optional PSPL + parallax fit
            # =================================================

            if fit_parallax:

                try:

                    pi_fit = fit_pspl_parallax(
                        t=t,
                        A_truth=A_truth,
                        best_pspl=best,
                        evaluator=pi_evaluator,
                        previous_solution=previous_pi_solution,
                    )

                    x_pi = pi_fit["x"]

                    A_fit_pi = pi_evaluator.magnification(
                        t0=x_pi[0],
                        u0=x_pi[1],
                        tE=x_pi[2],
                        piEN=x_pi[3],
                        piEE=x_pi[4],
                    )

                    D_PI[j_P] = compute_D_metric(
                        t=t,
                        A_truth=A_truth,
                        A_fit=np.asarray(A_fit_pi, dtype=float),
                    )

                    BEST_PI[j_P, :] = x_pi
                    PI_NESTED[j_P] = bool(pi_fit["nested"])
                    PI_START[j_P] = int(pi_fit["start"])
                    SUCCESS_PI[j_P] = True
                    previous_pi_solution = np.asarray(x_pi, dtype=float)

                except Exception as error:

                    print(
                        f"[ERROR parallax] "
                        f"P={P:.6g} d: "
                        f"{error}"
                    )

            # =================================================
            # Save light curves
            # =================================================

            if store_curves:

                A_truth_grid[
                    j_P,
                    :
                ] = np.asarray(
                    A_truth,
                    dtype=np.float32,
                )

                A_fit_grid[
                    j_P,
                    :
                ] = np.asarray(
                    A_fit,
                    dtype=np.float32,
                )

                F_truth_grid[
                    j_P,
                    :
                ] = np.asarray(
                    F_truth,
                    dtype=np.float32,
                )

                F_fit_grid[
                    j_P,
                    :
                ] = np.asarray(
                    F_fit,
                    dtype=np.float32,
                )

            SUCCESS[
                j_P
            ] = True

        except Exception as error:

            SUCCESS[
                j_P
            ] = False

            print(
                f"[ERROR] "
                f"P={P:.6g} d: "
                f"{error}"
            )

            continue

    # ========================================================
    # Payload
    # ========================================================

    payload = dict(

        t=t,

        P_grid=P_grid,

        xiE_of_P=XI_E,

        a_AU_of_P=A_AU,

        # ====================================================
        # Metrics
        # ====================================================

        RMS=RMS,

        MAXABS=MAXABS,

        # NUEVO
        D=D,

        TDEV=TDEV,

        # Explicit alias
        TFWHM=TDEV,

        TDEV_CENSORED=TDEV_CENSORED,

        # ====================================================
        # Biases
        # ====================================================

        DT0=DT0,

        DU0=DU0,

        DTE=DTE,

        SUCCESS=SUCCESS,

        BEST_T0U0TE=BEST_T0U0TE,

        Q_A=np.sqrt(Q_A),


        # ================================================
        # Numerical provenance
        # ================================================

        fit_objective=np.array(FIT_OBJECTIVE),

        code_commit=np.array(CODE_COMMIT),

        code_dirty=np.bool_(CODE_DIRTY),

        n_time=np.int64(n_t),

        n_period=np.int64(n_P),

        time_min=np.float64(np.min(t)),

        time_max=np.float64(np.max(t)),

        n_success=np.int64(np.count_nonzero(SUCCESS)),

        n_failed=np.int64(n_P - np.count_nonzero(SUCCESS)),

        q_mass_true=np.float64(q_mass_true),

        Mtot_Msun=np.float64(Mtot_Msun),

        qflux_true=np.float64(qflux_true),

        rEhat_AU=np.float64(rEhat_AU),

        store_curves=np.bool_(store_curves),
        truth=np.array(
            [
                t0_true,
                u0_true,
                tE_true,
                phi_true,
                i_true,
                M1_Msun,
                M2_Msun,
                rEhat_AU,
                qflux_true,
                theta_true,
                fsource_true,
                fblend_true,
                float(
                    use_magnification_fit
                ),
                (
                    -1.0
                    if override_xiE is None
                    else float(
                        override_xiE
                    )
                ),
                float(
                    set_flux_from_truth_photometry
                ),
                float(
                    rms_on_magnification
                ),
            ],
            dtype=float,
        ),
    )

    # ========================================================
    # Optional PSPL + parallax results
    # ========================================================

    if fit_parallax:

        payload.update(
            D_PI=D_PI,
            BEST_T0U0TE_PIEN_PIEE=BEST_PI,
            PI_NESTED=PI_NESTED,
            SUCCESS_PI=SUCCESS_PI,
            PI_START=PI_START,
            pie_start_norm=np.float64(PIE_START_NORM),
            parallax_time_offset=np.float64(parallax_time_offset),
            parallax_ephemerides_path=np.array(eph_path),
            parallax_ephemerides_sha256=np.array(eph_sha256),
        )

    # ========================================================
    # Save complete light curves
    # ========================================================

    if store_curves:

        payload[
            "A_truth_grid"
        ] = A_truth_grid

        payload[
            "A_fit_grid"
        ] = A_fit_grid

        payload[
            "F_truth_grid"
        ] = F_truth_grid

        payload[
            "F_fit_grid"
        ] = F_fit_grid

    # ========================================================
    # Save output
    # ========================================================

    np.savez_compressed(
        out_npz_path,
        **payload,
    )

    print(
        f"Saved: {out_npz_path} | "
        f"fits={np.count_nonzero(SUCCESS)}/{n_P} | "
        f"failed={n_P - np.count_nonzero(SUCCESS)}"
    )


# ============================================================
# EXECUTION EXAMPLE
# ============================================================

if __name__ == "__main__":

    t0_true = 50.0
    u0_true = 0.1
    tE_true = 173.0

    t = np.linspace(
        t0_true - 3.5 * tE_true,
        t0_true + 3.5 * tE_true,
        10000,
    )

    phi_true = 0.0
    i_true = np.pi / 2.0
    theta_true = 0.0
    qflux_true = 0.0

    M1 = 2.0
    M2 = 1.0
    rEhat = 5.0

    P_grid = np.logspace(
        1,
        5,
        60,
    )

    run_grid_and_save_npz_kepler(
        out_npz_path="scan_kepler.npz",
        t=t,
        t0_true=t0_true,
        u0_true=u0_true,
        tE_true=tE_true,
        phi_true=phi_true,
        i_true=i_true,
        qflux_true=qflux_true,
        theta_true=theta_true,
        M1_Msun=M1,
        M2_Msun=M2,
        rEhat_AU=rEhat,
        P_grid=P_grid,
        msource_true=24.0,
        mtotal_true=24.0,
        override_xiE=None,
        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,
        store_curves=True,
    )
