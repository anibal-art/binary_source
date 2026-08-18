import numpy as np
import pandas as pd
import os
import sys

current_path = os.getcwd()
parent_directory = os.path.abspath(
    os.path.join(current_path, os.pardir)
)

print("Parent Directory:", parent_directory)
sys.path.append(parent_directory)

from pyLIMA.models import PSPL_model

import scipy.optimize as so

from functions_aux import (
    mag,
    flux_to_mag,
    sigma_W149_func,
    mag_to_flux,
    sigma_flux_from_sigma_mag,
    sigma_flux_from_flux,
    orbital_period_kepler,
    chi2_theoretical,
    build_sim_event,
    a_from_P_kepler_days,
)


# ============================================================
# Función auxiliar:
# t_dev = FWHM de la feature dominante de |residual|
# ============================================================

def compute_tdev_fwhm(t, residual):
    """
    Calcula t_dev, definido como el ancho temporal del intervalo
    CONTIGUO que contiene el máximo global de |residual| y satisface

        |residual(t)| >= 0.5 * max(|residual|)

    Los cruces con half maximum se interpolan linealmente entre
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
        ventana temporal, de modo que el FWHM completo podría
        extenderse fuera del intervalo simulado.
    """

    t = np.asarray(t, dtype=float)
    residual = np.asarray(residual, dtype=float)

    if len(t) != len(residual):

        raise ValueError(
            "t y residual deben tener la misma longitud."
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

    # En principio t y resid deberían ser completamente válidos,
    # pero mantenemos este tratamiento por robustez.

    t_valid = t[valid]
    y_valid = abs_resid[valid]

    if len(t_valid) < 2:

        return np.nan, True

    # Ordenar temporalmente por seguridad

    order = np.argsort(t_valid)

    t_valid = t_valid[order]
    y_valid = y_valid[order]

    # Máximo global de |residual|

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
    # Buscar el intervalo CONTIGUO alrededor del máximo
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

        # La señal todavía está por encima de half max
        # en el borde izquierdo.

        t_left = t_valid[0]
        left_censored = True

    else:

        t1 = t_valid[i_left - 1]
        t2 = t_valid[i_left]

        y1 = y_valid[i_left - 1]
        y2 = y_valid[i_left]

        # Interpolación lineal:
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

        # La señal todavía está por encima de half max
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
# Métrica normalizada de degeneración BSPL--PSPL
# ============================================================

def compute_D_metric(
    t,
    A_truth,
    A_fit,
):
    """
    Calcula la métrica adimensional

                    integral [A_truth(t) - A_fit(t)]^2 dt
        D^2 =      ------------------------------------------
                    integral [A_truth(t) - 1]^2 dt

    es decir,

        D = ||A_truth - A_fit||_2 / ||A_truth - 1||_2.

    La métrica compara el residual irreducible del mejor ajuste
    PSPL con la escala total de la señal de microlensing BSPL
    respecto del baseline A = 1.

    Parameters
    ----------
    t : array-like
        Vector temporal.

    A_truth : array-like
        Magnificación verdadera del modelo BSPL.

    A_fit : array-like
        Magnificación del mejor ajuste PSPL.

    Returns
    -------
    D : float
        Métrica adimensional de mismatch.

        D -> 0:
            degeneración BSPL--PSPL muy fuerte.

        D grande:
            una fracción importante de la estructura de la curva
            BSPL no puede ser reproducida por el PSPL.

    Notes
    -----
    D no depende de las unidades temporales, porque el mismo
    diferencial dt aparece en numerador y denominador.

    Tampoco requiere asumir una precisión fotométrica particular.
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
    # Verificar tamaños
    # --------------------------------------------------------

    if not (
        len(t)
        == len(A_truth)
        == len(A_fit)
    ):

        raise ValueError(
            "t, A_truth y A_fit deben tener la misma longitud."
        )

    # --------------------------------------------------------
    # Máscara de valores válidos
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
    # Señal total de microlensing respecto del baseline A = 1
    # --------------------------------------------------------

    signal_A = (
        A_truth_valid
        - 1.0
    )

    # --------------------------------------------------------
    # Norma L2 al cuadrado
    # --------------------------------------------------------

    numerator = np.trapezoid(
        delta_A**2,
        x=t_valid,
    )

    denominator = np.trapezoid(
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
# Función principal
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

    # objective config
    use_magnification_fit: bool = False,

    # storage
    store_curves: bool = True,

    # ========================================================
    # Configuración adicional
    # ========================================================

    override_xiE: float | None = None,

    set_flux_from_truth_photometry: bool = True,

    rms_on_magnification: bool = True,
):

    """
    Para cada período:

    1) Genera el modelo BSPL/xallarap verdadero.
    2) Construye la fotometría teórica.
    3) Ajusta un PSPL variando [t0, u0, tE].
    4) Calcula las métricas de los residuos.
    5) Guarda los resultados en un archivo NPZ.

    Métricas guardadas
    ------------------
    RMS
        RMS del residual.

    MAXABS
        Máximo absoluto del residual.

    D
        Distancia normalizada entre la curva BSPL y su mejor
        ajuste PSPL:

            D = sqrt[
                    integral (A_BSPL - A_PSPL)^2 dt
                    --------------------------------
                    integral (A_BSPL - 1)^2 dt
                ]

        Esta métrica se calcula siempre usando magnificación.

    TDEV
        FWHM temporal de la feature dominante de |residual|.

    TFWHM
        Alias de TDEV.

    TDEV_CENSORED
        True cuando la región por encima del half maximum alcanza
        alguno de los extremos de la ventana temporal.

    Q_A
        sqrt(chi2/N), según la implementación original.
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
    # Arrays de resultados
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
    # Métrica normalizada de degeneración
    # ========================================================

    D = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # duración FWHM de la feature dominante
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

    A_AU = np.full(
        n_P,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # Guardar curvas completas
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
    # Loop en período
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
            # Modelo BSPL/xallarap verdadero
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
            # Curva verdadera en magnificación
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
            # Curva verdadera en flujo
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
            # Construcción de los datos
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
            # Ajuste [t0, u0, tE]
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
                chi2_theoretical,
                x0=x0,
                args=(
                    model_pspl,
                    False,
                    fsource_true,
                    ftotal_true,
                ),
                method="Nelder-Mead",
                options=dict(
                    maxiter=20000,
                    xatol=1e-10,
                    fatol=1e-6,
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
            # Reconstrucción PSPL
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
            # Residual utilizado para RMS/MAXABS/TDEV
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
            # Máximo absoluto
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
            # se calcula SIEMPRE en magnificación:
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
            # Bias de parámetros
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
            # Guardar curvas
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
        # Métricas
        # ====================================================

        RMS=RMS,

        MAXABS=MAXABS,

        # NUEVO
        D=D,

        TDEV=TDEV,

        # Alias explícito
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
    # Guardar curvas completas
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
    # Guardado
    # ========================================================

    np.savez_compressed(
        out_npz_path,
        **payload,
    )

    print(
        f"Saved: {out_npz_path}"
    )


# ============================================================
# EJEMPLO DE EJECUCIÓN
# ============================================================

# PSPL base

t0_true = 50.0
u0_true = 0.1
tE_true = 173.0

t = np.linspace(
    -3.5 * tE_true,
    3.5 * tE_true,
    10000,
)


# ============================================================
# Parámetros orbitales
# ============================================================

phi_true = 0.0

i_true = (
    np.pi / 2.0
)

theta_true = 0.0

qflux_true = 0.0


# ============================================================
# Sistema físico
# ============================================================

M1 = 2.0
M2 = 1.0

rEhat = 5.0

P_grid = np.logspace(
    1,
    5,
    60,
)


# ============================================================
# Ejecutar scan
# ============================================================

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
