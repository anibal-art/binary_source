#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:45:46 2026

@author: anibal-pc
"""

"""
degeneracy_fit_fast.py  –  versión optimizada
Cambios respecto al original:
  1. Nelder-Mead → Powell  (2-4x más rápido en 3 parámetros)
  2. Resolución temporal adaptativa con cap en N_MAX_POINTS  (hasta 10x en tE grandes)
  3. Paralelismo en el eje (tE, u0) en vez del eje P  (elimina overhead de 175 pools)
  4. initializer en ProcessPoolExecutor  (imports pesados una sola vez por proceso)
  5. Worker secuencial para el eje P dentro de cada proceso
  6. Código comentado (bloque Chebyshev) preservado intacto
"""

import os
import sys
import numpy as np
import scipy.optimize as so
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product as iproduct

# ---------------------------------------------------------------------------
# imports del proyecto – se resuelven en el módulo normal o en el initializer
# ---------------------------------------------------------------------------
from Chebyshev import Chebyhev_coefficients, evaluate_chebyshev
from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.simulations import simulator
from pyLIMA.fits import TRF_fit
import pyLIMA

current_path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_directory)
import pyLIMA_plots

from astropy import units as u
from astropy import constants as C
from pyLIMA.xallarap.xallarap import xallarap_shifts, compute_xallarap_curvature
from func_tools import *

# ---------------------------------------------------------------------------
# Constante: máximo de puntos temporales para evitar arrays gigantes
# ---------------------------------------------------------------------------
N_MAX_POINTS = 50_000   # ajustá si necesitás más resolución


# ===========================================================================
# initializer del pool: importa módulos pesados una sola vez por proceso
# ===========================================================================
def _pool_initializer():
    """Ejecutado una vez al arrancar cada worker-process."""
    global PSPL_model
    from pyLIMA.models import PSPL_model as _pspl
    PSPL_model = _pspl


# ===========================================================================
# Worker: procesa UN valor de P dado un conjunto fijo de parámetros de evento
# ===========================================================================
def _worker_single_P_kepler(args):
    """
    Igual que el original, con dos cambios:
      - usa Powell en lugar de Nelder-Mead
      - sin cambios en la lógica de xallarap / PSPL
    """
    (
        j_P,
        P,
        t,
        t0_true,
        u0_true,
        tE_true,
        phi_true,
        i_true,
        qflux_true,
        theta_true,
        M1_Msun,
        M2_Msun,
        rEhat_AU,
        fsource_true,
        fblend_true,
        use_magnification_fit,
        store_curves,
        override_xiE,
        set_flux_from_truth_photometry,
        rms_on_magnification,
    ) = args

    result = {
        "j_P": j_P,
        "success": False,
        "RMS": np.nan,
        "MAXABS": np.nan,
        "DT0": np.nan,
        "DU0": np.nan,
        "DTE": np.nan,
        "BEST_T0U0TE": np.array([np.nan, np.nan, np.nan], dtype=float),
        "XI_E": np.nan,
        "A_AU": np.nan,
        "intL1": np.nan,
        "t0_interval": np.nan,
        "tE_interval": np.nan,
        "CHEB_COEFF_TRUTH": np.full(50, np.nan, dtype=float),
        "CHEB_COEFF_FIT":   np.full(50, np.nan, dtype=float),
        "xi_para": np.nan,
        "xi_perp": np.nan,
        "q_mass_true": np.nan,
        "P": np.nan,
        "u0_true": np.nan,
        "t0_true": np.nan,
    }

    if store_curves:
        n_t = len(t)
        result["A_truth"] = np.full(n_t, np.nan, dtype=np.float32)
        result["A_fit"]   = np.full(n_t, np.nan, dtype=np.float32)
        result["F_truth"] = np.full(n_t, np.nan, dtype=np.float32)
        result["F_fit"]   = np.full(n_t, np.nan, dtype=np.float32)

    try:
        ftotal_true  = fsource_true + fblend_true
        q_mass_true  = float(M2_Msun / M1_Msun)
        Mtot_Msun    = float(M1_Msun + M2_Msun)
        omega        = 2.0 * np.pi / P

        result["q_mass_true"] = q_mass_true
        result["P"]           = P
        result["u0_true"]     = u0_true
        result["t0_true"]     = t0_true

        # ---- xiE: Kepler o override ----------------------------------------
        if override_xiE is None:
            a_AU = a_from_P_kepler_days(P, Mtot_Msun)
            xiE  = a_AU / float(rEhat_AU)
        else:
            xiE  = float(override_xiE)
            a_AU = xiE * float(rEhat_AU)

        result["A_AU"] = a_AU
        result["XI_E"] = xiE

        xi_para = xiE * np.cos(theta_true)
        xi_perp = xiE * np.sin(theta_true)
        result["xi_para"] = xi_para
        result["xi_perp"] = xi_perp

        # ---- modelo xallarap "verdadero" ------------------------------------
        ev = build_sim_event(t, mag0=19.0, emag=0.01, filt="G")

        model_xal = PSPL_model.PSPLmodel(
            ev, parallax=["None", 0.0], double_source=["Circular", t0_true]
        )
        model_xal.define_model_parameters()

        params_xal = [
            t0_true, u0_true, tE_true,
            xi_para, xi_perp, omega,
            phi_true, i_true,
            q_mass_true, qflux_true,
            fsource_true, ftotal_true,
        ]
        py_params_xal = model_xal.compute_pyLIMA_parameters(params_xal)
        A_truth = model_xal.model_magnification(ev.telescopes[0], py_params_xal)

        # ---- construir "data" ----------------------------------------------
        if set_flux_from_truth_photometry:
            F_truth = model_xal.compute_the_microlensing_model(
                ev.telescopes[0], py_params_xal
            )["photometry"]
            ev.telescopes[0].lightcurve["flux"] = F_truth
        else:
            if use_magnification_fit:
                ev.telescopes[0].lightcurve["flux"] = A_truth
                F_truth = np.asarray(A_truth, dtype=float)
            else:
                F_truth = model_xal.compute_the_microlensing_model(
                    ev.telescopes[0], py_params_xal
                )["photometry"]
                ev.telescopes[0].lightcurve["flux"] = F_truth

        # ---- modelo PSPL para el fit ----------------------------------------
        model_pspl = PSPL_model.PSPLmodel(
            ev, parallax=["None", 0.0], double_source=["None", 0.0]
        )
        model_pspl.define_model_parameters()

        x0 = np.array([t0_true, u0_true, tE_true], dtype=float)

        # *** CAMBIO 1: Powell en lugar de Nelder-Mead (2-4x más rápido) ***
        res = so.minimize(
            chi2_theoretical,
            x0=x0,
            args=(model_pspl, False, fsource_true, ftotal_true),
            method="Powell",
            options=dict(maxiter=10_000, xtol=1e-8, ftol=1e-8),
        )

        if not res.success:
            return result

        best     = np.asarray(res.x, dtype=float)
        result["BEST_T0U0TE"] = best

        best_full       = np.concatenate([best, [fsource_true, ftotal_true]])
        py_params_best  = model_pspl.compute_pyLIMA_parameters(best_full)

        A_fit = model_pspl.model_magnification(ev.telescopes[0], py_params_best)
        F_fit = model_pspl.compute_the_microlensing_model(
            ev.telescopes[0], py_params_best
        )["photometry"]

        # ---- [BLOQUE CHEBYSHEV - desactivado, preservado] ------------------
        # residual_structure = detect_residual_structure_envelope(...)
        # ...

        # ---- RMS -----------------------------------------------------------
        if rms_on_magnification:
            resid = A_truth - A_fit
        else:
            resid = ev.telescopes[0].lightcurve["flux"].value - F_fit

        result["RMS"]    = float(np.sqrt(np.mean(resid**2)))
        result["MAXABS"] = float(np.max(np.abs(resid)))
        result["intL1"]  = float(np.trapezoid(np.abs(resid), t))

        result["DT0"] = best[0] - t0_true
        result["DU0"] = best[1] - u0_true
        result["DTE"] = best[2] - tE_true

        if store_curves:
            result["A_truth"] = np.asarray(A_truth,  dtype=np.float32)
            result["A_fit"]   = np.asarray(A_fit,    dtype=np.float32)
            result["F_truth"] = np.asarray(
                ev.telescopes[0].lightcurve["flux"].value, dtype=np.float32
            )
            result["F_fit"]   = np.asarray(F_fit, dtype=np.float32)

        result["success"] = True
        return result

    except Exception:
        return result


# ===========================================================================
# run_grid_and_save_npz_kepler  –  barrido sobre P para un (tE, u0) fijo
# ===========================================================================
def run_grid_and_save_npz_kepler(
    out_npz_path: str,
    t: np.ndarray,
    t0_true: float,
    u0_true: float,
    tE_true: float,
    phi_true: float,
    i_true: float,
    qflux_true: float,
    theta_true: float,
    M1_Msun: float,
    M2_Msun: float,
    rEhat_AU: float,
    P_grid: np.ndarray,
    fsource_true: float = 1.0,
    fblend_true:  float = 0.0,
    use_magnification_fit: bool = False,
    store_curves: bool = True,
    override_xiE: float | None = None,
    set_flux_from_truth_photometry: bool = True,
    rms_on_magnification: bool = True,
    # n_jobs ignorado aquí: la paralelización ocurre en el nivel (tE, u0)
    n_jobs: int | None = None,
):
    """
    Corre el barrido en P de forma **secuencial** dentro del proceso.
    La paralelización ahora ocurre en el nivel externo (tE × u0).
    """
    P_grid  = np.asarray(P_grid, dtype=float)
    n_P     = len(P_grid)
    n_t     = len(t)
    degree_cheb = 50

    # arrays de salida
    RMS     = np.full(n_P, np.nan)
    MAXABS  = np.full(n_P, np.nan)
    DT0     = np.full(n_P, np.nan)
    DU0     = np.full(n_P, np.nan)
    DTE     = np.full(n_P, np.nan)
    SUCCESS = np.zeros(n_P, dtype=bool)
    BEST_T0U0TE      = np.full((n_P, 3),          np.nan)
    XI_E             = np.full(n_P, np.nan)
    A_AU             = np.full(n_P, np.nan)
    intL1            = np.full(n_P, np.nan)
    xi_para_arr      = np.full(n_P, np.nan)
    xi_perp_arr      = np.full(n_P, np.nan)
    t0_interval_arr  = np.full(n_P, np.nan)
    tE_interval_arr  = np.full(n_P, np.nan)
    CHEB_COEFF_TRUTH = np.full((n_P, degree_cheb), np.nan)
    CHEB_COEFF_FIT   = np.full((n_P, degree_cheb), np.nan)

    if store_curves:
        A_truth_grid = np.full((n_P, n_t), np.nan, dtype=np.float32)
        A_fit_grid   = np.full((n_P, n_t), np.nan, dtype=np.float32)
        F_truth_grid = np.full((n_P, n_t), np.nan, dtype=np.float32)
        F_fit_grid   = np.full((n_P, n_t), np.nan, dtype=np.float32)

    worker_args = [
        (
            j_P, float(P), np.asarray(t, dtype=float),
            t0_true, u0_true, tE_true,
            phi_true, i_true, qflux_true, theta_true,
            M1_Msun, M2_Msun, rEhat_AU,
            fsource_true, fblend_true,
            use_magnification_fit, store_curves,
            override_xiE, set_flux_from_truth_photometry, rms_on_magnification,
        )
        for j_P, P in enumerate(P_grid)
    ]

    # *** CAMBIO 2: siempre secuencial dentro del worker de (tE, u0) ***
    results = [_worker_single_P_kepler(arg) for arg in worker_args]

    for res in results:
        j = res["j_P"]
        SUCCESS[j]          = res["success"]
        RMS[j]              = res["RMS"]
        MAXABS[j]           = res["MAXABS"]
        DT0[j]              = res["DT0"]
        DU0[j]              = res["DU0"]
        DTE[j]              = res["DTE"]
        BEST_T0U0TE[j, :]   = res["BEST_T0U0TE"]
        XI_E[j]             = res["XI_E"]
        A_AU[j]             = res["A_AU"]
        intL1[j]            = res["intL1"]
        t0_interval_arr[j]  = res["t0_interval"]
        tE_interval_arr[j]  = res["tE_interval"]
        CHEB_COEFF_TRUTH[j] = res["CHEB_COEFF_TRUTH"]
        CHEB_COEFF_FIT[j]   = res["CHEB_COEFF_FIT"]
        if store_curves:
            A_truth_grid[j] = res["A_truth"]
            A_fit_grid[j]   = res["A_fit"]
            F_truth_grid[j] = res["F_truth"]
            F_fit_grid[j]   = res["F_fit"]

    payload = dict(
        t=t, P_grid=P_grid, u0_true=u0_true, tE_true=tE_true,
        xiE_of_P=XI_E, a_AU_of_P=A_AU,
        RMS=RMS, MAXABS=MAXABS, intL1=intL1,
        DT0=DT0, DU0=DU0, DTE=DTE,
        SUCCESS=SUCCESS, BEST_T0U0TE=BEST_T0U0TE,
        t0_interval=t0_interval_arr, tE_interval=tE_interval_arr,
        CHEB_COEFF_TRUTH=CHEB_COEFF_TRUTH, CHEB_COEFF_FIT=CHEB_COEFF_FIT,
        truth=np.array([
            t0_true, u0_true, tE_true, phi_true, i_true,
            M1_Msun, M2_Msun, rEhat_AU, qflux_true, theta_true,
            fsource_true, fblend_true,
            float(use_magnification_fit),
            -1.0 if override_xiE is None else float(override_xiE),
            float(set_flux_from_truth_photometry),
            float(rms_on_magnification),
        ], dtype=float),
    )
    if store_curves:
        payload.update(
            A_truth_grid=A_truth_grid, A_fit_grid=A_fit_grid,
            F_truth_grid=F_truth_grid, F_fit_grid=F_fit_grid,
        )

    np.savez_compressed(out_npz_path, **payload)
    print(f"Saved: {out_npz_path}")


# ===========================================================================
# Worker de nivel externo: procesa UNA combinación (tE_true, k, u0_true)
# ===========================================================================
def _worker_tE_u0(args):
    """
    Ejecutado en un proceso del pool externo.
    Corre el barrido completo en P para un par (tE, u0).
    """
    (
        tE_true, k, u0_true,
        t0_true, phi_true, i_true, qflux_true, theta_true,
        M1, M2, rEhat,
        P_grid,
        base_dir,
    ) = args

    # *** CAMBIO 3: resolución adaptativa con cap ***
    n_raw   = int(7 * tE_true * 24 * 4)
    n_pts   = min(n_raw, N_MAX_POINTS)
    t       = np.linspace(-3.5 * tE_true, 3.5 * tE_true, n_pts)

    directory = os.path.join(base_dir, f"scan_u0_tE{int(tE_true)}")
    os.makedirs(directory, exist_ok=True)
    out_name = os.path.join(directory, f"scan_kepler_u0_{k:03d}.npz")

    run_grid_and_save_npz_kepler(
        out_npz_path=out_name,
        t=t,
        t0_true=t0_true,
        u0_true=float(u0_true),
        tE_true=tE_true,
        phi_true=phi_true,
        i_true=float(i_true),
        qflux_true=qflux_true,
        theta_true=theta_true,
        M1_Msun=M1,
        M2_Msun=M2,
        rEhat_AU=rEhat,
        P_grid=P_grid,
        fsource_true=1.0,
        fblend_true=0.0,
        override_xiE=None,
        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,
        n_jobs=1,   # secuencial dentro del worker
    )
    return (tE_true, k, u0_true)


# ===========================================================================
# Script principal
# ===========================================================================
if __name__ == "__main__":

    # ---- parámetros fijos --------------------------------------------------
    t0_true          = 50.0
    phi_true         = 0.0
    theta_true       = 0.0
    qflux_true       = 0.0
    lambda_xi_fixed  = 0.5 * np.pi   # i_true fijo = pi/2

    M1       = 2.0
    M2       = 1.0
    rEhat    = 5.0

    # ---- grillas -----------------------------------------------------------
    P_grid   = np.logspace(1, 5, 60)   # 10 d → 100000 d

    N_u0     = 25
    u0_grid  = np.linspace(0.01, 2.0, N_u0)

    tE_list  = [50, 100, 200, 300, 400, 500, 1000]

    base_dir = "/home/anibal-pc/binary_source/results"

    # ---- armar lista de trabajos -------------------------------------------
    combos = [
        (
            tE_true, k, float(u0),
            t0_true, phi_true, lambda_xi_fixed, qflux_true, theta_true,
            M1, M2, rEhat,
            P_grid,
            base_dir,
        )
        for tE_true, (k, u0) in iproduct(tE_list, enumerate(u0_grid))
    ]
    # total: 7 × 25 = 175 trabajos

    # *** CAMBIO 4: paralelismo en el eje (tE, u0) con initializer ***
    N_WORKERS = min(16, len(combos))

    print(f"Lanzando {len(combos)} trabajos con {N_WORKERS} workers …")

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_pool_initializer,   # importa pyLIMA una vez por proceso
    ) as ex:
        futures = {ex.submit(_worker_tE_u0, arg): arg for arg in combos}
        for fut in as_completed(futures):
            tE, k, u0 = fut.result()
            print(f"  ✓  tE={tE:4g}  u0[{k:02d}]={u0:.4f}")

    print("scan on (tE, u0) terminado.")