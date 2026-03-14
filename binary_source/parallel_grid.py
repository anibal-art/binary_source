import os
import numpy as np
import scipy.optimize as so
from concurrent.futures import ProcessPoolExecutor, as_completed
from Chebyshev import Chebyhev_coefficients
from Chebyshev import evaluate_chebyshev
from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.simulations import simulator
from pyLIMA.fits import TRF_fit
import pyLIMA, sys
import pandas as pd

current_path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_path, os.pardir))
print("Parent Directory:", parent_directory)
sys.path.append(parent_directory)
import pyLIMA_plots
from astropy import units as u
from astropy import constants as C
from pyLIMA.xallarap.xallarap import xallarap_shifts, compute_xallarap_curvature

from func_tools import *
def _worker_single_P_kepler(args):
    """
    Worker para procesar un único valor de P.
    Devuelve un dict con todos los resultados del índice j_P.
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
        "CHEB_COEFF_FIT": np.full(50, np.nan, dtype=float),
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
        ftotal_true = fsource_true + fblend_true
        q_mass_true = float(M2_Msun / M1_Msun)
        Mtot_Msun = float(M1_Msun + M2_Msun)

        omega = 2.0 * np.pi / P

        result["q_mass_true"] = q_mass_true
        result["P"] = P
        result["u0_true"] = u0_true
        result["t0_true"] = t0_true
        # ============================================================
        # xiE: Kepler o override
        # ============================================================
        if override_xiE is None:
            a_AU = a_from_P_kepler_days(P, Mtot_Msun)
            xiE = a_AU / float(rEhat_AU)
        else:
            xiE = float(override_xiE)
            a_AU = xiE * float(rEhat_AU)

        result["A_AU"] = a_AU
        result["XI_E"] = xiE

        xi_para = xiE * np.cos(theta_true)
        xi_perp = xiE * np.sin(theta_true)
        result["xi_para"] = xi_para
        result["xi_perp"] = xi_perp
        # Event fresco
        ev = build_sim_event(t, mag0=19.0, emag=0.01, filt="G")

        # ---------- "Verdad" xallarap
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

        # ============================================================
        # construir "data" como en tu ejemplo
        # ============================================================
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

        # ---------- Modelo PSPL para el fit
        model_pspl = PSPL_model.PSPLmodel(
            ev, parallax=["None", 0.0], double_source=["None", 0.0]
        )
        model_pspl.define_model_parameters()

        x0 = np.array([t0_true, u0_true, tE_true], dtype=float)
        res = so.minimize(
            chi2_theoretical,
            x0=x0,
            args=(model_pspl, False, fsource_true, ftotal_true),
            method="Nelder-Mead",
            options=dict(maxiter=20000, xatol=1e-10, fatol=1e-10),
        )

        if not res.success:
            return result

        best = np.asarray(res.x, dtype=float)
        result["BEST_T0U0TE"] = best

        best_full = np.concatenate([best, [fsource_true, ftotal_true]])
        py_params_best = model_pspl.compute_pyLIMA_parameters(best_full)

        A_fit = model_pspl.model_magnification(ev.telescopes[0], py_params_best)
        F_fit = model_pspl.compute_the_microlensing_model(
            ev.telescopes[0], py_params_best
        )["photometry"]

        # ============================================================
        # [NUEVO BLOQUE AGREGADO] detección de estructura residual
        # ============================================================
        residual_structure = detect_residual_structure_envelope(
            t=t,
            A_truth=A_truth,
            A_fit=A_fit,
            t0_ref=t0_true,
            smooth_window=80,
            fraction_of_peak=0.001,
            min_points=5,
            pad_fraction=0.20
        )

        df_truth_struct = residual_structure["df_interval_truth"]
        df_fit_struct   = residual_structure["df_interval_fit"]

        t0_interval = residual_structure["t0_interval"]
        tE_interval = residual_structure["tE_interval"]

        result["t0_interval"] = t0_interval
        result["tE_interval"] = tE_interval

        # ============================================================
        # [NUEVO BLOQUE AGREGADO] Chebyshev en el intervalo detectado
        # ============================================================
        degree_cheb = 50

        coeff_truth = Chebyhev_coefficients(
            df_truth_struct,
            t0_interval,
            tE_interval,
            degree_cheb
        )

        coeff_fit = Chebyhev_coefficients(
            df_fit_struct,
            t0_interval,
            tE_interval,
            degree_cheb
        )

        result["CHEB_COEFF_TRUTH"] = np.asarray(coeff_truth, dtype=float)
        result["CHEB_COEFF_FIT"] = np.asarray(coeff_fit, dtype=float)

        cheb_truth = evaluate_chebyshev(
            df_truth_struct,
            t0_interval,
            tE_interval,
            coeff_truth
        )

        cheb_fit = evaluate_chebyshev(
            df_fit_struct,
            t0_interval,
            tE_interval,
            coeff_fit
        )

        t_cheb_truth = np.sort(df_truth_struct["t"].values)
        t_cheb_fit   = np.sort(df_fit_struct["t"].values)

        # ============================================================
        # RMS
        # ============================================================
        if rms_on_magnification:
            resid = A_truth - A_fit
        else:
            resid = ev.telescopes[0].lightcurve["flux"].value - F_fit

        result["RMS"] = float(np.sqrt(np.mean(resid**2)))
        result["MAXABS"] = float(np.max(np.abs(resid)))
        result["intL1"] = float(np.trapz(np.abs(resid), t))

        result["DT0"] = best[0] - t0_true
        result["DU0"] = best[1] - u0_true
        result["DTE"] = best[2] - tE_true

        if store_curves:
            result["A_truth"] = np.asarray(A_truth, dtype=np.float32)
            result["A_fit"]   = np.asarray(A_fit, dtype=np.float32)
            result["F_truth"] = np.asarray(ev.telescopes[0].lightcurve["flux"].value, dtype=np.float32)
            result["F_fit"]   = np.asarray(F_fit, dtype=np.float32)

        result["success"] = True
        return result

    except Exception:
        return result

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
    fsource_true: float = 1.0,
    fblend_true: float = 0.0,
    # objective config
    use_magnification_fit: bool = False,
    # storage
    store_curves: bool = True,
    # ============================================================
    # para replicar tu ejemplo
    # ============================================================
    override_xiE: float | None = None,
    set_flux_from_truth_photometry: bool = True,
    rms_on_magnification: bool = True,
    # ============================================================
    # NUEVO BLOQUE AGREGADO
    # ============================================================
    n_jobs: int | None = None,
):
    """
    Igual que tu ejemplo:
      1) Genera el modelo xallarap "verdadero".
      2) Construye data teórica poniendo telescope.lightcurve['flux'] = F_truth exacto.
      3) Ajusta PSPL variando [t0,u0,tE] minimizando SSE en flujo.
      4) Reporta RMS en magnificación (por defecto).

    NUEVO:
      - Si n_jobs is None o n_jobs == 1: corre secuencial.
      - Si n_jobs > 1: paraleliza sobre P_grid usando ProcessPoolExecutor.
    """
    ftotal_true = fsource_true + fblend_true
    P_grid = np.asarray(P_grid, dtype=float)

    q_mass_true = float(M2_Msun / M1_Msun)
    Mtot_Msun = float(M1_Msun + M2_Msun)

    n_P = len(P_grid)
    n_t = len(t)

    RMS     = np.full(n_P, np.nan, dtype=float)
    MAXABS  = np.full(n_P, np.nan, dtype=float)
    DT0     = np.full(n_P, np.nan, dtype=float)
    DU0     = np.full(n_P, np.nan, dtype=float)
    DTE     = np.full(n_P, np.nan, dtype=float)
    SUCCESS = np.zeros(n_P, dtype=bool)

    BEST_T0U0TE = np.full((n_P, 3), np.nan, dtype=float)
    XI_E = np.full(n_P, np.nan, dtype=float)
    A_AU = np.full(n_P, np.nan, dtype=float)
    intL1 = np.full(n_P, np.nan, dtype=float)

    xi_para_arr = np.full(n_P, np.nan, dtype=float)
    xi_perp_arr = np.full(n_P, np.nan, dtype=float)
    q_mass_true_arr = np.full(n_P, np.nan, dtype=float)
    P_arr = np.full(n_P, np.nan, dtype=float)
    u0_true_arr = np.full(n_P, np.nan, dtype=float)
    t0_true_arr = np.full(n_P, np.nan, dtype=float)
    degree_cheb = 50
    t0_interval_arr = np.full(n_P, np.nan, dtype=float)
    tE_interval_arr = np.full(n_P, np.nan, dtype=float)
    CHEB_COEFF_TRUTH = np.full((n_P, degree_cheb), np.nan, dtype=float)
    CHEB_COEFF_FIT = np.full((n_P, degree_cheb), np.nan, dtype=float)

    if store_curves:
        A_truth_grid = np.full((n_P, n_t), np.nan, dtype=np.float32)
        A_fit_grid   = np.full((n_P, n_t), np.nan, dtype=np.float32)
        F_truth_grid = np.full((n_P, n_t), np.nan, dtype=np.float32)
        F_fit_grid   = np.full((n_P, n_t), np.nan, dtype=np.float32)

    # ============================================================
    # NUEVO BLOQUE AGREGADO: argumentos por worker
    # ============================================================
    worker_args = [
        (
            j_P,
            float(P),
            np.asarray(t, dtype=float),
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
        )
        for j_P, P in enumerate(P_grid)
    ]

    # ============================================================
    # NUEVO BLOQUE AGREGADO: ejecución secuencial o paralela
    # ============================================================
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:
        results = [_worker_single_P_kepler(arg) for arg in worker_args]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_worker_single_P_kepler, arg) for arg in worker_args]
            for fut in as_completed(futures):
                results.append(fut.result())

    # ============================================================
    # NUEVO BLOQUE AGREGADO: recolección de resultados
    # ============================================================
    for res in results:
        j_P = res["j_P"]

        SUCCESS[j_P] = res["success"]
        RMS[j_P] = res["RMS"]
        MAXABS[j_P] = res["MAXABS"]
        DT0[j_P] = res["DT0"]
        DU0[j_P] = res["DU0"]
        DTE[j_P] = res["DTE"]
        BEST_T0U0TE[j_P, :] = res["BEST_T0U0TE"]
        XI_E[j_P] = res["XI_E"]
        A_AU[j_P] = res["A_AU"]
        intL1[j_P] = res["intL1"]

        # ============================================================
        # [NUEVO BLOQUE AGREGADO] recolectar intervalo y coeffs
        # ============================================================
        t0_interval_arr[j_P] = res["t0_interval"]
        tE_interval_arr[j_P] = res["tE_interval"]
        CHEB_COEFF_TRUTH[j_P, :] = res["CHEB_COEFF_TRUTH"]
        CHEB_COEFF_FIT[j_P, :] = res["CHEB_COEFF_FIT"]

        if store_curves:
            A_truth_grid[j_P, :] = res["A_truth"]
            A_fit_grid[j_P, :]   = res["A_fit"]
            F_truth_grid[j_P, :] = res["F_truth"]
            F_fit_grid[j_P, :]   = res["F_fit"]

    payload = dict(
        t=t,
        P_grid=P_grid,
        xiE_of_P=XI_E,
        a_AU_of_P=A_AU,
        RMS=RMS,
        MAXABS=MAXABS,
        intL1=intL1,
        DT0=DT0,
        DU0=DU0,
        DTE=DTE,
        SUCCESS=SUCCESS,
        BEST_T0U0TE=BEST_T0U0TE,

        # ============================================================
        # [NUEVO BLOQUE AGREGADO] guardar intervalo y coeffs
        # ============================================================
        t0_interval=t0_interval_arr,
        tE_interval=tE_interval_arr,
        CHEB_COEFF_TRUTH=CHEB_COEFF_TRUTH,
        CHEB_COEFF_FIT=CHEB_COEFF_FIT,

        truth=np.array([
            t0_true, u0_true, tE_true, phi_true, i_true,
            M1_Msun, M2_Msun, rEhat_AU, qflux_true, theta_true,
            fsource_true, fblend_true,
            float(use_magnification_fit),
            -1.0 if override_xiE is None else float(override_xiE),
            float(set_flux_from_truth_photometry),
            float(rms_on_magnification)
        ], dtype=float),
    )

    if store_curves:
        payload["A_truth_grid"] = A_truth_grid
        payload["A_fit_grid"]   = A_fit_grid
        payload["F_truth_grid"] = F_truth_grid
        payload["F_fit_grid"]   = F_fit_grid

    np.savez_compressed(out_npz_path, **payload)
    print(f"Saved: {out_npz_path}")

# from degeneracy_fit import run_grid_and_save_npz_kepler
import numpy as np
import os
# ============================================================
# time grid
# ============================================================
t = np.linspace(-500, 500, 5000)

# ============================================================
# "truth" PSPL params (base)
# ============================================================
t0_true = 50.0


# ============================================================
# orbital / xallarap angles (fixed)
# ============================================================
phi_true = 0.0
theta_true = 0.0
qflux_true = 0.0
lambda_xi_fixed = 0.5*np.pi   # <-- fijo: pi/2

# ============================================================
# physical system (fixed)
# ============================================================
M1 = 2.0
M2 = 1.0
rEhat = 5.0

# scan in P as before
P_grid = np.logspace(1, 5, 60)  # 10 d → 100000 d

# ============================================================
# [NUEVO] scan in u0
# ============================================================
N_u0 = 25
u0_grid = np.linspace(0.01, 1.0, N_u0)   # elegí rango; ajustalo a tu caso
# tE_true = 100.0
for tE_true in [50,100,150,200,300,400,500,1000]: 
    print("scan u0 for tE", tE_true)
    for k, u0_true in enumerate(u0_grid):
        directory = f"/home/anibal/binary_source/results/scan_u0_tE{int(tE_true)}/"
        os.makedirs(directory, exist_ok=True)
        out_name = directory+f"scan_kepler_u0_{k:03d}.npz"
    
        run_grid_and_save_npz_kepler(
            out_npz_path=out_name,
            t=t,
            t0_true=t0_true,
            u0_true=float(u0_true),      # <-- [CAMBIO CLAVE] barrido en u0
            tE_true=tE_true,
            phi_true=phi_true,
            i_true=float(lambda_xi_fixed),   # <-- fijo: lambda_xi = pi/2
            qflux_true=qflux_true,
            theta_true=theta_true,
            M1_Msun=M1,
            M2_Msun=M2,
            rEhat_AU=rEhat,
            P_grid=P_grid,
            fsource_true=1.0,
            fblend_true=0.0,
            override_xiE=None,               # Kepler-consistente
            set_flux_from_truth_photometry=True,
            rms_on_magnification=True,
            n_jobs=16,
        )
    
    print("scan on u0 finished (lambda_xi fixed to pi/2).")
