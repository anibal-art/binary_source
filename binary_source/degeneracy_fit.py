import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os, sys
current_path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_path, os.pardir))
print("Parent Directory:", parent_directory)
sys.path.append(parent_directory)

# from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
# from pyLIMA.simulations import simulator
# from pyLIMA.fits import TRF_fit
# from astropy import units as u
# from astropy import constants as C
# from pyLIMA.xallarap.xallarap import xallarap_shifts, compute_xallarap_curvature

import scipy.optimize as so
# from scipy.interpolate import interp1d
from functions_aux import mag,flux_to_mag, sigma_W149_func, mag_to_flux, sigma_flux_from_sigma_mag, sigma_flux_from_flux, orbital_period_kepler, chi2_theoretical, build_sim_event, a_from_P_kepler_days

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
    # objective config (kept for compatibility; example uses flux fit)
    use_magnification_fit: bool = False,
    # storage
    store_curves: bool = True,
    # ============================================================
    # [NUEVO BLOQUE] para replicar tu ejemplo
    # ============================================================
    override_xiE: float | None = None,   # si no es None, usa este xiE en vez de Kepler
    set_flux_from_truth_photometry: bool = True,  # igual que tu ejemplo: data = F_truth exacto
    rms_on_magnification: bool = True,   # igual que tu chequeo final: RMS(A_truth - A_fit)
):
    """
    Igual que tu ejemplo:
      1) Genera el modelo xallarap "verdadero".
      2) Construye data teórica poniendo telescope.lightcurve['flux'] = F_truth (photometry) exacto.
      3) Ajusta PSPL variando [t0,u0,tE] minimizando SSE en flujo.
      4) Reporta RMS en magnificación (por defecto), como hiciste al final.

    Si override_xiE is not None:
      - usa xiE fijo (como a/rEhat), y P solo define omega.
      - esto reproduce el estilo "a fijo, P kepleriano" de tu ejemplo puntual.
    """
    #ftotal_true = fsource_true + fblend_true
    P_grid = np.asarray(P_grid, dtype=float)

    # ============================================================
    # [NUEVO BLOQUE] q_mass consistente y Mtot
    # ============================================================
    q_mass_true = float(M2_Msun / M1_Msun)   # <-- M2/M1 (NO invertido)
    Mtot_Msun = float(M1_Msun + M2_Msun)

    n_P = len(P_grid)
    n_t = len(t)

    RMS     = np.full(n_P, np.nan, dtype=float)
    MAXABS  = np.full(n_P, np.nan, dtype=float)
    DT0     = np.full(n_P, np.nan, dtype=float)
    DU0     = np.full(n_P, np.nan, dtype=float)
    DTE     = np.full(n_P, np.nan, dtype=float)
    SUCCESS = np.zeros(n_P, dtype=bool)
    Q_A = np.full(n_P, np.nan, dtype=float)

    BEST_T0U0TE = np.full((n_P, 3), np.nan, dtype=float)
    XI_E = np.full(n_P, np.nan, dtype=float)
    A_AU = np.full(n_P, np.nan, dtype=float)

    if store_curves:
        A_truth_grid = np.full((n_P, n_t), np.nan, dtype=np.float32)
        A_fit_grid   = np.full((n_P, n_t), np.nan, dtype=np.float32)
        F_truth_grid = np.full((n_P, n_t), np.nan, dtype=np.float32)  # [NUEVO] para depurar como en tu ejemplo
        F_fit_grid   = np.full((n_P, n_t), np.nan, dtype=np.float32)  # [NUEVO]

    for j_P, P in enumerate(P_grid):
        try:
            omega = 2.0*np.pi / P

            # ============================================================
            # [NUEVO BLOQUE] xiE: Kepler o override (estilo tu ejemplo)
            # ============================================================
            if override_xiE is None:
                a_AU = a_from_P_kepler_days(P, Mtot_Msun)
                xiE = a_AU / float(rEhat_AU)
            else:
                xiE = float(override_xiE)
                a_AU = xiE * float(rEhat_AU)  # valor "equivalente" solo para guardar

            A_AU[j_P] = a_AU
            XI_E[j_P] = xiE

            xi_para = xiE * np.cos(theta_true)
            xi_perp = xiE * np.sin(theta_true)

            # Event fresco (como tu ejemplo)
            ev = build_sim_event(t, mag0=19.0, emag=0.01, filt="G")

            # ---------- "Verdad" xallarap (DSPL circular)
            model_xal = PSPL_model.PSPLmodel(
                ev, parallax=["None", 0.0], double_source=["Circular", t0_true]
            )
            model_xal.define_model_parameters()

            ZP =27.615
            fsource_true = mag_to_flux(msource_true, zp=ZP)
            ftotal_true =mag_to_flux(mtotal_true, zp=ZP)
            fblend_true=0


            params_xal = [
                t0_true, u0_true, tE_true,
                xi_para, xi_perp, omega,
                phi_true, i_true,
                q_mass_true, qflux_true,
                fsource_true, ftotal_true,
            ]
            py_params_xal = model_xal.compute_pyLIMA_parameters(params_xal)

            # Curva verdadera en magnificación
            A_truth = model_xal.model_magnification(ev.telescopes[0], py_params_xal)

            # ============================================================
            # [NUEVO BLOQUE] construir "data" exactamente como tu ejemplo
            # ============================================================
            if set_flux_from_truth_photometry:
                # Esto replica tu:
                # model_flux = model_xiE.compute_the_microlensing_model(... )['photometry']
                # telescope.lightcurve['flux'] = model_flux
                F_truth = model_xal.compute_the_microlensing_model(ev.telescopes[0], py_params_xal)['photometry']
                ev.telescopes[0].lightcurve['flux'] = F_truth
            else:
                # Alternativa: si querés data=A(t) (no es tu ejemplo)
                if use_magnification_fit:
                    ev.telescopes[0].lightcurve['flux'] = A_truth
                else:
                    F_truth = model_xal.compute_the_microlensing_model(ev.telescopes[0], py_params_xal)['photometry']
                    ev.telescopes[0].lightcurve['flux'] = F_truth

            # ---------- Modelo PSPL para el fit (sin xallarap)
            model_pspl = PSPL_model.PSPLmodel(
                ev, parallax=["None", 0.0], double_source=["None", 0.0]
            )
            model_pspl.define_model_parameters()

            # Fit solo [t0,u0,tE], igual que tu ejemplo
            x0 = np.array([t0_true, u0_true, tE_true], dtype=float)
            res = so.minimize(
                chi2_theoretical,
                x0=x0,
                args=(model_pspl, False, fsource_true, ftotal_true),  # <-- [CAMBIO] fit en flujo como tu ejemplo
                method="Nelder-Mead",
                options=dict(maxiter=20000, xatol=1e-10, fatol=1e-6),
            )
            if not res.success:
                SUCCESS[j_P] = False
                continue

            best = np.asarray(res.x, dtype=float)
            BEST_T0U0TE[j_P, :] = best

            # Reconstrucción del PSPL con flujos fijos (fs, ftotal)
            best_full = np.concatenate([best, [fsource_true, ftotal_true]])
            py_params_best = model_pspl.compute_pyLIMA_parameters(best_full)

            # Curvas PSPL resultantes
            A_fit = model_pspl.model_magnification(ev.telescopes[0], py_params_best)
            F_fit = model_pspl.compute_the_microlensing_model(ev.telescopes[0], py_params_best)['photometry']

            # ============================================================
            # [NUEVO BLOQUE] RMS igual a tu chequeo final: en magnificación
            # ============================================================
            if rms_on_magnification:
                resid = A_truth - A_fit
            else:
                # si quisieras RMS en flujo
                resid = ev.telescopes[0].lightcurve['flux'].value - F_fit

            mag_truth = flux_to_mag(F_truth)
            mag_fit   = flux_to_mag(F_fit)
            
            mask = ((mag_truth >= 12.0) & (mag_truth <= 27.0)
                & (mag_fit   >= 12.0) & (mag_fit   <= 27.0)
            )
            F_truth_m = F_truth[mask]
            F_fit_m   = F_fit[mask]
            
            err_truth = sigma_flux_from_flux(F_truth_m)
            err_fit   = sigma_flux_from_flux(F_fit_m)
            residuals = F_truth_m - F_fit_m
            
            q_A = np.sum((residuals / err_truth)**2) / (len(residuals))
            Q_A[j_P] = q_A
            


            RMS[j_P]    = float(np.sqrt(np.mean(resid**2)))
            MAXABS[j_P] = float(np.max(np.abs(resid)))

            DT0[j_P] = best[0] - t0_true
            DU0[j_P] = best[1] - u0_true
            DTE[j_P] = best[2] - tE_true

            if store_curves:
                A_truth_grid[j_P, :] = np.asarray(A_truth, dtype=np.float32)
                A_fit_grid[j_P, :]   = np.asarray(A_fit, dtype=np.float32)
                F_truth_grid[j_P, :] = np.asarray(ev.telescopes[0].lightcurve['flux'].value, dtype=np.float32)
                F_fit_grid[j_P, :]   = np.asarray(F_fit, dtype=np.float32)

            SUCCESS[j_P] = True

        except Exception:
            SUCCESS[j_P] = False
            continue

    payload = dict(
        t=t,
        P_grid=P_grid,
        xiE_of_P=XI_E,
        a_AU_of_P=A_AU,
        RMS=RMS,
        MAXABS=MAXABS,
        DT0=DT0,
        DU0=DU0,
        DTE=DTE,
        SUCCESS=SUCCESS,
        BEST_T0U0TE=BEST_T0U0TE,
        Q_A=np.sqrt(Q_A),
        truth=np.array([t0_true, u0_true, tE_true, phi_true, i_true,
                        M1_Msun, M2_Msun, rEhat_AU, qflux_true, theta_true,
                        fsource_true, fblend_true,
                        float(use_magnification_fit),
                        -1.0 if override_xiE is None else float(override_xiE),
                        float(set_flux_from_truth_photometry),
                        float(rms_on_magnification)], dtype=float),
    )

    if store_curves:
        payload["A_truth_grid"] = A_truth_grid
        payload["A_fit_grid"]   = A_fit_grid
        payload["F_truth_grid"] = F_truth_grid
        payload["F_fit_grid"]   = F_fit_grid

    np.savez_compressed(out_npz_path, **payload)
    print(f"Saved: {out_npz_path}")

    
import numpy as np

# Tiempo


# PSPL base
t0_true = 50.0
u0_true = 0.1
tE_true = 173.0
t = np.linspace(-3.5*tE_true , 3.5*tE_true , 10000)
# Parámetros orbitales
phi_true = 0.0
i_true = np.pi/2
theta_true = 0.0
qflux_true = 0.0

# Sistema físico
M1 = 2.0
M2 = 1.0
rEhat = 5.0
P_grid = np.logspace(1, 5, 60)   # 10 días → 100000 días


# P_grid = np.linspace(10.0, 100000.0, 100)
# P0 = orbital_period_kepler(50.0, 3.0).value   # si tu orbital_period_kepler recibe AU y Msun y devuelve días

# run_grid_and_save_npz_kepler(
#     out_npz_path="single_point_like_example.npz",
#     t=t,
#     t0_true=50.0, u0_true=0.1, tE_true=173.0,
#     phi_true=0.0, i_true=lambda_xi, qflux_true=0.0, theta_true=0.0,
#     M1_Msun=2.0, M2_Msun=1.0, rEhat_AU=5.0,
#     P_grid=np.array([P0]),
#     fsource_true=1.0, fblend_true=0.0,
#     override_xiE=10.0,                      # <-- clave para “hacer lo mismo”
#     set_flux_from_truth_photometry=True,    # <-- clave
#     rms_on_magnification=True,              # <-- RMS como tu chequeo
# )


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
    msource_true= 24.0,
    mtotal_true= 24.0,
    override_xiE=None,  # <-- importante: Kepler-consistente real
)
