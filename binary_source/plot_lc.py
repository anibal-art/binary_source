import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy.optimize as so

from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.simulations import simulator

# =========================
# helpers
# =========================
def mag(zp, Flux):
    return zp - 2.5 * np.log10(np.abs(Flux))

def chi2_theoretical(fit_params, your_model, use_magnification=False,
                     fs_fixed=1.0, fsecond_fixed=1.0):
    """
    SSE sin pesos. Ajusta SOLO (t0,u0,tE). Flujos fijos.
    OJO: fsecond_fixed acá representa tu segundo parámetro de flujo (fb o ftotal según tu modelo).
    """
    fit_params = np.asarray(fit_params, dtype=float)
    full_params = np.concatenate([fit_params, [fs_fixed, fsecond_fixed]])
    py_params = your_model.compute_pyLIMA_parameters(full_params)

    sse = 0.0
    for tel in your_model.event.telescopes:
        if tel.lightcurve is None:
            continue

        data = tel.lightcurve["flux"].value

        if use_magnification:
            model_pred = your_model.model_magnification(tel, py_params)
        else:
            model_pred = your_model.compute_the_microlensing_model(tel, py_params)["photometry"]

        resid = data - model_pred
        sse += np.sum(resid**2)

    return float(sse)

def build_sim_event(t, mag0=19.0, emag=1e-9, filt="G"):
    ev = event.Event()
    ev.name = "Simulated"
    ev.ra = 170
    ev.dec = -70

    lc = np.c_[t, np.full_like(t, mag0), np.full_like(t, emag)]
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

# =========================
# MAIN: single-period reproduction
# =========================
def single_period_fit_and_plot(
    P_days: float,
    # PSPL base
    t0_true: float = 50.0,
    u0_true: float = 0.1,
    tE_true: float = 173.0,
    # xallarap geometry
    xiE: float = 0.1,              # default; si querés Kepler-consistente, calculalo afuera
    theta: float = 0.0,
    phi0: float = 0.0,
    lambda_xi: float = 0.5*np.pi,  # inclinación fija
    q_mass: float = 1.0,           # M2/M1 o M1/M2 según convención de pyLIMA; mantené consistente con tu uso
    qflux: float = 0.0,
    # photometry
    fs: float = 1.0,
    fsecond: float = 1.0,          # OJO: esto es fb o ftotal según tu configuración
    # isolate knobs
    override_xiE: float | None = None,
    window_k: float | None = None, # ej 5.0 para |t-t0|<5 tE
    out_prefix: str = "singleP",
):
    """
    Genera Xallarap truth para un solo P, ajusta PSPL y grafica.

    override_xiE:
      - None: usa xiE tal cual lo pasaste
      - float: fuerza xiE a ese valor (útil para aislar el efecto de omega)
    window_k:
      - None: plotea todo el rango temporal
      - float: plotea solo |t-t0| <= window_k * tE_true
    """
    # time grid
    t = np.linspace(-500, 500, 5000)

    # omega
    omega = 2.0*np.pi/float(P_days)

    # xiE override
    if override_xiE is not None:
        xiE_use = float(override_xiE)
    else:
        xiE_use = float(xiE)

    xi_para = xiE_use*np.cos(theta)
    xi_perp = xiE_use*np.sin(theta)

    # event
    ev = build_sim_event(t, mag0=19.0, emag=1e-9, filt="G")

    # -------- truth model (xallarap)
    model_xal = PSPL_model.PSPLmodel(ev, parallax=["None", 0.0], double_source=["Circular", t0_true])
    model_xal.define_model_parameters()

    params_xal = [
        t0_true, u0_true, tE_true,
        xi_para, xi_perp, omega,
        phi0, lambda_xi,
        q_mass, qflux,
        fs, fsecond,
    ]
    py_params_xal = model_xal.compute_pyLIMA_parameters(params_xal)

    # simulate + set theoretical flux data
    simulator.simulate_lightcurve(model_xal, py_params_xal)

    F_truth = model_xal.compute_the_microlensing_model(ev.telescopes[0], py_params_xal)["photometry"]
    ev.telescopes[0].lightcurve["flux"] = F_truth
    ev.telescopes[0].lightcurve["mag"] = mag(27.0, F_truth)

    # -------- PSPL fit model
    model_pspl = PSPL_model.PSPLmodel(ev, parallax=["None", 0.0], double_source=["None", 0.0])
    model_pspl.define_model_parameters()

    x0 = np.array([t0_true, u0_true, tE_true], dtype=float)
    res = so.minimize(
        chi2_theoretical,
        x0=x0,
        args=(model_pspl, False, fs, fsecond),  # fit en flujo
        method="Nelder-Mead",
        options=dict(maxiter=20000, xatol=1e-10, fatol=1e-10),
    )
    if not res.success:
        raise RuntimeError(f"Fit failed: {res.message}")

    best = np.asarray(res.x, dtype=float)

    # reconstruct fitted curves
    best_full = np.concatenate([best, [fs, fsecond]])
    py_best = model_pspl.compute_pyLIMA_parameters(best_full)

    F_fit = model_pspl.compute_the_microlensing_model(ev.telescopes[0], py_best)["photometry"]
    A_truth = model_xal.model_magnification(ev.telescopes[0], py_params_xal)
    A_fit   = model_pspl.model_magnification(ev.telescopes[0], py_best)

    # residuals
    resid_F = F_truth - F_fit
    resid_A = A_truth - A_fit
    
    # optional window
    if window_k is not None:
        mask = np.abs(t - t0_true) <= float(window_k)*tE_true
    else:
        mask = np.ones_like(t, dtype=bool)

    rms_F = float(np.sqrt(np.mean((resid_F[mask])**2)))
    rms_A = float(np.sqrt(np.mean((resid_A[mask])**2)))

    # =========================
    # plots: FLUX + residual
    # =========================
    #fig = plt.figure(figsize=(7.2, 5.2))
    #gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.06)
    #ax = fig.add_subplot(gs[0, 0])
    #axr = fig.add_subplot(gs[1, 0], sharex=ax)

    #ax.plot(t[mask], F_truth[mask], label="Xallarap (truth)")
    #ax.plot(t[mask], F_fit[mask], "--", label="PSPL fit")
    #ax.set_ylabel("Flux")
    #ax.legend(frameon=False, loc="best")

    #ax.text(
     #   0.02, 0.95,
     #   rf"$P={P_days:.2f}\,\mathrm{{d}}$" + "\n" + rf"$\mathrm{{RMS}}_F={rms_F:.3e}$",
      #  transform=ax.transAxes, va="top", ha="left",
       # bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
    #)

    #axr.plot(t[mask], resid_F[mask])
    #axr.axhline(0.0, linestyle=":", linewidth=1.0)
    #axr.set_xlabel("Time [days]")
    #axr.set_ylabel("Residual")
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    #axr.yaxis.set_minor_locator(AutoMinorLocator())
    #axr.xaxis.set_minor_locator(AutoMinorLocator())
    #plt.setp(ax.get_xticklabels(), visible=False)

    #fig.tight_layout()
    #fig.savefig(f"{out_prefix}_flux.pdf", bbox_inches="tight")
    #fig.savefig(f"{out_prefix}_flux.png", bbox_inches="tight")
    #plt.show()

    # =========================
    # plots: MAGNIFICATION + residual
    # =========================
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.plot(t[mask], np.log10(A_truth[mask]), label="Xallarap (truth)")
    ax.plot(t[mask], np.log10(A_fit[mask]), "--", label="PSPL fit")
    ax.set_ylabel(r"Magnification $A(t)$")
    ax.legend(frameon=False, loc="best")
    INT_L1 = float(np.trapz(np.abs(resid_A[mask]), t[mask]))
    ax.text(
        0.02, 0.95,
        rf"$P={P_days:.2f}\,\mathrm{{d}}$" + "\n" + rf"$\mathrm{{RMS}}_A={rms_A:.3e}$"+"\n"+f"Int L1= {round(INT_L1,3)}"+"\n"+r"MAX $|F_\xi-F_{PSPL}|$="+f" {round(max(abs(resid_F)),3)}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
    )

    axr.plot(t[mask], resid_A[mask])
    axr.axhline(0.0, linestyle=":", linewidth=1.0)
    axr.set_xlabel("Time [days]")
    axr.set_ylabel(r"$\Delta A$")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    axr.yaxis.set_minor_locator(AutoMinorLocator())
    axr.xaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), visible=False)
   
    
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_mag.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}_mag.png", bbox_inches="tight")
    plt.show()

    print("Best-fit (t0,u0,tE) =", best)
    print("RMS_F =", rms_F)
    print("RMS_A =", rms_A)

    return dict(
        t=t,
        mask=mask,
        best=best,
        F_truth=F_truth,
        F_fit=F_fit,
        A_truth=A_truth,
        A_fit=A_fit,
        RMS_F=rms_F,
        RMS_A=rms_A,
    )

# =========================
# EJEMPLO DE USO (elegí tu P)
# =========================
# Ej: aislar un solo período
#%matplotlib inline
#plt.close("all")
#out = single_period_fit_and_plot(
#    P_days=173/2,
#    t0_true=50.0, u0_true=0.1, tE_true=173.0,
#    xiE=0.1,            # poné tu valor
#    theta=0.0, phi0=0.0,
#    lambda_xi=0.5*np.pi,
#    q_mass=1.0, qflux=0.0,
#    fs=1.0,
#    fsecond=1.0,        # OJO: si tu modelo usa ftotal, esto debería ser ftotal
#    override_xiE=None,  # o override_xiE=0.1 para forzar amplitud fija
#    window_k=5.0,       # None para todo el rango
#    out_prefix="singleP_210d",
#)

import argparse
#import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--P_days", type=float, required=True)
    parser.add_argument("--t0_true", type=float, default=50.0)
    parser.add_argument("--u0_true", type=float, default=0.1)
    parser.add_argument("--tE_true", type=float, default=173.0)

    parser.add_argument("--xiE", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--lambda_xi", type=float, default=0.5*np.pi)

    parser.add_argument("--q_mass", type=float, default=1.0)
    parser.add_argument("--qflux", type=float, default=0.0)

    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--fsecond", type=float, default=1.0)

    parser.add_argument("--window_k", type=float, default=5.0)

    parser.add_argument("--out_prefix", type=str, default="singleP")

    args = parser.parse_args()

    out = single_period_fit_and_plot(
        P_days=args.P_days,
        t0_true=args.t0_true,
        u0_true=args.u0_true,
        tE_true=args.tE_true,
        xiE=args.xiE,
        theta=args.theta,
        phi0=args.phi0,
        lambda_xi=args.lambda_xi,
        q_mass=args.q_mass,
        qflux=args.qflux,
        fs=args.fs,
        fsecond=args.fsecond,
        window_k=args.window_k,
        out_prefix=args.out_prefix,
    )
