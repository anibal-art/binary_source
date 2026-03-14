from degeneracy_fit import run_grid_and_save_npz_kepler
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
u0_true = 0.1

# ============================================================
# orbital / xallarap angles (fixed)
# ============================================================
phi_true = 0.0
theta_true = 0.0
qflux_true = 0.0
lambda_xi_fixed = 0.5 * np.pi

# ============================================================
# microlensing scale (fixed)
# ============================================================
rEhat_AU = 5.0

# ============================================================
# [BLOQUE AGREGADO 1]
# barrido físico en a_s/R_E y Mtot
# ============================================================
q_mass_fixed = 0.5  # q = M2/M1 fijo

a_over_RE_grid = np.logspace(-3, 0, 25)      # a_s / R_E
Mtot_grid = np.logspace(-1, 1, 30)           # 0.1 a 10 Msun
# Si querés incluir subestelares:
# Mtot_grid = np.logspace(np.log10(0.03), 1, 35)

# ============================================================
# scan in tE
# ============================================================
for tE_true in [50, 150, 500, 1000]:

    directory = f"/home/anibal-pc/binary_source/results/scan_aRE_Mtot_tE{int(tE_true)}/"
    os.makedirs(directory, exist_ok=True)

    for ia, a_over_RE in enumerate(a_over_RE_grid):
        a_sep_AU = float(a_over_RE * rEhat_AU)

        # órbita de la fuente brillante alrededor del CM
        a1_AU = (q_mass_fixed / (1.0 + q_mass_fixed)) * a_sep_AU
        xiE_override = a1_AU / rEhat_AU

        for im, Mtot in enumerate(Mtot_grid):
            # repartir masas manteniendo q = M2/M1 fijo
            M1 = Mtot / (1.0 + q_mass_fixed)
            M2 = q_mass_fixed * M1

            # Kepler:
            # P_yr^2 = a_AU^3 / Mtot_Msun
            P_yr = np.sqrt(a_sep_AU**3 / Mtot)
            P_day = 365.25 * P_yr

            out_name = directory + f"scan_aRE_{ia:03d}_Mtot_{im:03d}.npz"

            run_grid_and_save_npz_kepler(
                out_npz_path=out_name,
                t=t,
                t0_true=t0_true,
                u0_true=float(u0_true),
                tE_true=tE_true,
                phi_true=phi_true,
                i_true=float(lambda_xi_fixed),
                qflux_true=qflux_true,
                theta_true=theta_true,
                M1_Msun=float(M1),
                M2_Msun=float(M2),
                rEhat_AU=rEhat_AU,
                P_grid=np.array([float(P_day)]),   # un solo P por archivo
                fsource_true=1.0,
                fblend_true=0.0,
                override_xiE=float(xiE_override),
                set_flux_from_truth_photometry=True,
                rms_on_magnification=True,
            )

            # ====================================================
            # [BLOQUE AGREGADO 3]
            # guardar metadatos extra dentro del npz ya creado
            # ====================================================
            d = dict(np.load(out_name, allow_pickle=False))
            d["a_over_RE"] = np.array([float(a_over_RE)])
            d["Mtot_Msun"] = np.array([float(Mtot)])
            d["P_day_true"] = np.array([float(P_day)])
            d["q_mass_fixed"] = np.array([float(q_mass_fixed)])
            np.savez(out_name, **d)

    print(f"scan on (a_s/R_E, Mtot) finished for tE = {tE_true}")
