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
u0_true = 0.1   # <-- fijo para este mapa; cambialo si querés otro valor

# ============================================================
# orbital / xallarap angles (fixed)
# ============================================================
phi_true = 0.0
theta_true = 0.0
qflux_true = 0.0
lambda_xi_fixed = 0.5 * np.pi   # fijo: pi/2

# ============================================================
# physical microlensing scale (fixed)
# ============================================================
rEhat = 5.0   # AU

# ============================================================
# [BLOQUE AGREGADO 1]
# fixed source mass ratio and grids in a_s/R_E and P
# ============================================================
q_mass_fixed = 0.5   # q = M2/M1 fijo

# grid in total binary-source separation in units of Einstein radius
a_over_RE_grid = np.logspace(-3, 0, 25)   # por ejemplo: 1e-3 -> 1

# grid in orbital period [days]
P_grid = np.logspace(1, 5, 60)   # 10 d -> 100000 d

# ============================================================
# scan in tE
# ============================================================
for tE_true in [50, 150, 500, 1000]:

    directory = f"/home/anibal-pc/binary_source/results/scan_aRE_P_tE{int(tE_true)}/"
    os.makedirs(directory, exist_ok=True)

    # ========================================================
    # [BLOQUE AGREGADO 2]
    # nested scan in a_s/R_E and P, computing Mtot from Kepler
    # ========================================================
    for ia, a_over_RE in enumerate(a_over_RE_grid):

        # total separation a_s in AU
        a_sep_AU = a_over_RE * rEhat

        # xallarap amplitude for the bright source orbit:
        # a1 = (q / (1+q)) * a_s
        a1_AU = (q_mass_fixed / (1.0 + q_mass_fixed)) * a_sep_AU

        # xi_E = a1 / rEhat
        xiE_override = a1_AU / rEhat

        for ip, P_day in enumerate(P_grid):

            # convert period to years
            P_yr = P_day / 365.25

            # ====================================================
            # [BLOQUE AGREGADO 3]
            # Kepler-consistent total mass in solar masses:
            # P^2 = a^3 / Mtot   <=>   Mtot = a^3 / P^2
            # with a in AU, P in yr, M in Msun
            # ====================================================
            Mtot = a_sep_AU**3 / (P_yr**2)

            # split masses keeping q = M2/M1 fixed
            M1 = Mtot / (1.0 + q_mass_fixed)
            M2 = q_mass_fixed * M1

            out_name = directory + f"scan_aRE_{ia:03d}_P_{ip:03d}.npz"

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
                rEhat_AU=rEhat,
                P_grid=np.array([float(P_day)]),   # <-- un solo P por archivo
                fsource_true=1.0,
                fblend_true=0.0,
                override_xiE=float(xiE_override),  # <-- consistente con a_s/R_E y q
                set_flux_from_truth_photometry=True,
                rms_on_magnification=True,
            )

    print(f"scan on (a_s/R_E, P/tE) finished for tE = {tE_true}")
