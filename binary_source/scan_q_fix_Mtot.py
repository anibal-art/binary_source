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
lambda_xi_fixed = 0.5 * np.pi   # pi/2

# ============================================================
# physical system
# ============================================================
Mtot_fixed = 3.0   # Msun  <-- fijo
rEhat = 5.0        # AU

# ============================================================
# scan in P
# ============================================================
P_grid = np.logspace(1, 5, 60)   # 10 d -> 100000 d

# ============================================================
# scan in q = M2/M1
# ============================================================
N_q = 25
q_grid = np.logspace(-2, 1, N_q)   # 0.01 -> 10

# ============================================================
# scan over tE and q
# ============================================================
for tE_true in [50, 150, 500, 1000]:
    directory = f"/home/anibal/binary_source/results/scan_q_Mtotfixed_tE{int(tE_true)}/"
    os.makedirs(directory, exist_ok=True)

    for k, q_true in enumerate(q_grid):
        M1_true = Mtot_fixed / (1.0 + q_true)
        M2_true = Mtot_fixed * q_true / (1.0 + q_true)

        out_name = os.path.join(directory, f"scan_kepler_q_{k:03d}.npz")

        run_grid_and_save_npz_kepler(
            out_npz_path=out_name,
            t=t,
            t0_true=t0_true,
            u0_true=float(u0_true),
            tE_true=float(tE_true),
            phi_true=phi_true,
            i_true=float(lambda_xi_fixed),
            qflux_true=qflux_true,
            theta_true=theta_true,
            M1_Msun=float(M1_true),
            M2_Msun=float(M2_true),
            rEhat_AU=float(rEhat),
            P_grid=P_grid,
            fsource_true=1.0,
            fblend_true=0.0,
            override_xiE=None,               # Kepler-consistente
            set_flux_from_truth_photometry=True,
            rms_on_magnification=True,
        )

    print(f"scan on mass ratio with fixed Mtot finished for tE={tE_true}")