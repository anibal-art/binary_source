from degeneracy_fit import run_grid_and_save_npz_kepler
import numpy as np

# ============================================================
# time grid
# ============================================================
t = np.linspace(-500, 500, 5000)

# ============================================================
# "truth" PSPL params (base)
# ============================================================
t0_true = 50.0
tE_true = 173.0

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

for k, u0_true in enumerate(u0_grid):
    out_name = f"scan_kepler_u0_{k:03d}.npz"

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
    )

print("scan on u0 finished (lambda_xi fixed to pi/2).")
