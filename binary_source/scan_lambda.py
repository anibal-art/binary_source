
from degeneracy_fit import run_grid_and_save_npz_kepler
import numpy as np

# time
t = np.linspace(-500, 500, 5000)

# PSPL base
t0_true = 50.0
u0_true = 0.1
tE_true = 173.0

# Parámeters orbitals
phi_true = 0.0
theta_true = 0.0
qflux_true = 0.0

M1 = 2.0
M2 = 1.0
rEhat = 5.0
P_grid = np.logspace(1, 5, 60)   # 10 d → 100000 d
N_lambda = 25  # fix resolution
lambda_xi_grid = np.linspace(0.0, 0.5*np.pi, N_lambda)

for k, lambda_xi in enumerate(lambda_xi_grid):
    out_name = f"scan_kepler_lambda_{k:03d}.npz"

    run_grid_and_save_npz_kepler(
        out_npz_path=out_name,
        t=t,
        t0_true=t0_true,
        u0_true=u0_true,
        tE_true=tE_true,
        phi_true=phi_true,
        i_true=float(lambda_xi),      # <-- [CAMBIO CLAVE] inclinación = lambda_xi
        qflux_true=qflux_true,
        theta_true=theta_true,
        M1_Msun=M1,
        M2_Msun=M2,
        rEhat_AU=rEhat,
        P_grid=P_grid,
        fsource_true=1.0,
        fblend_true=0.0,
        override_xiE=None,            # Kepler-consistente
        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,
    )

print("scan on lambda_xi finished.")
