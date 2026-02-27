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
u0_true = 0.1        # <-- fijamos u0 (como hiciste al fijar lambda)
tE_true = 173.0

# ============================================================
# orbital / xallarap angles (fixed)
# ============================================================
phi_true = 0.0
theta_true = 0.0
qflux_true = 0.0
lambda_xi_fixed = 0.5*np.pi   # fijo: pi/2

# ============================================================
# physical system (fixed except mass ratio)
# ============================================================
M1 = 2.0
rEhat = 5.0

# scan in P as before
P_grid = np.logspace(1, 5, 60)  # 10 d → 100000 d

# ============================================================
# [NUEVO] scan in mass ratio q = M2/M1
# ============================================================
N_q = 25

# Elegí un rango razonable:
# - si querés planetario: q ~ 1e-5 .. 1e-2
# - si querés estelar:    q ~ 0.1 .. 10
q_grid = np.logspace(-3, 1, N_q)   # ejemplo: 1e-3 → 10

for k, q in enumerate(q_grid):
    M2 = float(q) * float(M1)
    out_name = f"scan_kepler_q_{k:03d}.npz"

    run_grid_and_save_npz_kepler(
        out_npz_path=out_name,
        t=t,
        t0_true=t0_true,
        u0_true=float(u0_true),          # <-- fijo
        tE_true=tE_true,
        phi_true=phi_true,
        i_true=float(lambda_xi_fixed),   # <-- fijo: pi/2
        qflux_true=qflux_true,
        theta_true=theta_true,
        M1_Msun=float(M1),
        M2_Msun=float(M2),               # <-- [CAMBIO CLAVE] barrido en M2 => q
        rEhat_AU=float(rEhat),
        P_grid=P_grid,
        fsource_true=1.0,
        fblend_true=0.0,
        override_xiE=None,               # Kepler-consistente
        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,
    )

print("scan on mass ratio finished (lambda_xi fixed to pi/2).")
