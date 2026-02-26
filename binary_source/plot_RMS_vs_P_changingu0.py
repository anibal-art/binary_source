import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

pattern = "scan_kepler_u0_*.npz"
files = sorted(glob.glob(pattern))
if len(files) == 0:
    raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")

# -------------------------------------------------
# Cargar datos primero para conocer rango de u0
# -------------------------------------------------
data_list = []
u0_values = []

for fn in files:
    d = np.load(fn, allow_pickle=False)

    P_grid = d["P_grid"].astype(float)
    RMS = d["RMS"].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)
    truth = d["truth"].astype(float)

    u0_true = float(truth[1])  # <-- u0

    m = SUCCESS & np.isfinite(RMS) & np.isfinite(P_grid)
    if not np.any(m):
        continue

    idx = np.argsort(P_grid[m])
    Pp = P_grid[m][idx]
    Rp = RMS[m][idx]

    u0_values.append(u0_true)
    data_list.append((u0_true, Pp, Rp))

u0_values = np.array(u0_values, dtype=float)
if u0_values.size == 0:
    raise RuntimeError("No hay curvas válidas (SUCCESS & finitos) para graficar.")

# -------------------------------------------------
# Colormap + normalización (usa rango real de u0)
# -------------------------------------------------
norm = colors.Normalize(vmin=float(np.min(u0_values)), vmax=float(np.max(u0_values)))
cmap = cm.plasma  # buen contraste

fig, ax = plt.subplots(figsize=(8, 5))

for u0_true, Pp, Rp in data_list:
    color = cmap(norm(u0_true))
    # opcional: grosor aumenta con u0 para marcar dirección
    lw = 1.0 + 1.5 * (u0_true - norm.vmin) / (norm.vmax - norm.vmin + 1e-30)

    ax.semilogx(Pp, Rp, color=color, linewidth=lw)

ax.set_xlabel(r"$P$ [days]")
ax.set_ylabel(r"RMS (magnification)")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_title(r"Kepler-consistent scan: RMS($P$) colored by $u_0$")

# Colorbar asociada al eje
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$u_0$")

plt.tight_layout()
plt.show()
