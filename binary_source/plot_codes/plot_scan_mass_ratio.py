import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

pattern = "scan_kepler_q_*.npz"
files = sorted(glob.glob(pattern))

if len(files) == 0:
    raise FileNotFoundError("No se encontraron archivos scan_kepler_q_*.npz")

data_list = []
q_values = []

for fn in files:
    d = np.load(fn, allow_pickle=False)

    P_grid = d["P_grid"].astype(float)
    RMS = d["RMS"].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)
    truth = d["truth"].astype(float)

    M1 = truth[5]
    M2 = truth[6]
    q = float(M2 / M1)

    m = SUCCESS & np.isfinite(RMS) & np.isfinite(P_grid)
    if not np.any(m):
        continue

    idx = np.argsort(P_grid[m])
    Pp = P_grid[m][idx]
    Rp = RMS[m][idx]

    q_values.append(q)
    data_list.append((q, Pp, Rp))

q_values = np.array(q_values)

# -------------------------------------------------
# Colormap logarítmico (MUY recomendable para q)
# -------------------------------------------------
norm = colors.LogNorm(vmin=np.min(q_values), vmax=np.max(q_values))
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(8,5))

for q, Pp, Rp in data_list:
    color = cmap(norm(q))
    ax.semilogx(Pp, Rp, color=color, linewidth=1.5)

ax.set_xlabel(r"$P$ [days]")
ax.set_ylabel(r"RMS (magnification)")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_title(r"RMS($P$) for different mass ratios $q = M_2/M_1$")

# Colorbar logarítmica
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$q = M_2/M_1$")

plt.tight_layout()
plt.show()