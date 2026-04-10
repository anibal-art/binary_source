import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})
# ============================================================
# Parámetros físicos (IMPORTANTES para xiE)
# ============================================================
tE_true = 150.0
Mtot_fixed = 3.0
rEhat_AU = 5.0

directory = f"/home/anibal/binary_source/results/scan_q_Mtotfixed_tE{int(tE_true)}/"
pattern = os.path.join(directory, "scan_kepler_q_*.npz")

metric_key = "RMS"

files = sorted(glob.glob(pattern))

# ============================================================
# Funciones P/tE <-> xiE
# ============================================================
def xiE_from_PoverTE(P_over_tE):
    P_days = P_over_tE * tE_true
    P_yr = P_days / 365.25
    a_AU = (Mtot_fixed * P_yr**2)**(1/3)
    return a_AU / rEhat_AU

def PoverTE_from_xiE(xiE):
    a_AU = xiE * rEhat_AU
    P_yr = np.sqrt(a_AU**3 / Mtot_fixed)
    P_days = 365.25 * P_yr
    return P_days / tE_true

# ============================================================
# Parseo q
# ============================================================
def extract_q_index(filename):
    base = os.path.basename(filename)
    m = re.match(r"scan_kepler_q_(\d+)\.npz", base)
    return int(m.group(1))

# ============================================================
# Bordes log
# ============================================================
def log_bin_edges(x):
    lx = np.log10(x)
    edges = np.empty(len(x)+1)
    edges[1:-1] = 0.5*(lx[:-1] + lx[1:])
    edges[0] = lx[0] - 0.5*(lx[1]-lx[0])
    edges[-1] = lx[-1] + 0.5*(lx[-1]-lx[-2])
    return 10**edges

# ============================================================
# Leer grilla
# ============================================================
q_dict = {}
P_grid_ref = None

for fn in files:
    k = extract_q_index(fn)
    d = np.load(fn)

    truth = d["truth"]
    M1, M2 = truth[5], truth[6]
    q = M2 / M1

    P_grid = d["P_grid"]

    if P_grid_ref is None:
        P_grid_ref = P_grid.copy()

    q_dict[k] = q

sorted_indices = sorted(q_dict.keys())
q_grid = np.array([q_dict[k] for k in sorted_indices])

Nq = len(q_grid)
NP = len(P_grid_ref)

RMS_map = np.full((Nq, NP), np.nan)

index_to_row = {k: i for i, k in enumerate(sorted_indices)}

for fn in files:
    k = extract_q_index(fn)
    row = index_to_row[k]

    d = np.load(fn)

    metric = d[metric_key]
    SUCCESS = d["SUCCESS"]

    m = SUCCESS & np.isfinite(metric) & (metric > 0)

    RMS_map[row, m] = metric[m]

# ============================================================
# Ejes
# ============================================================
P_over_tE = P_grid_ref / tE_true

q_edges = log_bin_edges(q_grid)
P_edges = log_bin_edges(P_over_tE)

# ============================================================
# Colormap
# ============================================================
positive = RMS_map[np.isfinite(RMS_map) & (RMS_map > 0)]

vmin = np.percentile(positive, 5)
vmax = np.percentile(positive, 95)

norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

pcm = ax.pcolormesh(
    q_edges,
    P_edges,
    np.ma.masked_invalid(RMS_map).T,
    cmap="viridis",
    norm=norm,
    shading="auto"
)

# ============================================================
# Contornos RMS
# ============================================================
Q2d, P2d = np.meshgrid(q_grid, P_over_tE, indexing="xy")

levels = np.geomspace(vmin, vmax, 5)

cs = ax.contour(Q2d, P2d, RMS_map.T, levels=levels, colors="white")
ax.clabel(cs, fmt=lambda x: f"{x:.1e}")

# ============================================================
# Detectabilidad
# ============================================================
threshold = 1e-2

cs_det = ax.contour(
    Q2d, P2d, RMS_map.T,
    levels=[threshold],
    colors="cyan",
    linewidths=2,
    linestyles="--"
)

ax.clabel(cs_det, fmt={threshold: r"$10^{-2}$"})

# ============================================================
# EJE DERECHO (xiE)
# ============================================================
secax = ax.secondary_yaxis(
    'right',
    functions=(xiE_from_PoverTE, PoverTE_from_xiE)
)

secax.set_ylabel(r"$\xi_E = \frac{a_s}{\hat{r}_E}$", fontsize=16, labelpad=8)
secax.set_yscale("log")
secax.tick_params(axis='y', which='both', pad=4)

# ============================================================
# Formato
# ============================================================
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$q=M_2/M_1$", fontsize=18)
ax.set_ylabel(r"$P/t_E$", fontsize=18)

ax.grid(True, which="both", alpha=0.25)

# dejar más espacio a la derecha para el eje secundario
fig.subplots_adjust(right=0.82)
plt.title("$M_{total} = 3M_{\odot},\ t_E=$"+f"{tE_true} days"+"$,\ u_0$=0.1",fontsize=20)
# colorbar más separado
cbar = fig.colorbar(pcm, ax=ax, pad=0.2)
cbar.set_label(r"RMS residual magnification", fontsize=14)

plt.show()