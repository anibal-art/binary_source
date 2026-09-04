import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ============================================================
# [NUEVO BLOQUE 1]
# parámetros del gráfico
# ============================================================
tE_true = 150.0
directory = f"/home/anibal-pc/binary_source/results/scan_q_tE{int(tE_true)}/"
pattern = os.path.join(directory, "scan_kepler_q_*.npz")
metric_key = "RMS"   # o "MAXABS"
files = sorted(glob.glob(pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")

# ============================================================
# [NUEVO BLOQUE 2]
# parseo del índice q desde el nombre del archivo
# ============================================================
def extract_q_index(filename):
    """
    Extrae k desde:
    scan_kepler_q_003.npz
    """
    base = os.path.basename(filename)
    m = re.match(r"scan_kepler_q_(\d+)\.npz", base)
    if m is None:
        raise ValueError(f"No pude parsear el índice desde: {base}")
    return int(m.group(1))

# ============================================================
# [NUEVO BLOQUE 3]
# función para bordes logarítmicos
# ============================================================
def log_bin_edges(x):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("Todos los valores deben ser positivos.")
    lx = np.log10(x)
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (lx[:-1] + lx[1:])
    edges[0] = lx[0] - 0.5 * (lx[1] - lx[0])
    edges[-1] = lx[-1] + 0.5 * (lx[-1] - lx[-2])
    return 10**edges

# ============================================================
# [NUEVO BLOQUE 4]
# reconstrucción de q_grid desde los archivos
# ============================================================
q_dict = {}
P_grid_ref = None

for fn in files:
    k = extract_q_index(fn)
    d = np.load(fn, allow_pickle=False)

    truth = d["truth"].astype(float)
    M1_true = float(truth[5])
    M2_true = float(truth[6])
    q_true = M2_true / M1_true

    P_grid = d["P_grid"].astype(float)

    if P_grid_ref is None:
        P_grid_ref = P_grid.copy()
    else:
        if len(P_grid) != len(P_grid_ref) or not np.allclose(P_grid, P_grid_ref):
            raise ValueError(f"P_grid no coincide entre archivos. Problema en {fn}")

    q_dict[k] = q_true

if len(q_dict) == 0:
    raise RuntimeError("No se pudo reconstruir q_grid desde los archivos.")

sorted_indices = np.array(sorted(q_dict.keys()), dtype=int)
q_grid = np.array([q_dict[k] for k in sorted_indices], dtype=float)

Nq = len(q_grid)
NP = len(P_grid_ref)

# ============================================================
# [NUEVO BLOQUE 5]
# construcción del mapa RMS(q, P/tE)
# ============================================================
RMS_map = np.full((Nq, NP), np.nan, dtype=float)

index_to_row = {k: i for i, k in enumerate(sorted_indices)}

for fn in files:
    k = extract_q_index(fn)
    row = index_to_row[k]

    d = np.load(fn, allow_pickle=False)

    P_grid = d["P_grid"].astype(float)
    metric = d[metric_key].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)

    m = SUCCESS & np.isfinite(P_grid) & np.isfinite(metric) & (metric > 0)
    if not np.any(m):
        continue

    RMS_map[row, m] = metric[m]

# ============================================================
# [NUEVO BLOQUE 6]
# eje vertical en P/tE
# ============================================================
P_over_tE_grid = P_grid_ref / tE_true

# ============================================================
# [NUEVO BLOQUE 7]
# bordes para pcolormesh
# x = q (log), y = P/tE (log)
# ============================================================
q_edges = log_bin_edges(q_grid)
P_over_tE_edges = log_bin_edges(P_over_tE_grid)

# ============================================================
# [NUEVO BLOQUE 8]
# normalización del colormap
# ============================================================
positive = RMS_map[np.isfinite(RMS_map) & (RMS_map > 0)]
if len(positive) == 0:
    raise RuntimeError("No hay valores positivos para graficar.")

vmin = np.percentile(positive, 5)
vmax = np.percentile(positive, 95)

if vmin <= 0 or vmax <= 0 or np.isclose(vmin, vmax):
    vmin = np.nanmin(positive)
    vmax = np.nanmax(positive)

norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# ============================================================
# [NUEVO BLOQUE 9]
# figura principal
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

RMS_masked = np.ma.masked_invalid(RMS_map)

# RMS_map shape = (Nq, NP)
# x=q, y=P/tE -> hace falta transponer
pcm = ax.pcolormesh(
    q_edges,
    P_over_tE_edges,
    RMS_masked.T,
    cmap="viridis",
    norm=norm,
    shading="auto",
    edgecolors="none",
    linewidth=0.0,
    antialiased=False,
    rasterized=True,
    zorder=2
)

# ============================================================
# [NUEVO BLOQUE 10]
# contornos
# ============================================================
Q_grid_2d, P_over_tE_grid_2d = np.meshgrid(q_grid, P_over_tE_grid, indexing="xy")

contour_levels = np.geomspace(vmin, vmax, 5)

cs = ax.contour(
    Q_grid_2d,
    P_over_tE_grid_2d,
    RMS_masked.T,
    levels=contour_levels,
    colors="white",
    linewidths=1.0,
    alpha=0.9,
    zorder=3
)

ax.clabel(
    cs,
    inline=True,
    fontsize=9,
    fmt=lambda x: f"{x:.1e}"
)

# ============================================================
# [NUEVO BLOQUE 11]
# contorno de detectabilidad
# ============================================================
detect_threshold = 1e-2

if np.nanmin(positive) <= detect_threshold <= np.nanmax(positive):
    cs_detect = ax.contour(
        Q_grid_2d,
        P_over_tE_grid_2d,
        RMS_masked.T,
        levels=[detect_threshold],
        colors="cyan",
        linewidths=2.5,
        linestyles="--",
        zorder=4
    )

    ax.clabel(
        cs_detect,
        inline=True,
        fontsize=10,
        fmt={detect_threshold: r"$10^{-2}$"}
    )

# ============================================================
# [NUEVO BLOQUE 12]
# formato de ejes
# ============================================================
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(q_grid.min(), q_grid.max())
ax.set_ylim(P_over_tE_grid.min(), P_over_tE_grid.max())

ax.set_xlabel(r"$q=M_2/M_1$", fontsize=16)
ax.set_ylabel(r"$P/t_E$", fontsize=16)
ax.set_title(
    rf"Binary-source detectability map "
    rf"($t_E={tE_true:.0f}\,\mathrm{{d}}$, metric={metric_key})",
    fontsize=17
)

ax.grid(True, which="both", alpha=0.25)

cbar = fig.colorbar(pcm, ax=ax)
if metric_key == "RMS":
    cbar.set_label(r"RMS residual magnification", fontsize=15)
else:
    cbar.set_label(metric_key, fontsize=15)

plt.tight_layout()
plt.savefig(
    f"/home/anibal-pc/binary_source/results/q_vs_PoverTE_{metric_key}_tE{int(tE_true)}.png",
    dpi=250,
    bbox_inches="tight"
)
plt.show()