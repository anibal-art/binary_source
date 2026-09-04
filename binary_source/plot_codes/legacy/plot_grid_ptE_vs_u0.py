import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.interpolate import NearestNDInterpolator


# ============================================================
# Parámetros del gráfico
# ============================================================
tE_true = 150.0

home_path = os.path.expanduser("~")

directory = home_path+"/binary_source/results/scan_u0_tE150/"
pattern = os.path.join(directory, "scan_kepler_u0_*.npz")

# Opciones típicas: "RMS", "MAXABS" o "Q_A"
metric_key = "RMS"

files = sorted(glob.glob(pattern))

if len(files) == 0:
    raise FileNotFoundError(
        f"No encontré archivos con patrón:\n{pattern}"
    )


# ============================================================
# Parseo del índice de u0 desde el nombre del archivo
# ============================================================
def extract_u0_index(filename):
    """
    Extrae k desde un nombre como:

        scan_kepler_u0_003.npz
    """
    base = os.path.basename(filename)

    match = re.fullmatch(
        r"scan_kepler_u0_(\d+)\.npz",
        base
    )

    if match is None:
        raise ValueError(
            f"No pude parsear el índice desde: {base}"
        )

    return int(match.group(1))


# ============================================================
# Bordes lineales para pcolormesh
# ============================================================
def linear_bin_edges(x):
    """
    Construye los bordes de una grilla lineal a partir
    de las posiciones centrales.
    """
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        raise ValueError(
            "Se necesitan al menos dos puntos para construir bordes."
        )

    edges = np.empty(len(x) + 1, dtype=float)

    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])

    return edges


# ============================================================
# Bordes logarítmicos para pcolormesh
# ============================================================
def log_bin_edges(x):
    """
    Construye los bordes de una grilla logarítmica a partir
    de las posiciones centrales.
    """
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        raise ValueError(
            "Se necesitan al menos dos puntos para construir bordes."
        )

    if np.any(x <= 0):
        raise ValueError(
            "Todos los valores deben ser positivos para "
            "construir bordes logarítmicos."
        )

    log_x = np.log10(x)

    log_edges = np.empty(len(x) + 1, dtype=float)

    log_edges[1:-1] = 0.5 * (log_x[:-1] + log_x[1:])
    log_edges[0] = log_x[0] - 0.5 * (log_x[1] - log_x[0])
    log_edges[-1] = log_x[-1] + 0.5 * (
        log_x[-1] - log_x[-2]
    )

    return 10**log_edges


# ============================================================
# Reconstrucción de u0_grid desde los archivos
# ============================================================
u0_dict = {}
P_grid_ref = None

for filename in files:

    k = extract_u0_index(filename)

    with np.load(filename, allow_pickle=False) as data:

        truth = data["truth"].astype(float)

        # truth = [t0_true, u0_true, tE_true, ...]
        u0_true = float(truth[1])

        P_grid = data["P_grid"].astype(float)

    if P_grid_ref is None:
        P_grid_ref = P_grid.copy()

    else:
        same_size = len(P_grid) == len(P_grid_ref)

        if not same_size or not np.allclose(P_grid, P_grid_ref):
            raise ValueError(
                f"P_grid no coincide entre archivos.\n"
                f"Problema encontrado en:\n{filename}"
            )

    u0_dict[k] = u0_true


if len(u0_dict) == 0:
    raise RuntimeError(
        "No se pudo reconstruir u0_grid desde los archivos."
    )


sorted_indices = np.array(
    sorted(u0_dict.keys()),
    dtype=int
)

u0_grid = np.array(
    [u0_dict[k] for k in sorted_indices],
    dtype=float
)

Nu0 = len(u0_grid)
NP = len(P_grid_ref)


# ============================================================
# Construcción del mapa de la métrica
# ============================================================
metric_map = np.full(
    (Nu0, NP),
    np.nan,
    dtype=float
)

index_to_row = {
    k: row
    for row, k in enumerate(sorted_indices)
}


for filename in files:

    k = extract_u0_index(filename)
    row = index_to_row[k]

    with np.load(filename, allow_pickle=False) as data:

        P_grid = data["P_grid"].astype(float)
        metric = data[metric_key].astype(float)
        success = data["SUCCESS"].astype(bool)

    valid = (
        success
        & np.isfinite(P_grid)
        & np.isfinite(metric)
        & (metric > 0)
    )

    if not np.any(valid):
        continue

    metric_map[row, valid] = metric[valid]


# ============================================================
# Eje vertical: P/tE
# ============================================================
P_over_tE_grid = P_grid_ref / tE_true


# ============================================================
# Interpolación de celdas inválidas con vecino más cercano
# en el espacio log10(u0), log10(P/tE)
# ============================================================
log_u0 = np.log10(u0_grid)
log_P_over_tE = np.log10(P_over_tE_grid)

UU_log, PP_log = np.meshgrid(
    log_u0,
    log_P_over_tE,
    indexing="ij"
)

mask_valid = (
    np.isfinite(metric_map)
    & (metric_map > 0)
)

mask_invalid = ~mask_valid


if not np.any(mask_valid):
    raise RuntimeError(
        f"El mapa está completamente vacío para tE={tE_true}. "
        f"Revisá que existan datos con SUCCESS=True y "
        f"{metric_key}>0."
    )


if np.any(mask_invalid):

    interpolator = NearestNDInterpolator(
        np.column_stack(
            [
                UU_log[mask_valid],
                PP_log[mask_valid]
            ]
        ),
        metric_map[mask_valid]
    )

    metric_map[mask_invalid] = interpolator(
        UU_log[mask_invalid],
        PP_log[mask_invalid]
    )

    print(
        f"Se interpolaron {np.count_nonzero(mask_invalid)} "
        f"celdas mediante vecino más cercano."
    )


# ============================================================
# Bordes para pcolormesh
# ============================================================
u0_edges = log_bin_edges(u0_grid)
P_over_tE_edges = log_bin_edges(P_over_tE_grid)


# ============================================================
# Normalización logarítmica del colormap
# ============================================================
positive = metric_map[
    np.isfinite(metric_map)
    & (metric_map > 0)
]

if len(positive) == 0:
    raise RuntimeError(
        "No hay valores positivos para graficar."
    )


vmin = np.percentile(positive, 5)
vmax = np.percentile(positive, 95)

if (
    vmin <= 0
    or vmax <= 0
    or np.isclose(vmin, vmax)
):
    vmin = np.nanmin(positive)
    vmax = np.nanmax(positive)


norm = colors.LogNorm(
    vmin=vmin,
    vmax=vmax,
    clip=False
)


# ============================================================
# Figura principal
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

metric_masked = np.ma.masked_invalid(metric_map)


pcm = ax.pcolormesh(
    u0_edges,
    P_over_tE_edges,
    metric_masked.T,
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
# Grilla bidimensional para los contornos
# ============================================================
U0_grid_2d, P_over_tE_grid_2d = np.meshgrid(
    u0_grid,
    P_over_tE_grid,
    indexing="xy"
)


# ============================================================
# Contornos fijos:
# 10^-4, 10^-3, 10^-2 y 10^-1
# ============================================================
requested_contour_levels = np.array(
    [1e-4, 1e-3, 1e-1],
    dtype=float
)

data_min = np.nanmin(positive)
data_max = np.nanmax(positive)

# Matplotlib solo puede dibujar niveles dentro del rango de datos.
contour_levels = requested_contour_levels[
    (requested_contour_levels >= data_min)
    & (requested_contour_levels <= data_max)
]


if len(contour_levels) > 0:

    cs = ax.contour(
        U0_grid_2d,
        P_over_tE_grid_2d,
        metric_masked.T,
        levels=contour_levels,
        colors="white",
        linewidths=1.5,
        linestyles="solid",
        alpha=0.95,
        zorder=3
    )

    contour_labels = {
        1e-4: r"$10^{-4}$",
        1e-3: r"$10^{-3}$",
        1e-2: r"$10^{-2}$",
        1e-1: r"$10^{-1}$"
    }

    ax.clabel(
        cs,
        inline=True,
        inline_spacing=5,
        fontsize=10,
        fmt=contour_labels
    )

else:
    print(
        "Advertencia: ninguno de los contornos solicitados "
        "está dentro del rango de los datos."
    )


# ============================================================
# Resaltar el contorno de detectabilidad 10^-2
# ============================================================
detect_threshold = 1e-2

if data_min <= detect_threshold <= data_max:

    cs_detect = ax.contour(
        U0_grid_2d,
        P_over_tE_grid_2d,
        metric_masked.T,
        levels=[detect_threshold],
        colors="cyan",
        linewidths=2.8,
        linestyles="--",
        zorder=4
    )

    ax.clabel(
        cs_detect,
        inline=True,
        inline_spacing=6,
        fontsize=11,
        fmt={
            detect_threshold: r"$10^{-2}$"
        }
    )


# ============================================================
# Formato de los ejes
# ============================================================
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(
    u0_grid.min(),
    u0_grid.max()
)

ax.set_ylim(
    P_over_tE_grid.min(),
    P_over_tE_grid.max()
)

ax.set_xlabel(
    r"$u_0$",
    fontsize=16
)

ax.set_ylabel(
    r"$P/t_E$",
    fontsize=16
)

ax.set_title(
    rf"Binary-source detectability map "
    rf"($t_E={tE_true:.0f}\,\mathrm{{d}}$, "
    rf"metric={metric_key})",
    fontsize=17
)

ax.grid(
    True,
    which="both",
    alpha=0.25
)

ax.tick_params(
    axis="both",
    which="both",
    labelsize=12
)


# ============================================================
# Barra de color
# ============================================================
cbar = fig.colorbar(
    pcm,
    ax=ax,
    pad=0.02
)

if metric_key == "RMS":
    cbar.set_label(
        r"RMS residual magnification",
        fontsize=15
    )

elif metric_key == "MAXABS":
    cbar.set_label(
        r"Maximum absolute residual magnification",
        fontsize=15
    )

elif metric_key == "Q_A":
    cbar.set_label(
        r"$\sqrt{\chi^2/N}$",
        fontsize=15
    )

else:
    cbar.set_label(
        metric_key,
        fontsize=15
    )

cbar.ax.tick_params(labelsize=12)


# ============================================================
# Guardado
# ============================================================
plt.tight_layout()

output_file = os.path.join(
    home_path,
    "binary_source",
    "results",
    f"u0_vs_PoverTE_{metric_key}_tE{int(tE_true)}.png"
)

plt.savefig(
    output_file,
    dpi=250,
    bbox_inches="tight"
)

print(f"Figura guardada en:\n{output_file}")

plt.show()

#%%

import glob, os, re
import numpy as np

tE_true  = 150.0

home_path = os.path.expanduser("~")
directory = home_path + f"/binary_source/results/scan_u0_tE{int(tE_true)}/"
pattern   = os.path.join(directory, "scan_kepler_u0_*.npz")
files     = sorted(glob.glob(pattern))

print(f"Archivos encontrados: {len(files)}\n")

total_bins   = 0
success_bins = 0
nan_metric   = 0
zero_metric  = 0
neg_metric   = 0

for fn in files:
    d       = np.load(fn, allow_pickle=False)
    SUCCESS = d["SUCCESS"].astype(bool)
    RMS     = d["RMS"].astype(float)
    P_grid  = d["P_grid"].astype(float)
    u0_true = float(d["truth"][1])

    n_total   = len(SUCCESS)
    n_success = SUCCESS.sum()
    n_nan     = (~np.isfinite(RMS)).sum()
    n_zero    = (RMS == 0).sum()
    n_neg     = (RMS < 0).sum()

    total_bins   += n_total
    success_bins += n_success
    nan_metric   += n_nan
    zero_metric  += n_zero
    neg_metric   += n_neg

    # muestra los archivos con muchos fallos
    fail_rate = 1.0 - n_success / n_total
    if fail_rate > 0.3:
        print(f"  ⚠ u0={u0_true:.4f}  fallo={fail_rate*100:.0f}%  "
              f"(success={n_success}/{n_total})  nan_RMS={n_nan}")

print(f"\n--- Resumen global ---")
print(f"Total bins        : {total_bins}")
print(f"SUCCESS=True      : {success_bins}  ({100*success_bins/total_bins:.1f}%)")
print(f"SUCCESS=False     : {total_bins-success_bins}  ({100*(total_bins-success_bins)/total_bins:.1f}%)")
print(f"RMS NaN/inf       : {nan_metric}")
print(f"RMS == 0          : {zero_metric}")
print(f"RMS < 0           : {neg_metric}")

#%%
for fn in files:
    d = np.load(fn, allow_pickle=False)
    SUCCESS = d["SUCCESS"].astype(bool)
    if not SUCCESS.all():
        u0_true = float(d["truth"][1])
        P_fail  = d["P_grid"][~SUCCESS]
        print(f"u0={u0_true:.4f}  P_fail={np.round(P_fail,1)}")
        
#%%