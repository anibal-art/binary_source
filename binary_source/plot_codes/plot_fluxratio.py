#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# ============================================================
# Parámetros físicos
# ============================================================
tE_true = 150.0
Mtot_fixed = 3.0
rEhat_AU = 5.0

home_path = os.path.expanduser("~")

directory = home_path + f"/binary_source/results/scan_u0_tE{int(tE_true)}/"
pattern = os.path.join(directory, "scan_kepler_u0_*.npz")

metric_key = "RMS"

files = sorted(glob.glob(pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")

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
# Parseo índice
# ============================================================
def extract_qflux_index(filename):
    base = os.path.basename(filename)
    m = re.match(r"scan_kepler_u0_(\d+)\.npz", base)

    if m is None:
        raise ValueError(f"No pude parsear índice desde {base}")

    return int(m.group(1))

# ============================================================
# Bordes logarítmicos
# ============================================================
def log_bin_edges(x):
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Todos los valores deben ser positivos.")

    lx = np.log10(x)

    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (lx[:-1] + lx[1:])
    edges[0] = lx[0] - 0.5 * (lx[1] - lx[0])
    edges[-1] = lx[-1] + 0.5 * (lx[-1] - lx[-2])

    return 10**edges

# ============================================================
# Reconstrucción de q_flux grid
# IMPORTANTE: debe coincidir con el grid usado en la simulación
# ============================================================
N_qflux = 25
qflux_grid_input = np.logspace(-6, 0, N_qflux)

qflux_dict = {}
P_grid_ref = None

for fn in files:
    k = extract_qflux_index(fn)

    if k >= len(qflux_grid_input):
        raise ValueError(
            f"El índice k={k} excede la longitud de qflux_grid_input={len(qflux_grid_input)}"
        )

    d = np.load(fn)

    P_grid = d["P_grid"]

    if P_grid_ref is None:
        P_grid_ref = P_grid.copy()
    else:
        if len(P_grid) != len(P_grid_ref) or not np.allclose(P_grid, P_grid_ref):
            raise ValueError(f"P_grid no coincide entre archivos. Problema en {fn}")

    qflux_dict[k] = qflux_grid_input[k]

sorted_indices = sorted(qflux_dict.keys())
qflux_grid = np.array([qflux_dict[k] for k in sorted_indices], dtype=float)

Nq = len(qflux_grid)
NP = len(P_grid_ref)

# ============================================================
# Construcción del mapa
# ============================================================
metric_map = np.full((Nq, NP), np.nan)
success_map = np.full((Nq, NP), False)

index_to_row = {k: i for i, k in enumerate(sorted_indices)}

bad_details = []

for fn in files:
    k = extract_qflux_index(fn)
    row = index_to_row[k]

    d = np.load(fn)

    if metric_key not in d.files:
        raise KeyError(
            f"La clave {metric_key!r} no existe en {fn}. "
            f"Claves disponibles: {d.files}"
        )

    metric = d[metric_key].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)

    success_map[row, :] = SUCCESS

    m = SUCCESS & np.isfinite(metric) & (metric > 0)
    metric_map[row, m] = metric[m]

    bad = ~m
    if np.any(bad):
        bad_details.append({
            "filename": fn,
            "row": row,
            "k": k,
            "bad_indices": np.where(bad)[0],
            "SUCCESS_bad": SUCCESS[bad],
            "metric_bad": metric[bad],
        })

# ============================================================
# Ejes
# ============================================================
P_over_tE = P_grid_ref / tE_true

qflux_edges = log_bin_edges(qflux_grid)
P_edges = log_bin_edges(P_over_tE)

# ============================================================
# Diagnóstico de bines faltantes
# ============================================================
positive = metric_map[np.isfinite(metric_map) & (metric_map > 0)]

if positive.size == 0:
    raise RuntimeError(
        f"No hay valores positivos para graficar con metric_key={metric_key!r}."
    )

print(f"{metric_key} min =", np.nanmin(positive))
print(f"{metric_key} max =", np.nanmax(positive))
print("Cantidad de NaNs =", np.sum(~np.isfinite(metric_map)), "/", metric_map.size)

fail_map = ~np.isfinite(metric_map)

print()
print("Total bines faltantes:", np.sum(fail_map), "/", fail_map.size)

rows_with_fail = np.where(np.any(fail_map, axis=1))[0]

for row in rows_with_fail:
    bad_cols = np.where(fail_map[row, :])[0]

    print()
    print("qflux =", qflux_grid[row])
    print("faltantes =", len(bad_cols), "de", metric_map.shape[1])
    print("índices P malos =", bad_cols)
    print("P/tE faltantes =", P_over_tE[bad_cols])

# Diagnóstico más detallado: por qué falló cada bin
print()
print("Detalle de fallos:")

for item in bad_details:
    row = item["row"]
    bad_indices = item["bad_indices"]

    if len(bad_indices) == 0:
        continue

    print()
    print("Archivo:", item["filename"])
    print("qflux =", qflux_grid[row])
    print("índices malos =", bad_indices)
    print("P/tE malos =", P_over_tE[bad_indices])
    print("SUCCESS malos =", item["SUCCESS_bad"])
    print("metric malos =", item["metric_bad"])

# ============================================================
# Colormap
# ============================================================
vmin = 5e-3
vmax = 1e0

norm = colors.LogNorm(vmin=vmin, vmax=vmax)

cmap = plt.cm.viridis.copy()
cmap.set_bad("lightgray")

metric_masked = np.ma.masked_invalid(metric_map)

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

pcm = ax.pcolormesh(
    qflux_edges,
    P_edges,
    metric_masked.T,
    cmap=cmap,
    norm=norm,
    shading="auto",
    edgecolors="none",
    linewidth=0.0,
    rasterized=True,
    zorder=1
)

# ============================================================
# Interpolación fina para contornos
# ============================================================
qfine = np.logspace(
    np.log10(qflux_grid.min()),
    np.log10(qflux_grid.max()),
    500
)

Pfine = np.logspace(
    np.log10(P_over_tE.min()),
    np.log10(P_over_tE.max()),
    500
)

interp = RegularGridInterpolator(
    (qflux_grid, P_over_tE),
    metric_map,
    bounds_error=False,
    fill_value=np.nan
)

QFfine, Pfine2d = np.meshgrid(
    qfine,
    Pfine,
    indexing="ij"
)

pts = np.column_stack([
    QFfine.ravel(),
    Pfine2d.ravel()
])

metric_fine = interp(pts).reshape(QFfine.shape)

positive_fine = metric_fine[
    np.isfinite(metric_fine) &
    (metric_fine > 0)
]

# ============================================================
# Contornos blancos
# ============================================================
levels_white = [2e-2, 5e-2, 1e-1, 3e-1, 1e0]

levels_white = [
    lev for lev in levels_white
    if np.nanmin(positive_fine) < lev < np.nanmax(positive_fine)
]

if len(levels_white) > 0:
    cs = ax.contour(
        qfine,
        Pfine,
        metric_fine.T,
        levels=levels_white,
        colors="white",
        linewidths=1.0,
        zorder=5
    )

    ax.clabel(
        cs,
        fmt=lambda x: f"{x:.1e}",
        fontsize=9
    )

# ============================================================
# Contorno de detectabilidad RMS = 10^-2
# ============================================================
threshold = 1e-2

if np.nanmin(positive_fine) < threshold < np.nanmax(positive_fine):
    cs_det = ax.contour(
        qfine,
        Pfine,
        metric_fine.T,
        levels=[threshold],
        colors="cyan",
        linewidths=3.5,
        linestyles="--",
        zorder=10
    )

    ax.clabel(
        cs_det,
        fmt={threshold: r"$10^{-2}$"},
        fontsize=13,
        colors="cyan"
    )
else:
    print(
        f"No se puede dibujar el contorno {threshold:.1e}: "
        f"rango interpolado = [{np.nanmin(positive_fine):.3e}, "
        f"{np.nanmax(positive_fine):.3e}]"
    )

# ============================================================
# Eje derecho: xiE
# ============================================================
secax = ax.secondary_yaxis(
    "right",
    functions=(xiE_from_PoverTE, PoverTE_from_xiE)
)

secax.set_ylabel(
    r"$\xi_E = \frac{a_s}{\hat{r}_E}$",
    fontsize=16,
    labelpad=8
)

secax.set_yscale("log")
secax.tick_params(axis="y", which="both", pad=4)

# ============================================================
# Formato
# ============================================================
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$q_{\rm flux}$", fontsize=18)
ax.set_ylabel(r"$P/t_E$", fontsize=18)

ax.grid(True, which="both", alpha=0.25)

ax.set_title(
    rf"$t_E={tE_true:.0f}\,\mathrm{{days}},\ "
    rf"u_0=0.5$",
    fontsize=20
)

fig.subplots_adjust(right=0.82)

# ============================================================
# Colorbar
# ============================================================
cbar = fig.colorbar(
    pcm,
    ax=ax,
    pad=0.2,
    ticks=[1e-2, 1e-1, 1e0]
)

cbar.ax.set_yticklabels([
    r"$10^{-2}$",
    r"$10^{-1}$",
    r"$10^{0}$"
])

if metric_key == "RMS":
    cbar.set_label(r"RMS residual magnification", fontsize=14)
elif metric_key == "Q_A":
    cbar.set_label(r"$\sqrt{\chi^2/N}$", fontsize=14)
else:
    cbar.set_label(metric_key, fontsize=14)

# ============================================================
# Guardar
# ============================================================
out_png = (
    home_path
    + f"/binary_source/results/"
    + f"qflux_vs_PoverTE_{metric_key}_tE{int(tE_true)}.png"
)

plt.savefig(
    out_png,
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print(f"Figura guardada en:\n{out_png}")

#%%
# ============================================================
# Print detallado de eventos problemáticos
# ============================================================
for item in bad_details:

    row = item["row"]
    bad_indices = item["bad_indices"]

    if len(bad_indices) == 0:
        continue

    fn = item["filename"]

    d = np.load(fn, allow_pickle=False)

    truth = d["truth"]

    print()
    print("="*80)
    print("Archivo:", fn)
    print("="*80)

    print("qflux =", qflux_grid[row])

    print("\ntruth vector:")
    print(truth)

    # Intento de interpretación estándar
    if len(truth) >= 7:
        print("\nParámetros verdaderos:")
        print("t0_true                      =", truth[0])
        print("u0_true                      =", truth[1])
        print("tE_true                      =", truth[2])
        print("phi_true                     =", truth[3])
        print("i_true                       =", truth[4])
        print("M1_Msun                      =", truth[5])
        print("M2_Msun                      =", truth[6])
        print("rEhat_AU                     =", truth[7])
        print("qflux_true                   =", truth[8])
        print("theta_true                   =", truth[9])
        print("fsource_true                 =", truth[10])
        print("fblend_true                  =", truth[11])
        print("use_magnification_fit        =", bool(truth[12]))
        print("override_xiE                 =", None if truth[13] == -1.0 else truth[13])
        print("set_flux_from_truth_photometry =", bool(truth[14]))
        print("rms_on_magnification         =", bool(truth[15]))

    print()

    for j in bad_indices:

        print("-"*60)

        P = d["P_grid"][j]
        P_over_tE_bad = P / tE_true

        print("Índice P =", j)
        print("P =", P, "days")
        print("P/tE =", P_over_tE_bad)

        if "xiE_of_P" in d.files:
            print("xiE =", d["xiE_of_P"][j])

        if "a_AU_of_P" in d.files:
            print("a_AU =", d["a_AU_of_P"][j])

        if "SUCCESS" in d.files:
            print("SUCCESS =", d["SUCCESS"][j])

        if metric_key in d.files:
            print(f"{metric_key} =", d[metric_key][j])

        if "MAXABS" in d.files:
            print("MAXABS =", d["MAXABS"][j])

        if "BEST_T0U0TE" in d.files:
            print("BEST_T0U0TE =", d["BEST_T0U0TE"][j])

        if "DT0" in d.files:
            print("DT0 =", d["DT0"][j])

        if "DU0" in d.files:
            print("DU0 =", d["DU0"][j])

        if "DTE" in d.files:
            print("DTE =", d["DTE"][j])

        # Mostrar si hay NaNs explícitos
        for key in [metric_key, "MAXABS", "DT0", "DU0", "DTE"]:

            if key in d.files:

                val = d[key][j]

                if isinstance(val, float) and np.isnan(val):
                    print(f"WARNING: {key} is NaN")