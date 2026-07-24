

import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import NearestNDInterpolator

# ============================================================
# Datos observacionales de eventos 1L2S / xallarap con 
# compañera débil u oscura de la literatura
# ============================================================

# Formato: (nombre, u0, P_días, tE_días, etiqueta_tipo)
# Solo incluimos eventos donde P está medido (xallarap)
# y la compañera es débil/oscura (no resuelta fotométricamente)

observed_xallarap = [
    # (nombre, u0, P [d], tE [d], marker)
    # Miyazaki+2020: 2L2S, compañera fuente oscura MC~0.14 Msun, qF,I~1.4e-3
    # u0 muy pequeño (alta magnificación), P~37d, tE~95d
    ("OGLE-2013-BLG-0911", 4.8e-3, 36.67, 94.7, "*"),
    # Rota+2021: 1L1S+xallarap, compañera débil, P~29d, tE~65d
    ("MOA-2006-BLG-074",   0.05,   29.0,  65.0, "o"),
    # Satoh+2023: xallarap P~5.5d, tE~30d; compañera BD oscura
    ("OGLE-2019-BLG-0825", 0.05,    5.5,  30.0, "s"),
    # Alcock+2001: xallarap P~28d, tE~40d; compañera no resuelta (LMC)
    ("MACHO 96-LMC-2",     0.30,   28.0,  40.0, "^"),
    # Palanque-Delabrouille+1998: P~28d, tE~100d (SMC)
    ("MACHO-SMC-1",        0.20,   28.0, 100.0, "D"),
    # Li+2024: P~70d, tE~100d; compañera K dwarf (visible, referencia)
    ("OGLE-2015-BLG-0845", 0.40,   70.0, 100.0, "P"),
]
# ============================================================
# Parámetros generales
# ============================================================

home_path = os.path.expanduser("~")

base_directory = os.path.join(
    home_path,
    "binary_source/results/results_logu0_qflux0"
)

tE_list = [50, 100,150, 200,250, 300,400, 500,600]

metric_key = "RMS"        # también puede ser "MAXABS"
detect_threshold = 1e-2   # contorno que querés comparar

save_path = os.path.join(
    home_path,
    f"binary_source/results/compare_contours_{metric_key}_threshold_{detect_threshold:.0e}.png"
)


# ============================================================
# Funciones auxiliares
# ============================================================

def extract_u0_index(filename):
    """
    Extrae el índice k desde nombres del tipo:

    scan_kepler_u0_003.npz
    """
    base = os.path.basename(filename)
    match = re.match(r"scan_kepler_u0_(\d+)\.npz", base)

    if match is None:
        raise ValueError(f"No pude parsear el índice desde: {base}")

    return int(match.group(1))


def log_bin_edges(x):
    """
    Construye bordes logarítmicos para una grilla positiva.
    Sirve para calcular áreas en espacio log-log.
    """
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Todos los valores deben ser positivos para bordes logarítmicos.")

    lx = np.log10(x)

    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (lx[:-1] + lx[1:])
    edges[0] = lx[0] - 0.5 * (lx[1] - lx[0])
    edges[-1] = lx[-1] + 0.5 * (lx[-1] - lx[-2])

    return 10**edges


def load_metric_map_for_tE(
    tE_true,
    base_directory,
    metric_key="RMS",
    interpolate_nans=True,
):
    """
    Carga todos los archivos scan_kepler_u0_*.npz para un tE dado
    y reconstruye el mapa metric(u0, P/tE).

    Devuelve un diccionario con:
        u0_grid
        P_grid
        P_over_tE_grid
        metric_map
        success_fraction
        n_files
    """

    directory = os.path.join(
        base_directory,
        f"scan_u0_tE{int(tE_true)}"
    )

    pattern = os.path.join(directory, "scan_kepler_u0_*.npz")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")

    # --------------------------------------------------------
    # Primer recorrido: reconstruir u0_grid y verificar P_grid
    # --------------------------------------------------------

    u0_dict = {}
    P_grid_ref = None

    total_bins = 0
    success_bins = 0

    for fn in files:
        k = extract_u0_index(fn)
        d = np.load(fn, allow_pickle=False)

        truth = d["truth"].astype(float)
        u0_true = float(truth[1])

        P_grid = d["P_grid"].astype(float)

        if P_grid_ref is None:
            P_grid_ref = P_grid.copy()
        else:
            same_size = len(P_grid) == len(P_grid_ref)
            same_values = np.allclose(P_grid, P_grid_ref)

            if (not same_size) or (not same_values):
                raise ValueError(f"P_grid no coincide entre archivos. Problema en {fn}")

        SUCCESS = d["SUCCESS"].astype(bool)

        total_bins += len(SUCCESS)
        success_bins += np.sum(SUCCESS)

        u0_dict[k] = u0_true

    sorted_indices = np.array(sorted(u0_dict.keys()), dtype=int)
    u0_grid = np.array([u0_dict[k] for k in sorted_indices], dtype=float)

    Nu0 = len(u0_grid)
    NP = len(P_grid_ref)

    # --------------------------------------------------------
    # Segundo recorrido: llenar el mapa
    # --------------------------------------------------------

    metric_map = np.full((Nu0, NP), np.nan, dtype=float)
    index_to_row = {k: i for i, k in enumerate(sorted_indices)}

    for fn in files:
        k = extract_u0_index(fn)
        row = index_to_row[k]

        d = np.load(fn, allow_pickle=False)

        P_grid = d["P_grid"].astype(float)
        metric = d[metric_key].astype(float)
        SUCCESS = d["SUCCESS"].astype(bool)

        valid = (
            SUCCESS
            & np.isfinite(P_grid)
            & np.isfinite(metric)
            & (metric > 0)
        )

        metric_map[row, valid] = metric[valid]

    P_over_tE_grid = P_grid_ref / float(tE_true)

    # --------------------------------------------------------
    # Interpolación de NaNs con vecino más cercano en log-log
    # --------------------------------------------------------

    if interpolate_nans:
        log_u0 = np.log10(u0_grid)
        log_P_over_tE = np.log10(P_over_tE_grid)

        UU, PP = np.meshgrid(log_u0, log_P_over_tE, indexing="ij")

        mask_valid = np.isfinite(metric_map) & (metric_map > 0)
        mask_nan = ~mask_valid

        if np.any(mask_valid) and np.any(mask_nan):
            interp = NearestNDInterpolator(
                np.column_stack([UU[mask_valid], PP[mask_valid]]),
                metric_map[mask_valid],
            )

            metric_map[mask_nan] = interp(UU[mask_nan], PP[mask_nan])

    success_fraction = success_bins / total_bins

    return {
        "tE_true": float(tE_true),
        "directory": directory,
        "u0_grid": u0_grid,
        "P_grid": P_grid_ref,
        "P_over_tE_grid": P_over_tE_grid,
        "metric_map": metric_map,
        "success_fraction": success_fraction,
        "n_files": len(files),
    }


def compute_threshold_summary(result, threshold):
    """
    Calcula fracciones del mapa por encima y por debajo del umbral.

    Como los ejes son logarítmicos, calcula la fracción de área
    en el plano log10(u0), log10(P/tE).
    """

    u0_grid = result["u0_grid"]
    y_grid = result["P_over_tE_grid"]
    metric_map = result["metric_map"]

    u0_edges = log_bin_edges(u0_grid)
    y_edges = log_bin_edges(y_grid)

    du = np.diff(np.log10(u0_edges))
    dy = np.diff(np.log10(y_edges))

    area = du[:, None] * dy[None, :]

    finite = np.isfinite(metric_map) & (metric_map > 0)

    if not np.any(finite):
        return {
            "fraction_above": np.nan,
            "fraction_below": np.nan,
            "median_metric": np.nan,
            "min_metric": np.nan,
            "max_metric": np.nan,
        }

    total_area = np.sum(area[finite])

    above = finite & (metric_map >= threshold)
    below = finite & (metric_map < threshold)

    fraction_above = np.sum(area[above]) / total_area
    fraction_below = np.sum(area[below]) / total_area

    return {
        "fraction_above": fraction_above,
        "fraction_below": fraction_below,
        "median_metric": np.nanmedian(metric_map[finite]),
        "min_metric": np.nanmin(metric_map[finite]),
        "max_metric": np.nanmax(metric_map[finite]),
    }


def plot_threshold_contours(
    results,
    threshold=1e-2,
    metric_key="RMS",
    save_path=None,
):
    """
    Superpone el contorno metric = threshold para todos los tE.
    """

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, len(results)))

    legend_handles = []

    for color, result in zip(colors, results):
        tE_true = result["tE_true"]
        u0_grid = result["u0_grid"]
        y_grid = result["P_over_tE_grid"]
        metric_map = result["metric_map"]

        positive = metric_map[np.isfinite(metric_map) & (metric_map > 0)]

        if len(positive) == 0:
            print(f"tE={tE_true:.0f}: no hay valores positivos.")
            continue

        min_metric = np.nanmin(positive)
        max_metric = np.nanmax(positive)

        if not (min_metric <= threshold <= max_metric):
            print(
                f"tE={tE_true:.0f}: el umbral {threshold:.1e} "
                f"queda fuera del rango [{min_metric:.2e}, {max_metric:.2e}]."
            )
            continue

        U0_grid_2d, Y_grid_2d = np.meshgrid(
            u0_grid,
            y_grid,
            indexing="xy"
        )

        cs = ax.contour(
            U0_grid_2d,
            Y_grid_2d,
            metric_map.T,
            levels=[threshold],
            colors=[color],
            linewidths=2.4,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.4,
                label=rf"$t_E={tE_true:.0f}\,\mathrm{{d}}$",
            )
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$u_0$", fontsize=16)
    ax.set_ylabel(r"$P/t_E$", fontsize=16)

    ax.set_title(
        rf"Contour {metric_key} $= {threshold:.0e}$ for differents $t_E$",
        fontsize=16,
    )

    ax.grid(True, which="both", alpha=0.25)
# ---- Puntos observados (xallarap / 1L2S con compañera débil) ----
    
    obs_colors = {
        "dark":    "crimson",   # compañera oscura confirmada
        "faint":   "orangered", # compañera débil / no resuelta
        "visible": "gray",      # compañera visible (referencia)
    }
    
    darkness = {
        "OGLE-2013-BLG-0911":  "dark",   # MC ~ 0.14 Msun, qF,I ~ 1.4e-3
        "MOA-2006-BLG-074":    "dark",
        "OGLE-2019-BLG-0825":  "dark",
        "MACHO 96-LMC-2":      "faint",
        "MACHO-SMC-1":         "faint",
        "OGLE-2015-BLG-0845":  "visible",
    }
    
    markers = {
        "OGLE-2013-BLG-0911":  (5, 1, 0),  # estrella de 5 puntas
        "MOA-2006-BLG-074":    "o",
        "OGLE-2019-BLG-0825":  "s",
        "MACHO 96-LMC-2":      "^",
        "MACHO-SMC-1":         "D",
        "OGLE-2015-BLG-0845":  "P",
    }
    
    for name, u0_obs, P_obs, tE_obs, _ in observed_xallarap:
        y_obs = P_obs / tE_obs        # P/tE adimensional
        col   = obs_colors[darkness[name]]
        mk    = markers[name]
        ax.scatter(u0_obs, y_obs, marker=mk, color=col,
                   s=120, zorder=10, edgecolors="k", linewidths=0.7,
                   label=name)
    
    legend_handles += [
        Line2D([0],[0], marker=mk, color=obs_colors[darkness[name]],
               markeredgecolor="k", markersize=9, linestyle="None",
               label=f"{name}  ($t_E={tE_obs:.0f}$d, $P={P_obs:.1f}$d)")
        for name, u0_obs, P_obs, tE_obs, mk
        in observed_xallarap
    ]
    if len(legend_handles) > 0:
        ax.legend(handles=legend_handles, fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Figura guardada en:\n{save_path}")

    plt.show()


def plot_threshold_summary(
    tE_values,
    fraction_above_values,
    metric_key="RMS",
    threshold=1e-2,
):
    """
    Grafica la fracción del mapa con metric >= threshold
    como función de tE.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        tE_values,
        fraction_above_values,
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel(r"$t_E$ [days]", fontsize=15)

    ax.set_ylabel(
        rf"Fraction with {metric_key} $\geq {threshold:.0e}$",
        fontsize=14,
    )

    ax.set_title(
        rf"detectable area as function of $t_E$",
        fontsize=15,
    )

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Cargar mapas para todos los tE
# ============================================================

results = []

for tE_true in tE_list:
    print(f"\nCargando tE = {tE_true} días")

    result = load_metric_map_for_tE(
        tE_true=tE_true,
        base_directory=base_directory,
        metric_key=metric_key,
        interpolate_nans=True,
    )

    results.append(result)

    print(f"  archivos encontrados : {result['n_files']}")
    print(f"  SUCCESS fraction     : {100 * result['success_fraction']:.1f}%")


# ============================================================
# Superponer contornos RMS = 1e-2
# ============================================================

plot_threshold_contours(
    results,
    threshold=detect_threshold,
    metric_key=metric_key,
    save_path=save_path,
)


# ============================================================
# Resumen cuantitativo
# ============================================================

summary_rows = []

for result in results:
    summary = compute_threshold_summary(
        result,
        threshold=detect_threshold,
    )

    row = {
        "tE_true": result["tE_true"],
        "success_fraction": result["success_fraction"],
        "fraction_metric_above_threshold": summary["fraction_above"],
        "fraction_metric_below_threshold": summary["fraction_below"],
        "median_metric": summary["median_metric"],
        "min_metric": summary["min_metric"],
        "max_metric": summary["max_metric"],
    }

    summary_rows.append(row)


print("\nResumen por tE")
print("-" * 100)
print(
    f"{'tE':>8} "
    f"{'success':>12} "
    f"{f'frac {metric_key}>={detect_threshold:.0e}':>24} "
    f"{'median':>14} "
    f"{'min':>14} "
    f"{'max':>14}"
)

for row in summary_rows:
    print(
        f"{row['tE_true']:8.0f} "
        f"{row['success_fraction']:12.3f} "
        f"{row['fraction_metric_above_threshold']:24.3f} "
        f"{row['median_metric']:14.3e} "
        f"{row['min_metric']:14.3e} "
        f"{row['max_metric']:14.3e}"
    )


# ============================================================
# Gráfico de fracción detectable vs tE
# ============================================================

tE_values = np.array([row["tE_true"] for row in summary_rows], dtype=float)

fraction_above_values = np.array(
    [row["fraction_metric_above_threshold"] for row in summary_rows],
    dtype=float,
)

plot_threshold_summary(
    tE_values,
    fraction_above_values,
    metric_key=metric_key,
    threshold=detect_threshold,
)