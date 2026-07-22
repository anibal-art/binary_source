
# %%
# import numpy as np

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.interpolate import NearestNDInterpolator


# ============================================================
# Parámetros generales
# ============================================================

home_path = os.path.expanduser("~")

base_directory = os.path.join(
    home_path,
    "binary_source",
    "results","scan_tE"
)

# Directorios producidos por el nuevo barrido:
#
# scan_tE_u0_0p0100/
# scan_tE_u0_0p0300/
# scan_tE_u0_0p1000/
# ...
directory_pattern = os.path.join(
    base_directory,
    "scan_tE_u0_*",
)

# Opciones típicas:
# "RMS", "MAXABS" o "Q_A"
metric_key = "RMS"


# ============================================================
# Parámetros físicos
#
# Se usan solamente si xiE no está guardado en el NPZ.
# ============================================================

M1_Msun = 2.0
M2_Msun = 1.0
rEhat_AU = 5.0


# ============================================================
# Opciones del gráfico
# ============================================================

requested_contour_levels = np.array(
    [
        1e-4,
        1e-3,
        1e-2,
        1e-1,
    ],
    dtype=float,
)

# Contorno que se quiere resaltar
detect_threshold = 1e-2

# Dibujar líneas auxiliares P/tE = constante
draw_P_over_tE_lines = True

P_over_tE_reference = np.array(
    [
        0.1,
        1.0,
        10.0,
        100.0,
    ],
    dtype=float,
)

# Dibujar la línea espacial xiE/u0 = 1
draw_xiE_over_u0_equal_one = True


# ============================================================
# Relación de Kepler
# ============================================================

def xiE_from_kepler(
    P_days,
    M1_Msun,
    M2_Msun,
    rEhat_AU,
):
    """
    Calcula xiE para una órbita binaria kepleriana.

    Se supone que xiE corresponde a la órbita de la fuente 1
    alrededor del centro de masa:

        xiE = a1 / rEhat

    con

        a1 = a_rel * M2 / (M1 + M2)

    y, usando P en años, masas en masas solares y a en AU:

        a_rel^3 = (M1 + M2) P^2
    """

    P_days = np.asarray(P_days, dtype=float)

    if np.any(P_days <= 0):
        raise ValueError(
            "Todos los períodos deben ser positivos."
        )

    M_total = M1_Msun + M2_Msun
    P_years = P_days / 365.25

    a_relative_AU = (
        M_total * P_years**2
    )**(1.0 / 3.0)

    a1_AU = (
        a_relative_AU
        * M2_Msun
        / M_total
    )

    xiE = a1_AU / rEhat_AU

    return xiE


# ============================================================
# Lectura de xiE
# ============================================================

def get_xiE_grid(data, P_grid):
    """
    Intenta leer xiE directamente desde el archivo NPZ.

    Si no encuentra un array de xiE compatible con P_grid,
    lo calcula mediante la relación de Kepler.
    """

    for key in data.files:

        normalized_key = (
            key.lower()
            .replace("_", "")
            .replace("-", "")
        )

        if normalized_key.startswith("xie"):

            candidate = np.asarray(
                data[key],
                dtype=float,
            ).squeeze()

            if (
                candidate.ndim == 1
                and candidate.size == P_grid.size
                and np.all(np.isfinite(candidate))
                and np.all(candidate > 0)
            ):
                print(
                    f"Usando xiE guardado en la clave "
                    f"'{key}'."
                )

                return candidate

    # Si xiE no está guardado, calcularlo
    return xiE_from_kepler(
        P_days=P_grid,
        M1_Msun=M1_Msun,
        M2_Msun=M2_Msun,
        rEhat_AU=rEhat_AU,
    )


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
            "Se necesitan al menos dos puntos para "
            "construir bordes."
        )

    if np.any(x <= 0):
        raise ValueError(
            "Todos los valores deben ser positivos para "
            "construir bordes logarítmicos."
        )

    log_x = np.log10(x)

    log_edges = np.empty(
        len(x) + 1,
        dtype=float,
    )

    log_edges[1:-1] = 0.5 * (
        log_x[:-1] + log_x[1:]
    )

    log_edges[0] = (
        log_x[0]
        - 0.5 * (log_x[1] - log_x[0])
    )

    log_edges[-1] = (
        log_x[-1]
        + 0.5 * (log_x[-1] - log_x[-2])
    )

    return 10**log_edges


# ============================================================
# Etiqueta segura para nombres de archivo
# ============================================================

def format_float_for_path(value, precision=4):
    return (
        f"{value:.{precision}f}"
        .replace(".", "p")
    )


# ============================================================
# Interpolación de celdas inválidas
# ============================================================

def interpolate_invalid_cells(
    metric_map,
    tE_grid,
    xiE_over_u0_grid,
):
    """
    Interpola celdas inválidas mediante vecino más cercano
    en el espacio:

        log10(tE), log10(xiE/u0)

    metric_map tiene forma:

        (N_tE, N_xiE)
    """

    metric_map = np.asarray(
        metric_map,
        dtype=float,
    ).copy()

    log_tE = np.log10(tE_grid)
    log_xiE_over_u0 = np.log10(
        xiE_over_u0_grid
    )

    TT_log, XX_log = np.meshgrid(
        log_tE,
        log_xiE_over_u0,
        indexing="ij",
    )

    mask_valid = (
        np.isfinite(metric_map)
        & (metric_map > 0)
    )

    mask_invalid = ~mask_valid

    if not np.any(mask_valid):
        raise RuntimeError(
            "El mapa no contiene ninguna celda válida."
        )

    if np.any(mask_invalid):

        interpolator = NearestNDInterpolator(
            np.column_stack(
                [
                    TT_log[mask_valid],
                    XX_log[mask_valid],
                ]
            ),
            metric_map[mask_valid],
        )

        metric_map[mask_invalid] = interpolator(
            TT_log[mask_invalid],
            XX_log[mask_invalid],
        )

        print(
            f"Se interpolaron "
            f"{np.count_nonzero(mask_invalid)} "
            f"celdas mediante vecino más cercano."
        )

    return metric_map


# ============================================================
# Lectura de un directorio correspondiente a u0 fijo
# ============================================================

def load_fixed_u0_directory(directory):
    """
    Lee todos los archivos correspondientes a un valor fijo
    de u0.

    Cada archivo debe contener un valor de tE y el barrido
    completo en P.
    """

    pattern = os.path.join(
        directory,
        "scan_kepler_tE_*.npz",
    )

    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No encontré archivos con patrón:\n"
            f"{pattern}"
        )

    records = []

    P_grid_ref = None
    xiE_grid_ref = None
    u0_ref = None

    for filename in files:

        with np.load(
            filename,
            allow_pickle=False,
        ) as data:

            truth = np.asarray(
                data["truth"],
                dtype=float,
            )

            # Convención esperada:
            # truth = [t0_true, u0_true, tE_true, ...]
            u0_true = float(truth[1])
            tE_true = float(truth[2])

            P_grid = np.asarray(
                data["P_grid"],
                dtype=float,
            )

            metric = np.asarray(
                data[metric_key],
                dtype=float,
            )

            success = np.asarray(
                data["SUCCESS"],
                dtype=bool,
            )

            xiE_grid = get_xiE_grid(
                data=data,
                P_grid=P_grid,
            )

        # ----------------------------------------------------
        # Verificar que u0 sea fijo en todo el directorio
        # ----------------------------------------------------

        if u0_ref is None:
            u0_ref = u0_true

        elif not np.isclose(
            u0_true,
            u0_ref,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"El directorio no tiene u0 fijo.\n"
                f"Valor inicial: {u0_ref}\n"
                f"Valor encontrado: {u0_true}\n"
                f"Archivo: {filename}"
            )

        # ----------------------------------------------------
        # Verificar que P_grid sea común
        # ----------------------------------------------------

        if P_grid_ref is None:
            P_grid_ref = P_grid.copy()

        else:
            same_size = (
                len(P_grid)
                == len(P_grid_ref)
            )

            if (
                not same_size
                or not np.allclose(
                    P_grid,
                    P_grid_ref,
                )
            ):
                raise ValueError(
                    f"P_grid no coincide entre archivos.\n"
                    f"Problema encontrado en:\n"
                    f"{filename}"
                )

        # ----------------------------------------------------
        # Verificar que xiE_grid sea común
        # ----------------------------------------------------

        if xiE_grid_ref is None:
            xiE_grid_ref = xiE_grid.copy()

        else:
            same_size = (
                len(xiE_grid)
                == len(xiE_grid_ref)
            )

            if (
                not same_size
                or not np.allclose(
                    xiE_grid,
                    xiE_grid_ref,
                )
            ):
                raise ValueError(
                    f"xiE_grid no coincide entre archivos.\n"
                    f"Problema encontrado en:\n"
                    f"{filename}"
                )

        valid = (
            success
            & np.isfinite(P_grid)
            & np.isfinite(xiE_grid)
            & np.isfinite(metric)
            & (P_grid > 0)
            & (xiE_grid > 0)
            & (metric > 0)
        )

        metric_row = np.full(
            len(P_grid),
            np.nan,
            dtype=float,
        )

        metric_row[valid] = metric[valid]

        records.append(
            {
                "tE": tE_true,
                "metric": metric_row,
                "filename": filename,
            }
        )

    # Ordenar las filas por tE verdadero
    records.sort(
        key=lambda record: record["tE"]
    )

    tE_grid = np.array(
        [
            record["tE"]
            for record in records
        ],
        dtype=float,
    )

    metric_map = np.vstack(
        [
            record["metric"]
            for record in records
        ]
    )

    xiE_over_u0_grid = (
        xiE_grid_ref / u0_ref
    )

    return {
        "directory": directory,
        "u0": u0_ref,
        "tE_grid": tE_grid,
        "P_grid": P_grid_ref,
        "xiE_grid": xiE_grid_ref,
        "xiE_over_u0_grid": xiE_over_u0_grid,
        "metric_map": metric_map,
    }


# ============================================================
# Buscar todos los directorios de u0 fijo
# ============================================================

directories = sorted(
    directory
    for directory in glob.glob(directory_pattern)
    if os.path.isdir(directory)
)

if len(directories) == 0:
    raise FileNotFoundError(
        "No encontré directorios con el patrón:\n"
        f"{directory_pattern}"
    )


# ============================================================
# Cargar todos los mapas
# ============================================================

datasets = []

for directory in directories:

    print("=" * 70)
    print(f"Leyendo:\n{directory}")

    try:
        dataset = load_fixed_u0_directory(
            directory
        )

        datasets.append(dataset)

        print(
            f"u0 fijo = {dataset['u0']:.6g}"
        )

        print(
            f"Número de valores de tE = "
            f"{len(dataset['tE_grid'])}"
        )

        print(
            f"Número de valores de P = "
            f"{len(dataset['P_grid'])}"
        )

    except Exception as error:
        print(
            f"[ERROR] No se pudo cargar "
            f"{directory}:\n{error}"
        )


if len(datasets) == 0:
    raise RuntimeError(
        "No se pudo cargar ningún mapa válido."
    )


# ============================================================
# Normalización global
#
# Se usa la misma escala de colores para todos los u0.
# ============================================================

all_positive = []

for dataset in datasets:

    positive = dataset["metric_map"][
        np.isfinite(dataset["metric_map"])
        & (dataset["metric_map"] > 0)
    ]

    if len(positive) > 0:
        all_positive.append(positive)


if len(all_positive) == 0:
    raise RuntimeError(
        "No hay valores positivos para graficar."
    )


all_positive = np.concatenate(
    all_positive
)

vmin = np.percentile(
    all_positive,
    5,
)

vmax = np.percentile(
    all_positive,
    95,
)

if (
    vmin <= 0
    or vmax <= 0
    or np.isclose(vmin, vmax)
):
    vmin = np.nanmin(all_positive)
    vmax = np.nanmax(all_positive)


norm = colors.LogNorm(
    vmin=vmin,
    vmax=vmax,
    clip=False,
)


# ============================================================
# Crear una figura para cada valor fijo de u0
# ============================================================

for dataset in datasets:

    u0_true = dataset["u0"]
    tE_grid = dataset["tE_grid"]
    P_grid = dataset["P_grid"]

    xiE_over_u0_grid = dataset[
        "xiE_over_u0_grid"
    ]

    metric_map = interpolate_invalid_cells(
        metric_map=dataset["metric_map"],
        tE_grid=tE_grid,
        xiE_over_u0_grid=xiE_over_u0_grid,
    )

    positive = metric_map[
        np.isfinite(metric_map)
        & (metric_map > 0)
    ]

    data_min = np.nanmin(positive)
    data_max = np.nanmax(positive)

    # --------------------------------------------------------
    # Bordes para pcolormesh
    # --------------------------------------------------------

    tE_edges = log_bin_edges(
        tE_grid
    )

    xiE_over_u0_edges = log_bin_edges(
        xiE_over_u0_grid
    )

    # --------------------------------------------------------
    # Figura principal
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    metric_masked = np.ma.masked_invalid(
        metric_map
    )

    pcm = ax.pcolormesh(
        tE_edges,
        xiE_over_u0_edges,
        metric_masked.T,
        cmap="viridis",
        norm=norm,
        shading="auto",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
        zorder=2,
    )

    # --------------------------------------------------------
    # Grilla para los contornos
    # --------------------------------------------------------

    TE_grid_2d, XI_grid_2d = np.meshgrid(
        tE_grid,
        xiE_over_u0_grid,
        indexing="xy",
    )

    # --------------------------------------------------------
    # Contornos normales
    #
    # Se excluye 10^-2 porque se dibuja después resaltado.
    # Así no aparece repetido.
    # --------------------------------------------------------

    regular_requested_levels = (
        requested_contour_levels[
            ~np.isclose(
                requested_contour_levels,
                detect_threshold,
            )
        ]
    )

    regular_contour_levels = (
        regular_requested_levels[
            (
                regular_requested_levels
                >= data_min
            )
            & (
                regular_requested_levels
                <= data_max
            )
        ]
    )

    if len(regular_contour_levels) > 0:

        cs = ax.contour(
            TE_grid_2d,
            XI_grid_2d,
            metric_masked.T,
            levels=regular_contour_levels,
            colors="white",
            linewidths=1.5,
            linestyles="solid",
            alpha=0.95,
            zorder=3,
        )

        contour_labels = {
            level: (
                rf"$10^{{"
                rf"{int(np.round(np.log10(level)))}"
                rf"}}$"
            )
            for level in regular_contour_levels
        }

        ax.clabel(
            cs,
            inline=True,
            inline_spacing=5,
            fontsize=10,
            fmt=contour_labels,
        )

    # --------------------------------------------------------
    # Contorno resaltado de 10^-2
    # --------------------------------------------------------

    if (
        data_min
        <= detect_threshold
        <= data_max
    ):

        cs_detect = ax.contour(
            TE_grid_2d,
            XI_grid_2d,
            metric_masked.T,
            levels=[detect_threshold],
            colors="cyan",
            linewidths=2.8,
            linestyles="--",
            zorder=4,
        )

        ax.clabel(
            cs_detect,
            inline=True,
            inline_spacing=6,
            fontsize=11,
            fmt={
                detect_threshold: r"$10^{-2}$"
            },
        )

    # --------------------------------------------------------
    # Línea espacial xiE/u0 = 1
    # --------------------------------------------------------

    if draw_xiE_over_u0_equal_one:

        if (
            xiE_over_u0_grid.min()
            <= 1.0
            <= xiE_over_u0_grid.max()
        ):

            ax.axhline(
                1.0,
                color="red",
                linewidth=1.8,
                linestyle=":",
                label=r"$\xi_E/u_0=1$",
                zorder=5,
            )

    # --------------------------------------------------------
    # Líneas temporales P/tE = constante
    #
    # No son contornos de RMS. Son solamente referencias
    # físicas para comparar P con tE.
    # --------------------------------------------------------

    if draw_P_over_tE_lines:

        tE_line = np.logspace(
            np.log10(tE_grid.min()),
            np.log10(tE_grid.max()),
            500,
        )

        for ratio in P_over_tE_reference:

            P_line = ratio * tE_line

            valid_line = (
                (P_line >= P_grid.min())
                & (P_line <= P_grid.max())
            )

            if not np.any(valid_line):
                continue

            xiE_line = xiE_from_kepler(
                P_days=P_line[valid_line],
                M1_Msun=M1_Msun,
                M2_Msun=M2_Msun,
                rEhat_AU=rEhat_AU,
            )

            y_line = xiE_line / u0_true

            visible = (
                (y_line >= xiE_over_u0_grid.min())
                & (y_line <= xiE_over_u0_grid.max())
            )

            if not np.any(visible):
                continue

            ax.plot(
                tE_line[valid_line][visible],
                y_line[visible],
                color="white",
                linewidth=1.1,
                linestyle="--",
                alpha=0.65,
                label=rf"$P/t_E={ratio:g}$",
                zorder=3.5,
            )

    # --------------------------------------------------------
    # Formato de ejes
    # --------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(
        tE_grid.min(),
        tE_grid.max(),
    )

    ax.set_ylim(
        xiE_over_u0_grid.min(),
        xiE_over_u0_grid.max(),
    )

    ax.set_xlabel(
        r"$t_E\;[\mathrm{d}]$",
        fontsize=16,
    )

    ax.set_ylabel(
        r"$\xi_E/u_0$",
        fontsize=16,
    )

    ax.set_title(
        rf"Binary-source detectability map "
        rf"($u_0={u0_true:.3g}$, "
        rf"metric={metric_key})",
        fontsize=17,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.20,
    )

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=12,
    )

    # Evitar etiquetas repetidas en la leyenda
    handles, labels = ax.get_legend_handles_labels()

    unique_legend = {}

    for handle, label in zip(handles, labels):
        if label not in unique_legend:
            unique_legend[label] = handle

    if len(unique_legend) > 0:

        ax.legend(
            unique_legend.values(),
            unique_legend.keys(),
            fontsize=9,
            loc="best",
            framealpha=0.75,
        )

    # --------------------------------------------------------
    # Barra de color
    # --------------------------------------------------------

    cbar = fig.colorbar(
        pcm,
        ax=ax,
        pad=0.02,
    )

    if metric_key == "RMS":

        cbar.set_label(
            r"RMS residual magnification",
            fontsize=15,
        )

    elif metric_key == "MAXABS":

        cbar.set_label(
            r"Maximum absolute residual magnification",
            fontsize=15,
        )

    elif metric_key == "Q_A":

        cbar.set_label(
            r"$\sqrt{\chi^2/N}$",
            fontsize=15,
        )

    else:

        cbar.set_label(
            metric_key,
            fontsize=15,
        )

    cbar.ax.tick_params(
        labelsize=12
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    plt.tight_layout()

    u0_tag = format_float_for_path(
        u0_true,
        precision=4,
    )

    output_file = os.path.join(
        base_directory,
        (
            f"xiEoveru0_vs_tE_"
            f"{metric_key}_"
            f"u0_{u0_tag}.png"
        ),
    )

    plt.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )

    print(
        f"Figura guardada para u0={u0_true:.6g}:\n"
        f"{output_file}"
    )

    plt.show()

    # Cerrar para no acumular figuras en memoria
    plt.close(fig)
# np.load("/home/anibal-pc/binary_source/results/scan_u0_tE150/scan_kepler_u0_000.npz", allow_pickle=False)
#%%%

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

directory = "/home/anibal-pc/binary_source/results/scan_u0_tE150/"
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