
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