import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe

# ============================================================
# Parámetros del barrido
# ============================================================
tE_true = 50.0
rEhat_AU = 5.0

# Estas grillas DEBEN coincidir con las del script de generación
a_over_RE_grid = np.logspace(-3, 0, 25)
Mtot_grid = np.logspace(-1, 2, 30)   # 0.1 a 100 Msun

# rango "físico" que querés remarcar en la figura
Mmin_phys = 0.1
Mmax_phys = 10.0

pattern = f"/home/anibal-pc/binary_source/results/scan_aRE_Mtot_tE{int(tE_true)}/scan_aRE_*_Mtot_*.npz"
files = sorted(glob.glob(pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")

# ============================================================
# Parseo de índices desde nombre de archivo
# ============================================================
def extract_indices_from_filename(filename):
    """
    Extrae ia e im desde:
    scan_aRE_003_Mtot_015.npz
    """
    base = os.path.basename(filename)
    m = re.match(r"scan_aRE_(\d+)_Mtot_(\d+)\.npz", base)
    if m is None:
        raise ValueError(f"No pude parsear índices desde: {base}")
    ia = int(m.group(1))
    im = int(m.group(2))
    return ia, im

# ============================================================
# Función para bordes logarítmicos
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
# Reconstrucción de matrices en la grilla (a_over_RE, Mtot)
# ============================================================
Na = len(a_over_RE_grid)
Nm = len(Mtot_grid)

RMS_map = np.full((Na, Nm), np.nan, dtype=float)
P_over_tE_map = np.full((Na, Nm), np.nan, dtype=float)

for fn in files:
    ia, im = extract_indices_from_filename(fn)

    if not (0 <= ia < Na and 0 <= im < Nm):
        continue

    d = np.load(fn, allow_pickle=False)

    P_grid_file = d["P_grid"].astype(float)
    RMS = d["RMS"].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)

    m = SUCCESS & np.isfinite(P_grid_file) & np.isfinite(RMS)
    if not np.any(m):
        continue

    P_day = float(P_grid_file[m][0])
    rms_val = float(RMS[m][0])

    RMS_map[ia, im] = rms_val
    P_over_tE_map[ia, im] = P_day / tE_true

# ============================================================
# Coordenadas de la grilla en el plano (a_s/R_E, P/t_E)
# ============================================================
A_grid, M_grid = np.meshgrid(a_over_RE_grid, Mtot_grid, indexing="ij")
a_AU_grid = A_grid * rEhat_AU
P_yr_grid = np.sqrt(a_AU_grid**3 / M_grid)
P_day_grid = 365.25 * P_yr_grid
P_over_tE_grid = P_day_grid / tE_true

# ============================================================
# Limpieza conservadora de artefactos de borde
# ============================================================
RMS_map[~np.isfinite(RMS_map)] = np.nan
RMS_map[RMS_map <= 0] = np.nan

positive_all = RMS_map[np.isfinite(RMS_map)]
if len(positive_all) == 0:
    raise RuntimeError("No hay valores positivos de RMS para graficar.")

tiny_threshold = np.percentile(positive_all, 0.5)

border_mask = np.zeros_like(RMS_map, dtype=bool)
border_mask[0, :] = True
border_mask[-1, :] = True
border_mask[:, 0] = True
border_mask[:, -1] = True

RMS_map[border_mask & np.isfinite(RMS_map) & (RMS_map < tiny_threshold)] = np.nan

# ============================================================
# Normalización de color
# ============================================================
positive = RMS_map[np.isfinite(RMS_map) & (RMS_map > 0)]
if len(positive) == 0:
    raise RuntimeError("No quedaron valores positivos de RMS luego de limpiar el mapa.")

vmin = np.percentile(positive, 5)
vmax = np.percentile(positive, 95)
norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

# ============================================================
# Fondo gris no físico
# ============================================================
x_curve = np.logspace(np.log10(a_over_RE_grid.min()),
                      np.log10(a_over_RE_grid.max()), 800)

def p_over_tE_for_constant_mass(a_over_RE, Mtot_Msun, rEhat_AU, tE_days):
    a_AU = a_over_RE * rEhat_AU
    P_yr = np.sqrt(a_AU**3 / Mtot_Msun)
    P_days = 365.25 * P_yr
    return P_days / tE_days

y_mass_min = p_over_tE_for_constant_mass(x_curve, Mmin_phys, rEhat_AU, tE_true)
y_mass_max = p_over_tE_for_constant_mass(x_curve, Mmax_phys, rEhat_AU, tE_true)

finite_p = P_over_tE_grid[np.isfinite(P_over_tE_grid)]
ymin_plot = np.nanmin(finite_p)
ymax_plot = np.nanmax(finite_p)

# región de masas demasiado bajas: P/tE demasiado grande
ax.fill_between(
    x_curve,
    y_mass_min,
    ymax_plot * np.ones_like(x_curve),
    color="lightgray",
    alpha=0.5,
    zorder=0
)

# región de masas demasiado altas: P/tE demasiado chico
ax.fill_between(
    x_curve,
    ymin_plot * np.ones_like(x_curve),
    y_mass_max,
    color="lightgray",
    alpha=0.5,
    zorder=0
)

# ============================================================
# Mapa coloreado
# ============================================================
a_edges = log_bin_edges(a_over_RE_grid)
m_edges = log_bin_edges(Mtot_grid)

A_edges, M_edges = np.meshgrid(a_edges, m_edges, indexing="ij")
a_AU_edges = A_edges * rEhat_AU
P_yr_edges = np.sqrt(a_AU_edges**3 / M_edges)
P_day_edges = 365.25 * P_yr_edges
P_over_tE_edges = P_day_edges / tE_true

RMS_masked = np.ma.masked_invalid(RMS_map)

pcm = ax.pcolormesh(
    A_edges,
    P_over_tE_edges,
    RMS_masked,
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
# Contornos con etiquetas
# ============================================================
RMS_for_contour = np.ma.masked_invalid(RMS_map)

contour_levels = np.geomspace(vmin, vmax, 5)

cs = ax.contour(
    A_grid,
    P_over_tE_grid,
    RMS_for_contour,
    levels=contour_levels,
    colors="white",
    linewidths=1.0,
    alpha=0.9,
    norm=norm,
    zorder=3
)

clabels = ax.clabel(
    cs,
    inline=True,
    fontsize=8,
    fmt=lambda x: f"{x:.1e}"
)

for txt in clabels:
    txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])

# ============================================================
# [NUEVO BLOQUE A]
# colores distintos para cada curva de masa
# ============================================================
mass_list = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
mass_colors = plt.cm.tab10(np.linspace(0, 1, len(mass_list)))

# ============================================================
# [NUEVO BLOQUE B]
# función para ubicar labels en la región gris superior
# ============================================================
def choose_label_in_upper_gray_region(x_curve, y_curve, y_mass_min_curve, frac_x=0.88, y_factor=1.18):
    """
    Busca una posición en la región gris superior, ligeramente por encima
    de la curva de masa mínima física.
    """
    lx = np.log10(x_curve)
    xtarget = 10 ** (lx.min() + frac_x * (lx.max() - lx.min()))
    idx = np.argmin(np.abs(np.log10(x_curve) - np.log10(xtarget)))

    xlab = x_curve[idx]
    ybase = y_mass_min_curve[idx]
    ylab = ybase * y_factor

    if ylab >= ymax_plot:
        ylab = ybase * 1.08

    if ylab <= ybase:
        return None, None

    return xlab, ylab

# ============================================================
# [NUEVO BLOQUE C]
# curvas de masa constante + labels en la zona gris
# ============================================================
xmin_plot = a_over_RE_grid.min()
xmax_plot = a_over_RE_grid.max()

for i, Mtot in enumerate(mass_list):
    curve_color = mass_colors[i]
    y_curve = p_over_tE_for_constant_mass(x_curve, Mtot, rEhat_AU, tE_true)

    mask = (
        np.isfinite(y_curve) &
        (y_curve >= ymin_plot) &
        (y_curve <= ymax_plot)
    )

    if np.any(mask):
        ax.plot(
            x_curve[mask],
            y_curve[mask],
            color=curve_color,
            linewidth=1.1,
            alpha=0.95,
            zorder=4
        )

        # label en la región gris superior
        xlab, ylab = choose_label_in_upper_gray_region(
            x_curve, y_curve, y_mass_min, frac_x=0.90, y_factor=1.15 + 0.04 * i
        )

        if xlab is not None and (ylab < ymax_plot):
            label = rf"$M={Mtot:g}\,M_\odot$"
            txt = ax.text(
                xlab,
                ylab,
                label,
                color=curve_color,
                fontsize=10,
                ha="left",
                va="bottom",
                zorder=6
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

# remarcar bordes físicos
ax.plot(x_curve, y_mass_min, color="gray", linewidth=1.0, linestyle="--", zorder=1)
ax.plot(x_curve, y_mass_max, color="gray", linewidth=1.0, linestyle="--", zorder=1)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$a_s/R_E$", fontsize=16)
ax.set_ylabel(r"$P/t_E$", fontsize=16)
ax.set_title(
    rf"Binary-source detectability map "
    rf"($t_E={tE_true:.0f}\,\mathrm{{d}}$, $\hat r_E={rEhat_AU:.1f}\,\mathrm{{AU}}$)",
    fontsize=18
)

ax.set_xlim(xmin_plot, xmax_plot)
ax.set_ylim(ymin_plot, ymax_plot)

ax.grid(True, which="both", alpha=0.25)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r"RMS (magnification)", fontsize=16)

plt.tight_layout()
plt.savefig("/home/anibal-pc/grid_rms.png", dpi=250, bbox_inches="tight")
plt.show()