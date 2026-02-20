import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# [NUEVO BLOQUE] 1) Cargar resultados y utilidades
# ============================================================
def load_grid_npz(path_npz: str):
    z = np.load(path_npz, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    return out

def _mesh_from_centers(x, y):
    """
    Dado x (nx,) y y (ny,) centros de celdas, devuelve mallas (ny+1, nx+1)
    para pcolormesh, asumiendo spacing no necesariamente uniforme.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def edges_1d(c):
        e = np.empty(c.size + 1, dtype=float)
        e[1:-1] = 0.5 * (c[1:] + c[:-1])
        e[0] = c[0] - 0.5 * (c[1] - c[0])
        e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
        return e

    xe = edges_1d(x)
    ye = edges_1d(y)
    X, Y = np.meshgrid(xe, ye)
    return X, Y

def plot_heatmap(
    xiE_grid, P_grid, Z, success=None, *,
    title="", xlabel=r"$\xi_E$", ylabel=r"$P$ [days]",
    log10=False, vmin=None, vmax=None, cmap=None,
    show_colorbar=True
):
    xiE_grid = np.asarray(xiE_grid, float)
    P_grid   = np.asarray(P_grid, float)
    Z = np.asarray(Z, float)

    if success is not None:
        Z = np.where(success, Z, np.nan)

    if log10:
        Zp = np.log10(Z)
        cbar_label = r"$\log_{10}$(" + title + ")"
    else:
        Zp = Z
        cbar_label = title

    X, Y = _mesh_from_centers(xiE_grid, P_grid)
    # Nota: Z está en (n_xi, n_P). Para pcolormesh queremos (n_P, n_xi).
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    m = ax.pcolormesh(X, Y, Zp.T, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_colorbar:
        cb = fig.colorbar(m, ax=ax)
        cb.set_label(cbar_label)

    return fig, ax

def overlay_contour(ax, xiE_grid, P_grid, Z, level, success=None, *, color=None, linewidths=1.6):
    xiE_grid = np.asarray(xiE_grid, float)
    P_grid   = np.asarray(P_grid, float)
    Z = np.asarray(Z, float)
    if success is not None:
        Z = np.where(success, Z, np.nan)

    Xi, Pi = np.meshgrid(xiE_grid, P_grid, indexing="ij")  # (n_xi, n_P)
    # contour espera Z como (ny, nx) con X,Y como (ny,nx). Usamos transpuesta.
    cs = ax.contour(
        xiE_grid, P_grid, Z.T,
        levels=[level],
        linewidths=linewidths,
        colors=color
    )
    return cs

def pick_best_points(RMS, success, n=5):
    """
    Devuelve lista de (i_xi, j_P) de los n mejores puntos (menor RMS).
    """
    RMSm = np.where(success, RMS, np.nan)
    flat = RMSm.ravel()
    idx = np.argsort(flat)
    out = []
    for k in idx:
        if np.isnan(flat[k]):
            continue
        i = k // RMSm.shape[1]
        j = k %  RMSm.shape[1]
        out.append((i, j))
        if len(out) >= n:
            break
    return out

def plot_curve_pair(t, A_truth, A_fit, *, title=""):
    t = np.asarray(t, float)
    A_truth = np.asarray(A_truth, float)
    A_fit   = np.asarray(A_fit, float)
    resid = A_truth - A_fit

    fig1, ax1 = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax1.plot(t, A_truth, label="Truth (xallarap)")
    ax1.plot(t, A_fit,   label="Best PSPL fit", linestyle="--")
    ax1.set_title(title)
    ax1.set_xlabel("time [JD]")
    ax1.set_ylabel("Magnification A(t)")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    fig2, ax2 = plt.subplots(figsize=(8, 3.5), dpi=160)
    ax2.plot(t, resid)
    ax2.set_title("Residual: A_truth - A_fit")
    ax2.set_xlabel("time [JD]")
    ax2.set_ylabel(r"$\Delta A$")
    ax2.grid(True, alpha=0.25)

    return (fig1, ax1), (fig2, ax2)


# ============================================================
# [NUEVO BLOQUE] 2) Script principal de gráficos
# ============================================================
npz_path = "xallarap_vs_pspl_grid.npz"
D = load_grid_npz(npz_path)

t        = D["t"]
xiE_grid = D["xiE_grid"]
P_grid   = D["P_grid"]

RMS     = D["RMS"]
MAXABS  = D["MAXABS"]
DT0     = D["DT0"]
DU0     = D["DU0"]
DTE     = D["DTE"]
SUCCESS = D["SUCCESS"].astype(bool)

has_curves = ("A_truth_grid" in D) and ("A_fit_grid" in D)
if has_curves:
    A_truth_grid = D["A_truth_grid"]
    A_fit_grid   = D["A_fit_grid"]

# ---- (a) Heatmaps básicos
fig, ax = plot_heatmap(xiE_grid, P_grid, RMS, success=SUCCESS,
                       title="RMS of (A_truth - A_fit)", log10=True)
# Contorno de "indistinguible": elegí un umbral razonable (ajustalo)
rms_thr = 1e-3
cs = overlay_contour(ax, xiE_grid, P_grid, RMS, level=rms_thr, success=SUCCESS, color=None)
ax.clabel(cs, inline=True, fontsize=9, fmt={rms_thr: f"RMS={rms_thr:g}"})
plt.show()

fig, ax = plot_heatmap(xiE_grid, P_grid, MAXABS, success=SUCCESS,
                       title="MAXABS of (A_truth - A_fit)", log10=True)
max_thr = 5e-3
cs = overlay_contour(ax, xiE_grid, P_grid, MAXABS, level=max_thr, success=SUCCESS, color=None)
ax.clabel(cs, inline=True, fontsize=9, fmt={max_thr: f"MAX={max_thr:g}"})
plt.show()

# ---- (b) Sesgos en parámetros del PSPL (t0,u0,tE)
fig, ax = plot_heatmap(xiE_grid, P_grid, DT0, success=SUCCESS, title=r"$\Delta t_0$ [days]", log10=False)
plt.show()

fig, ax = plot_heatmap(xiE_grid, P_grid, DU0, success=SUCCESS, title=r"$\Delta u_0$", log10=False)
plt.show()

fig, ax = plot_heatmap(xiE_grid, P_grid, DTE, success=SUCCESS, title=r"$\Delta t_E$ [days]", log10=False)
plt.show()

# ---- (c) Mostrar algunas curvas ejemplo (mejores RMS)
if has_curves:
    best_pts = pick_best_points(RMS, SUCCESS, n=4)
    for (i_xi, j_P) in best_pts:
        xiE = float(xiE_grid[i_xi])
        P   = float(P_grid[j_P])
        (fig1, ax1), (fig2, ax2) = plot_curve_pair(
            t,
            A_truth_grid[i_xi, j_P, :],
            A_fit_grid[i_xi, j_P, :],
            title=f"xiE={xiE:.4f}, P={P:.2f} days | RMS={RMS[i_xi,j_P]:.3e}, MAX={MAXABS[i_xi,j_P]:.3e}"
        )
        plt.show()
else:
    print("No encontré A_truth_grid / A_fit_grid en el npz. Re-ejecutá con store_curves=True.")

import numpy as np
import matplotlib.pyplot as plt

# --- asumimos que ya cargaste: xiE_grid, P_grid, RMS, SUCCESS
RMS_thr = 1e-3
mask = SUCCESS & np.isfinite(RMS) & (RMS < RMS_thr)

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)

# máscara binaria
im = ax.imshow(
    mask.T.astype(int),
    origin="lower",
    aspect="auto",
    extent=[xiE_grid.min(), xiE_grid.max(), P_grid.min(), P_grid.max()],
    interpolation="nearest",
)

# contorno del umbral sobre RMS
Z = np.where(SUCCESS, RMS, np.nan)
cs = ax.contour(xiE_grid, P_grid, Z.T, levels=[RMS_thr], linewidths=1.8)
ax.clabel(cs, inline=True, fontsize=9, fmt={RMS_thr: f"RMS={RMS_thr:g}"})

ax.set_xlabel(r"$\xi_E$")
ax.set_ylabel(r"$P$ [days]")
ax.set_title(f"Indistinguible region (RMS < {RMS_thr:g})")
cb = fig.colorbar(im, ax=ax, ticks=[0,1])
cb.ax.set_yticklabels(["No", "Sí"])
cb.set_label("Indistinguible")
plt.show()

