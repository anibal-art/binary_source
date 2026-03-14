import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

pattern = "scan_kepler_lambda_*.npz"
files = sorted(glob.glob(pattern))

fig, ax = plt.subplots(figsize=(8,5))

data_list = []
lambda_values = []

for fn in files:
    d = np.load(fn, allow_pickle=False)

    P_grid = d["P_grid"].astype(float)
    RMS = d["RMS"].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)
    truth = d["truth"].astype(float)

    lambda_xi = float(truth[4])

    m = SUCCESS & np.isfinite(RMS)
    if not np.any(m):
        continue

    idx = np.argsort(P_grid[m])
    Pp = P_grid[m][idx]
    Rp = RMS[m][idx]

    lambda_values.append(lambda_xi)
    data_list.append((lambda_xi, Pp, Rp))

lambda_values = np.array(lambda_values)

# Colormap físico
norm = colors.Normalize(vmin=0.0, vmax=0.5*np.pi)
cmap = cm.plasma

for lambda_xi, Pp, Rp in data_list:
    color = cmap(norm(lambda_xi))
    lw = 1.0 + 1.5*(lambda_xi/(0.5*np.pi))
    ax.semilogx(Pp, Rp, color=color, linewidth=lw)

ax.set_xlabel(r"$P$ [days]")
ax.set_ylabel(r"RMS (magnification)")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_title(r"RMS($P$) for different orbital inclinations $\lambda_\xi$")

# 🔹 Colorbar asociada correctamente al eje
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$\lambda_\xi$ [rad]")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8,5))

data_list = []
lambda_values = []

for fn in files:
    d = np.load(fn, allow_pickle=False)

    P_grid = d["xiE_of_P"].astype(float)
    RMS = d["RMS"].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)
    truth = d["truth"].astype(float)

    lambda_xi = float(truth[4])

    m = SUCCESS & np.isfinite(RMS)
    if not np.any(m):
        continue

    idx = np.argsort(P_grid[m])
    Pp = P_grid[m][idx]
    Rp = RMS[m][idx]

    lambda_values.append(lambda_xi)
    data_list.append((lambda_xi, Pp, Rp))

lambda_values = np.array(lambda_values)

# Colormap físico
norm = colors.Normalize(vmin=0.0, vmax=0.5*np.pi)
cmap = cm.plasma

for lambda_xi, Pp, Rp in data_list:
    color = cmap(norm(lambda_xi))
    lw = 1.0 + 1.5*(lambda_xi/(0.5*np.pi))
    ax.semilogx(Pp, Rp, color=color, linewidth=lw)

ax.set_xlabel(r"$\xi_E(P)$")
ax.set_ylabel(r"RMS (magnification)")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_title(r"RMS($P$) for different orbital inclinations $\lambda_\xi$")

# 🔹 Colorbar asociada correctamente al eje
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$\lambda_\xi$ [rad]")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8,5))

data_list = []
lambda_values = []

for fn in files:
    d = np.load(fn, allow_pickle=False)
    # print(d)
    P_grid = d["P_grid"].astype(float)
    RMS = d["MAXABS"].astype(float)
    SUCCESS = d["SUCCESS"].astype(bool)
    truth = d["truth"].astype(float)

    lambda_xi = float(truth[4])

    m = SUCCESS & np.isfinite(RMS)
    if not np.any(m):
        continue

    idx = np.argsort(P_grid[m])
    Pp = P_grid[m][idx]
    Rp = RMS[m][idx]

    lambda_values.append(lambda_xi)
    data_list.append((lambda_xi, Pp, Rp))

lambda_values = np.array(lambda_values)

# Colormap físico
norm = colors.Normalize(vmin=0.0, vmax=0.5*np.pi)
cmap = cm.plasma

for lambda_xi, Pp, Rp in data_list:
    color = cmap(norm(lambda_xi))
    lw = 1.0 + 1.5*(lambda_xi/(0.5*np.pi))
    ax.semilogx(Pp, Rp, color=color, linewidth=lw)

ax.set_xlabel(r"$P$ [days]")
ax.set_ylabel(r"$MAX|F_{\xi_E}-F_{PSPL}|$")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_title(r"RMS($P$) for different orbital inclinations $\lambda_\xi$")

# 🔹 Colorbar asociada correctamente al eje
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$\lambda_\xi$ [rad]")

plt.tight_layout()
plt.show()
