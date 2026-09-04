import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def load_npz_curve(npz_path):
    """
    Carga un archivo .npz y trata de encontrar:
      - el grid de periodos
      - la curva RMS
      - el valor de u0_true

    Ajustá los nombres de claves si en tus archivos usan otros.
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"\nLeyendo {npz_path}")
    print("Claves disponibles:", keys)

    # ----------------------------
    # posibles nombres para P_grid
    # ----------------------------
    P_candidates = [
        "P_grid",
        "period_grid",
        "periods",
        "P_days_grid",
        "P_days",
    ]

    P = None
    for k in P_candidates:
        if k in data:
            P = np.asarray(data[k], dtype=float)
            break

    if P is None:
        raise KeyError(
            f"No encontré el grid de periodos en {npz_path}. "
            f"Claves disponibles: {keys}"
        )

    # ----------------------------
    # posibles nombres para RMS
    # ----------------------------
    rms_candidates = [
        "rms",
        "RMS",
        "rms_grid",
        "RMS_grid",
        "rms_values",
        "RMS_values",
        "rms_mag",
        "RMS_mag",
        "rms_magnification",
        "RMS_magnification",
    ]

    rms = None
    for k in rms_candidates:
        if k in data:
            rms = np.asarray(data[k], dtype=float)
            break

    if rms is None:
        raise KeyError(
            f"No encontré la curva RMS en {npz_path}. "
            f"Claves disponibles: {keys}"
        )

    # ----------------------------
    # posibles nombres para u0_true
    # ----------------------------
    u0_candidates = [
        "u0_true",
        "u0",
        "u0_input",
        "u0_base",
    ]

    u0_val = None
    for k in u0_candidates:
        if k in data:
            arr = np.asarray(data[k])
            u0_val = float(arr.reshape(-1)[0])
            break

    if u0_val is None:
        # fallback: lo extraemos del nombre si no está guardado
        u0_val = np.nan

    return P, rms, u0_val, keys


def make_u0_sweep_gif(
    pattern="scan_kepler_u0_*.npz",
    gif_name="scan_u0.gif",
    fps=6,
    logx=True,
    logy=False,
):
    npz_files = sorted(glob.glob(pattern))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")

    # -------------------------------------------------
    # primera pasada: cargar todo para fijar límites
    # -------------------------------------------------
    curves = []
    all_x = []
    all_y = []

    for f in npz_files:
        P, rms, u0_val, keys = load_npz_curve(f)

        # ordenamos por P por seguridad
        idx = np.argsort(P)
        P = P[idx]
        rms = rms[idx]

        curves.append((f, P, rms, u0_val))
        all_x.append(P)
        all_y.append(rms[np.isfinite(rms)])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min = np.nanmin(all_x)
    x_max = np.nanmax(all_x)
    y_min = np.nanmin(all_y)
    y_max = np.nanmax(all_y)

    # pequeño margen vertical
    if logy:
        y_min_plot = y_min / 1.2
        y_max_plot = y_max * 1.2
    else:
        dy = y_max - y_min
        if dy == 0:
            dy = 1.0
        y_min_plot = y_min - 0.05 * dy
        y_max_plot = y_max + 0.05 * dy

    # -------------------------------------------------
    # frames
    # -------------------------------------------------
    frames = []

    for i, (fname, P, rms, u0_val) in enumerate(curves):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        ax.plot(P, rms, lw=2)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min_plot, y_max_plot)

        ax.set_xlabel(r"Period $P$ [days]")
        ax.set_ylabel("RMS")
        ax.set_title("Kepler-consistent xallarap scan")

        txt = (
            rf"$u_0={u0_val:.3f}$" + "\n" +
            rf"frame {i+1}/{len(curves)}"
        )
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9)
        )

        ax.grid(alpha=0.25)

        fig.tight_layout()

        # convertir la figura a array sin guardar PNG intermedio
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, :3]  # RGB

        frames.append(frame)
        plt.close(fig)

    # -------------------------------------------------
    # guardar GIF
    # -------------------------------------------------
    imageio.mimsave(gif_name, frames, fps=fps, loop=0)
    print(f"\nGIF guardado en: {gif_name}")


if __name__ == "__main__":
    make_u0_sweep_gif(
        pattern="scan_kepler_u0_*.npz",
        gif_name="scan_u0.gif",
        fps=5,
        logx=True,
        logy=False,
    )
