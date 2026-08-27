from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from degeneracy_fit import run_grid_and_save_npz_kepler

import multiprocessing as mp
import numpy as np
import os


# ============================================================
# Parámetros globales del experimento
# ============================================================

t0_true = 50.0

# Ángulos orbitales / xallarap
phi_true = 0.0
theta_true = 0.0
lambda_xi_fixed = 0.5 * np.pi

# Sistema físico
M1 = 2.0
M2 = 1.0
rEhat = 5.0

# ============================================================
# Barrido en período orbital
# ============================================================

P_grid = np.logspace(
    -5,
    5,
    200,
)  # 10 días -> 100000 días


# ============================================================
# Barrido en tiempo de Einstein
# ============================================================

N_tE = 200

tE_grid = np.logspace(
    np.log10(1.0),
    np.log10(500.0),
    N_tE,
)


# ============================================================
# Valores fijos de u0
# ============================================================

u0_fixed_values = np.array([
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
])


# ============================================================
# Configuración temporal
# ============================================================

N_time_points = 10_000
time_window_tE = 3.5


def format_float_for_path(value, precision=4):
    """
    Convierte un número flotante en una cadena segura para usar
    en nombres de directorios y archivos.

    Ejemplos
    --------
    0.01 -> 0p0100
    10.5 -> 10p5000
    """
    return f"{value:.{precision}f}".replace(".", "p")


def run_single_tE(task):
    """
    Ejecuta el barrido completo en P para una combinación fija
    de u0 y tE.

    Cada proceso escribe un archivo NPZ diferente.

    Parameters
    ----------
    task : tuple
        Contiene:

        - índice de u0
        - índice de tE
        - u0 verdadero
        - tE verdadero
        - directorio de salida
    """

    iu0, itE, u0_true, tE_true, directory = task

    tE_tag = format_float_for_path(tE_true, precision=4)

    out_name = os.path.join(
        directory,
        f"scan_kepler_tE_{itE:03d}_{tE_tag}d.npz",
    )

    # --------------------------------------------------------
    # Evitar recalcular resultados existentes
    # --------------------------------------------------------

    if os.path.exists(out_name):
        return {
            "iu0": iu0,
            "itE": itE,
            "u0": u0_true,
            "tE": tE_true,
            "file": out_name,
            "status": "already_exists",
        }

    # --------------------------------------------------------
    # Grilla temporal centrada en t0
    # --------------------------------------------------------

    t = np.linspace(
        t0_true - time_window_tE * tE_true,
        t0_true + time_window_tE * tE_true,
        N_time_points,
    )

    # --------------------------------------------------------
    # Barrido en P
    # --------------------------------------------------------

    run_grid_and_save_npz_kepler(
        out_npz_path=out_name,
        t=t,
        t0_true=t0_true,
        u0_true=u0_true,
        tE_true=tE_true,
        phi_true=phi_true,
        i_true=float(lambda_xi_fixed),
        qflux_true=0.0,
        theta_true=theta_true,
        M1_Msun=M1,
        M2_Msun=M2,
        rEhat_AU=rEhat,
        P_grid=P_grid,
        msource_true=24.0,
        mtotal_true=24.0,
        override_xiE=None,
        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,
    )

    return {
        "iu0": iu0,
        "itE": itE,
        "u0": u0_true,
        "tE": tE_true,
        "file": out_name,
        "status": "finished",
    }


def main():
    home_path = os.path.expanduser("~")

    # Ajustar según la cantidad de núcleos y memoria disponible
    max_workers = min(
        12,
        os.cpu_count() or 1,
    )

    all_tasks = []

    # ========================================================
    # Crear un directorio para cada valor fijo de u0
    # ========================================================

    for iu0, u0_true in enumerate(u0_fixed_values):

        u0_tag = format_float_for_path(
            u0_true,
            precision=4,
        )

        directory = os.path.join(
            home_path,
            "binary_source",
            "results","scan_tE",
            f"scan_tE_u0_{u0_tag}",
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        # Una tarea por cada valor de tE
        for itE, tE_true in enumerate(tE_grid):

            task = (
                int(iu0),
                int(itE),
                float(u0_true),
                float(tE_true),
                directory,
            )

            all_tasks.append(task)

    total_tasks = len(all_tasks)

    print("=" * 70)
    print("Barrido Kepleriano en tE y P")
    print("=" * 70)
    print(f"Valores fijos de u0 : {u0_fixed_values}")
    print(f"Número de valores tE: {len(tE_grid)}")
    print(
        f"Rango de tE         : "
        f"{tE_grid.min():.3f} - {tE_grid.max():.3f} días"
    )
    print(f"Número de valores P : {len(P_grid)}")
    print(
        f"Rango de P          : "
        f"{P_grid.min():.3f} - {P_grid.max():.3f} días"
    )
    print(f"Tareas totales      : {total_tasks}")
    print(f"Procesos simultáneos: {max_workers}")
    print("=" * 70)

    # ========================================================
    # Ejecutar todas las combinaciones (u0, tE)
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:

        futures = {
            executor.submit(run_single_tE, task): task
            for task in all_tasks
        }

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            task = futures[future]

            iu0, itE, u0_true, tE_true, _ = task

            try:
                result = future.result()

                print(
                    f"[{completed:03d}/{total_tasks:03d}] "
                    f"u0={result['u0']:.4f}, "
                    f"tE={result['tE']:8.3f} d, "
                    f"estado={result['status']}"
                )

            except Exception as error:
                print(
                    f"[ERROR] "
                    f"iu0={iu0}, "
                    f"itE={itE}, "
                    f"u0={u0_true:.4f}, "
                    f"tE={tE_true:.4f} d: "
                    f"{error}"
                )

    print("=" * 70)
    print("Barrido completo terminado.")
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
