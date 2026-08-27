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

# Sistema físico
M1 = 2.0
M2 = 1.0
rEhat = 5.0

# Flujo de la segunda fuente
qflux_true = 0.0

# Barrido en período
P_grid = np.logspace(1, 5, 60)

# Barrido en u0
N_u0 = 25
u0_grid = np.logspace(-2, 1, N_u0)

# Número de realizaciones angulares
N_angles = 200

# Semilla para reproducibilidad
RANDOM_SEED = 12345


# ============================================================
# Generación de configuraciones angulares
# ============================================================

def generate_angular_samples(n_angles, seed=12345):
    """
    Genera realizaciones conjuntas de los tres ángulos orbitales.

    phi_xi:
        Fase orbital, uniforme en [0, 2*pi).

    theta_xi:
        Orientación proyectada respecto de la trayectoria del lente,
        uniforme en [0, 2*pi).

    lambda_xi:
        Parámetro de inclinación usado por el modelo:
            lambda_xi = 0     -> órbita proyectada lineal
            lambda_xi = pi/2  -> órbita proyectada circular

        Para orientaciones isotrópicas:
            sin(lambda_xi) ~ Uniform(0, 1).
    """
    rng = np.random.default_rng(seed)

    phi_samples = rng.uniform(
        0.0,
        2.0 * np.pi,
        n_angles,
    )

    theta_samples = rng.uniform(
        0.0,
        2.0 * np.pi,
        n_angles,
    )

    sin_lambda = rng.uniform(
        0.0,
        1.0,
        n_angles,
    )

    lambda_samples = np.arcsin(sin_lambda)

    return phi_samples, theta_samples, lambda_samples


# ============================================================
# Ejecución de una realización
# ============================================================

def run_single_configuration(task):
    """
    Ejecuta el barrido completo en P para una combinación de:

        u0,
        phi_xi,
        theta_xi,
        lambda_xi.

    Cada tarea escribe un archivo NPZ diferente.
    """
    (
        u0_index,
        angle_index,
        u0_true,
        tE_true,
        phi_true,
        theta_true,
        lambda_xi,
        directory,
    ) = task

    # Un subdirectorio por valor de u0 evita miles de archivos
    # dentro del mismo directorio.
    u0_directory = os.path.join(
        directory,
        f"u0_{u0_index:03d}",
    )

    os.makedirs(u0_directory, exist_ok=True)

    out_name = os.path.join(
        u0_directory,
        f"angles_{angle_index:04d}.npz",
    )

    # Evitar recalcular resultados existentes
    if os.path.exists(out_name):
        return {
            "u0_index": u0_index,
            "angle_index": angle_index,
            "u0": u0_true,
            "file": out_name,
            "status": "already_exists",
        }

    # La misma ventana en unidades de tE para todos los eventos
    t = np.linspace(
        -3.5 * tE_true,
        3.5 * tE_true,
        10_000,
    )

    run_grid_and_save_npz_kepler(
        out_npz_path=out_name,
        t=t,
        t0_true=t0_true,
        u0_true=u0_true,
        tE_true=tE_true,
        phi_true=phi_true,
        i_true=lambda_xi,
        qflux_true=qflux_true,
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
        "u0_index": u0_index,
        "angle_index": angle_index,
        "u0": u0_true,
        "file": out_name,
        "status": "finished",
    }


# ============================================================
# Programa principal
# ============================================================

def main():
    home_path = os.path.expanduser("~")

    max_workers = min(12, os.cpu_count() or 1)

    # Las realizaciones angulares se generan una sola vez.
    # Se usan exactamente las mismas para todos los valores de u0.
    phi_samples, theta_samples, lambda_samples = (
        generate_angular_samples(
            n_angles=N_angles,
            seed=RANDOM_SEED,
        )
    )

    for tE_true in [150.0]:

        directory = os.path.join(
            home_path,
            "binary_source",
            "results",
            f"scan_u0_angles_tE{int(tE_true)}",
        )

        os.makedirs(directory, exist_ok=True)

        # Guardar los ángulos usados para poder reconstruir
        # y verificar todas las realizaciones.
        angles_file = os.path.join(
            directory,
            "angular_samples.npz",
        )

        if not os.path.exists(angles_file):
            np.savez_compressed(
                angles_file,
                angle_index=np.arange(N_angles),
                phi=phi_samples,
                theta=theta_samples,
                lambda_xi=lambda_samples,
                sin_lambda=np.sin(lambda_samples),
                random_seed=RANDOM_SEED,
            )

        tasks = []

        for u0_index, u0_true in enumerate(u0_grid):
            for angle_index in range(N_angles):

                tasks.append(
                    (
                        u0_index,
                        angle_index,
                        float(u0_true),
                        float(tE_true),
                        float(phi_samples[angle_index]),
                        float(theta_samples[angle_index]),
                        float(lambda_samples[angle_index]),
                        directory,
                    )
                )

        print(
            f"Iniciando scan para tE={tE_true:g} días\n"
            f"  N_u0       = {N_u0}\n"
            f"  N_angles   = {N_angles}\n"
            f"  N_tasks    = {len(tasks)}\n"
            f"  N_periods  = {len(P_grid)}\n"
            f"  workers    = {max_workers}"
        )

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:

            futures = {
                executor.submit(
                    run_single_configuration,
                    task,
                ): task
                for task in tasks
            }

            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                task = futures[future]

                (
                    u0_index,
                    angle_index,
                    u0_true,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = task

                try:
                    result = future.result()

                    print(
                        f"[{completed:05d}/{len(tasks):05d}] "
                        f"u0[{u0_index:02d}]="
                        f"{result['u0']:.5f}, "
                        f"ang={angle_index:04d}, "
                        f"estado={result['status']}"
                    )

                except Exception as error:
                    print(
                        f"[ERROR] "
                        f"u0_index={u0_index}, "
                        f"u0={u0_true:.5f}, "
                        f"angle_index={angle_index}: "
                        f"{error}"
                    )

        print(
            f"Scan angular terminado para "
            f"tE={tE_true:g} días."
        )


if __name__ == "__main__":
    main()
