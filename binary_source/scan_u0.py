from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Barrido en período
P_grid = np.logspace(1, 5, 60)
# Barrido en u0
N_u0 = 25
u0_grid = np.logspace(-2, 1, N_u0)

def run_single_u0(task):
    """
    Ejecuta el barrido en P para un único valor de u0.

    Cada proceso escribe un archivo NPZ diferente.
    """
    k, u0_true, tE_true, directory = task

    out_name = os.path.join(
        directory,
        f"scan_kepler_u0_{k:03d}.npz",
    )

    # Evitar recalcular resultados existentes
    if os.path.exists(out_name):
        return {
            "index": k,
            "u0": u0_true,
            "file": out_name,
            "status": "already_exists",
        }

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
        i_true=float(lambda_xi_fixed),
        qflux_true=0,
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
        "index": k,
        "u0": u0_true,
        "file": out_name,
        "status": "finished",
    }


def main():
    home_path = os.path.expanduser("~")

    # Ajustar según la cantidad de núcleos y memoria disponible
    max_workers = min(12, os.cpu_count() or 1)

    for tE_true in [150]:

        directory = os.path.join(
            home_path,
            "binary_source",
            "results",
            f"scan_u0_tE{int(tE_true)}",
        )

        os.makedirs(directory, exist_ok=True)

        tasks = [
            (k, float(u0_true), float(tE_true), directory)
            for k, u0_true in enumerate(u0_grid)
        ]

        print(
            f"Iniciando scan para tE={tE_true} días "
            f"con {max_workers} procesos."
        )

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:

            futures = {
                executor.submit(run_single_u0, task): task
                for task in tasks
            }

            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                task = futures[future]
                k, u0_true, _, _ = task

                try:
                    result = future.result()

                    print(
                        f"[{completed:02d}/{len(tasks):02d}] "
                        f"u0={result['u0']:.4f}, "
                        f"estado={result['status']}"
                    )

                except Exception as error:
                    print(
                        f"[ERROR] k={k}, "
                        f"u0={u0_true:.4f}: {error}"
                    )

        print(
            f"Scan en u0 terminado para tE={tE_true} días."
        )


if __name__ == "__main__":
    main()
