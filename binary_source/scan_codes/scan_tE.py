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
# Global experiment parameters
# ============================================================

t0_true = 50.0

# Orbital angles / xallarap
phi_true = 0.0
theta_true = 0.0
lambda_xi_fixed = 0.5 * np.pi

# Physical system
M1 = 2.0
M2 = 1.0
rEhat = 5.0

# ============================================================
# Orbital-period scan
# ============================================================

P_grid = np.logspace(
    -5,
    5,
    200,
)  # 10 days -> 100000 days


# ============================================================
# Einstein-timescale scan
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
# Time configuration
# ============================================================

N_time_points = 10_000
time_window_tE = 3.5


def format_float_for_path(value, precision=4):
    """
    Convert a floating-point number into a safe string for use
    in directory and file names.

    Ejemplos
    --------
    0.01 -> 0p0100
    10.5 -> 10p5000
    """
    return f"{value:.{precision}f}".replace(".", "p")


def run_single_tE(task):
    """
    Run the complete P scan for a fixed combination
    de u0 y tE.

    Each process writes a different NPZ file.

    Parameters
    ----------
    task : tuple
        Contiene:

        - u0 index
        - tE index
        - true u0
        - true tE
        - output directory
    """

    iu0, itE, u0_true, tE_true, directory = task

    tE_tag = format_float_for_path(tE_true, precision=4)

    out_name = os.path.join(
        directory,
        f"scan_kepler_tE_{itE:03d}_{tE_tag}d.npz",
    )

    # --------------------------------------------------------
    # Avoid recomputing existing results
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

    # Adjust according to the available CPU cores and memory
    max_workers = min(
        12,
        os.cpu_count() or 1,
    )

    all_tasks = []

    # ========================================================
    # Create one directory for each fixed u0 value
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

        # One task for each tE value
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
    print(f"Number of tE values: {len(tE_grid)}")
    print(
        f"Rango de tE         : "
        f"{tE_grid.min():.3f} - {tE_grid.max():.3f} days"
    )
    print(f"Number of P values : {len(P_grid)}")
    print(
        f"Rango de P          : "
        f"{P_grid.min():.3f} - {P_grid.max():.3f} days"
    )
    print(f"Tareas totales      : {total_tasks}")
    print(f"Concurrent processes: {max_workers}")
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
