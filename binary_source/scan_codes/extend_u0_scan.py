# ============================================================
# EXTENSION DEL SCAN HACIA u0 PEQUEÑO
#
# Scan original:
#     u0 = 1e-2 ... 1e1
#
# Extension:
#     u0 = 1e-4 ... 1e-2
#
# No repetimos u0=1e-2.
# ============================================================

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
# PARAMETROS
# ============================================================

t0_true = 50.0
tE_true = 150.0

phi_true = 0.0
theta_true = 0.0
lambda_xi_fixed = 0.5 * np.pi

M1 = 2.0
M2 = 1.0

rEhat = 5.0
qflux_value = 0.0


# ============================================================
# MISMO P_grid QUE EL SCAN ORIGINAL
# ============================================================

N_P = 100

P_grid = np.logspace(
    1,
    5,
    N_P,
)


# ============================================================
# NUEVA GRILLA EN u0
#
# Extendemos dos décadas hacia abajo.
# ============================================================

N_u0_extension = 60

u0_grid_extension = np.logspace(
    -4,
    -2,
    N_u0_extension + 1,
)[:-1]   # excluye u0 = 1e-2, que ya existe


print(
    "u0 extension:",
    u0_grid_extension.min(),
    "->",
    u0_grid_extension.max(),
)


# ============================================================
# WORKER
# ============================================================

def run_single_u0(task):

    k, u0_true, directory = task

    out_name = os.path.join(
        directory,
        f"scan_kepler_u0_low_{k:03d}.npz",
    )

    if os.path.exists(out_name):

        return {
            "index": k,
            "u0": u0_true,
            "file": out_name,
            "status": "already_exists",
        }


    # Misma ventana temporal que el scan original

    t = np.linspace(
        t0_true - 3.5 * tE_true,
        t0_true + 3.5 * tE_true,
        10_000,
    )


    run_grid_and_save_npz_kepler(

        out_npz_path=out_name,

        t=t,

        t0_true=t0_true,
        u0_true=float(u0_true),
        tE_true=tE_true,

        phi_true=phi_true,
        i_true=float(lambda_xi_fixed),
        qflux_true=qflux_value,
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


# ============================================================
# MAIN
# ============================================================

def main():

    home = os.path.expanduser("~")


    directory = os.path.join(
        home,
        "binary_source",
        "results",
        f"scan_u0_tE{int(tE_true)}_lowu0",
    )


    os.makedirs(
        directory,
        exist_ok=True,
    )


    max_workers = min(
        12,
        os.cpu_count() or 1,
    )


    tasks = [
        (
            k,
            float(u0),
            directory,
        )
        for k, u0 in enumerate(
            u0_grid_extension
        )
    ]


    print("=" * 70)

    print(
        f"Extending u0 scan toward small impact parameters:"
    )

    print(
        f"{u0_grid_extension.min():.3e}"
        f" <= u0 <= "
        f"{u0_grid_extension.max():.3e}"
    )

    print(
        f"N new u0 values = {len(u0_grid_extension)}"
    )

    print("=" * 70)


    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:


        futures = {
            executor.submit(
                run_single_u0,
                task,
            ): task
            for task in tasks
        }


        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):

            task = futures[future]

            try:

                result = future.result()

                print(
                    f"[{completed:02d}/{len(tasks):02d}] "
                    f"u0={result['u0']:.5e} "
                    f"{result['status']}"
                )

            except Exception as error:

                print(
                    f"[ERROR] "
                    f"u0={task[1]:.5e}: "
                    f"{error}"
                )


    print()
    print("Finished.")
    print(directory)


if __name__ == "__main__":
    main()
