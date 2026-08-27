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
u0_true = 0.1


# ============================================================
# Ángulos orbitales / xallarap
# ============================================================

phi_true = 0.0
theta_true = 0.0

lambda_xi_fixed = 0.5 * np.pi

qflux_value = 0.0


# ============================================================
# Sistema físico
#
# Mantenemos fija la masa total:
#
#       Mtot = M1 + M2
#
# y variamos
#
#       q = M2 / M1
#
# de modo que
#
#       M1 = Mtot / (1 + q)
#       M2 = q Mtot / (1 + q)
#
# ============================================================

Mtot_fixed = 3.0      # Msun
rEhat = 5.0           # AU


# ============================================================
# Resolución de las grillas
# ============================================================

N_q = 100
N_P = 100


# ============================================================
# Barrido en período
#
# 10 d -> 100000 d
# ============================================================

P_grid = np.logspace(
    1,
    5,
    N_P,
)


# ============================================================
# Barrido en mass ratio
#
# q = M2 / M1
#
# 10^-4 -> 1
# ============================================================

q_grid = np.logspace(
    -4,
    0,
    N_q,
)


# ============================================================
# Worker
#
# Cada proceso:
#
#   1. recibe un único q
#   2. calcula M1 y M2 manteniendo Mtot fija
#   3. ejecuta todo el barrido en P
#   4. guarda un NPZ independiente
#
# ============================================================

def run_single_q(task):

    k, q_true, tE_true, directory = task


    # ========================================================
    # Archivo de salida
    # ========================================================

    out_name = os.path.join(
        directory,
        f"scan_kepler_q_{k:03d}.npz",
    )


    # ========================================================
    # Evitar recalcular resultados existentes
    # ========================================================

    if os.path.exists(out_name):

        return {
            "index": k,
            "q": q_true,
            "file": out_name,
            "status": "already_exists",
        }


    # ========================================================
    # Mantener Mtot fija
    #
    # q = M2/M1
    #
    # Mtot = M1 + M2 = M1(1+q)
    # ========================================================

    M1_true = (
        Mtot_fixed
        /
        (
            1.0
            + q_true
        )
    )

    M2_true = (
        Mtot_fixed
        * q_true
        /
        (
            1.0
            + q_true
        )
    )


    # ========================================================
    # Fracción de la órbita relativa descrita por source 1
    #
    # a1 / a_rel = q / (1+q)
    #
    # Para qflux=0, esta es particularmente relevante porque
    # la señal está asociada a la trayectoria de la fuente
    # luminosa.
    # ========================================================

    a1_over_arel = (
        q_true
        /
        (
            1.0
            + q_true
        )
    )


    # ========================================================
    # Time grid
    #
    # Mantengo exactamente la misma convención que en el
    # barrido en u0 para que ambos experimentos sean
    # directamente comparables.
    # ========================================================

    t = np.linspace(
        -3.5 * tE_true,
        3.5 * tE_true,
        10_000,
    )


    # ========================================================
    # Ejecutar barrido completo en P
    # ========================================================

    run_grid_and_save_npz_kepler(

        out_npz_path=out_name,

        t=t,

        # ----------------------------------------------------
        # Parámetros PSPL base
        # ----------------------------------------------------

        t0_true=t0_true,

        u0_true=float(
            u0_true
        ),

        tE_true=float(
            tE_true
        ),


        # ----------------------------------------------------
        # Geometría orbital
        # ----------------------------------------------------

        phi_true=phi_true,

        i_true=float(
            lambda_xi_fixed
        ),

        qflux_true=qflux_value,

        theta_true=theta_true,


        # ----------------------------------------------------
        # Sistema físico
        # ----------------------------------------------------

        M1_Msun=float(
            M1_true
        ),

        M2_Msun=float(
            M2_true
        ),

        rEhat_AU=rEhat,


        # ----------------------------------------------------
        # Barrido en período
        # ----------------------------------------------------

        P_grid=P_grid,


        # ----------------------------------------------------
        # Fotometría
        #
        # Mismos valores que en el scan de u0
        # ----------------------------------------------------

        msource_true=24.0,

        mtotal_true=24.0,


        # ----------------------------------------------------
        # Kepler-consistente
        # ----------------------------------------------------

        override_xiE=None,


        # ----------------------------------------------------
        # Configuración del experimento
        # ----------------------------------------------------

        set_flux_from_truth_photometry=True,

        rms_on_magnification=True,
    )


    return {

        "index": k,

        "q": q_true,

        "M1": M1_true,

        "M2": M2_true,

        "a1_over_arel": a1_over_arel,

        "file": out_name,

        "status": "finished",
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Home automático
    # ========================================================

    home_path = os.path.expanduser("~")


    # ========================================================
    # Número máximo de procesos
    #
    # Igual que en el scan de u0.
    # ========================================================

    max_workers = min(
        12,
        os.cpu_count() or 1,
    )


    # ========================================================
    # Barrido en tE
    #
    # Por ahora un único tE.
    # Después se puede extender directamente.
    # ========================================================

    for tE_true in [150]:


        # ====================================================
        # Directorio de salida
        # ====================================================

        directory = os.path.join(

            home_path,

            "binary_source",

            "results",

            f"scan_q_Mtotfixed_tE{int(tE_true)}",
        )


        os.makedirs(
            directory,
            exist_ok=True,
        )


        # ====================================================
        # Construir tareas
        #
        # Una tarea por q.
        # ====================================================

        tasks = [

            (
                k,
                float(q_true),
                float(tE_true),
                directory,
            )

            for k, q_true
            in enumerate(q_grid)
        ]


        # ====================================================
        # Información inicial
        # ====================================================

        print()

        print("=" * 80)

        print(
            f"Iniciando scan en mass ratio para "
            f"tE={tE_true} días"
        )

        print("=" * 80)

        print(
            f"Mtot fija       = {Mtot_fixed:.3f} Msun"
        )

        print(
            f"u0 fijo         = {u0_true:.4f}"
        )

        print(
            f"q_flux          = {qflux_value:.3f}"
        )

        print(
            f"q range         = "
            f"[{q_grid.min():.3e}, "
            f"{q_grid.max():.3e}]"
        )

        print(
            f"N_q             = {N_q}"
        )

        print(
            f"P range         = "
            f"[{P_grid.min():.3e}, "
            f"{P_grid.max():.3e}] d"
        )

        print(
            f"N_P             = {N_P}"
        )

        print(
            f"procesos        = {max_workers}"
        )

        print(
            f"output          = {directory}"
        )

        print("=" * 80)

        print()


        # ====================================================
        # Ejecutar en paralelo sobre q
        # ====================================================

        with ProcessPoolExecutor(

            max_workers=max_workers,

            mp_context=mp.get_context(
                "spawn"
            ),

        ) as executor:


            # =================================================
            # Enviar tareas
            # =================================================

            futures = {

                executor.submit(
                    run_single_q,
                    task,
                ):
                task

                for task in tasks
            }


            # =================================================
            # Leer a medida que terminan
            # =================================================

            for completed, future in enumerate(

                as_completed(
                    futures
                ),

                start=1,

            ):

                task = futures[
                    future
                ]

                k, q_true, _, _ = task


                try:

                    result = future.result()


                    # =========================================
                    # Resultado terminado
                    # =========================================

                    if result[
                        "status"
                    ] == "finished":

                        print(

                            f"[{completed:03d}/{len(tasks):03d}] "

                            f"q={result['q']:.4e}, "

                            f"M1={result['M1']:.5f}, "

                            f"M2={result['M2']:.5f}, "

                            f"a1/a_rel="
                            f"{result['a1_over_arel']:.4e}, "

                            f"estado="
                            f"{result['status']}"
                        )


                    # =========================================
                    # Archivo ya existente
                    # =========================================

                    else:

                        print(

                            f"[{completed:03d}/{len(tasks):03d}] "

                            f"q={result['q']:.4e}, "

                            f"estado="
                            f"{result['status']}"
                        )


                # =============================================
                # Error en una tarea
                # =============================================

                except Exception as error:

                    print(

                        f"[ERROR] "
                        f"k={k}, "
                        f"q={q_true:.4e}: "
                        f"{error}"
                    )


        # ====================================================
        # Terminado
        # ====================================================

        print()

        print("=" * 80)

        print(
            f"Scan en mass ratio terminado "
            f"para tE={tE_true} días."
        )

        print(
            f"Resultados:"
        )

        print(
            directory
        )

        print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
