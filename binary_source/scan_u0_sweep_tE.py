from concurrent.futures import ProcessPoolExecutor, as_completed
from degeneracy_fit import run_grid_and_save_npz_kepler

import multiprocessing as mp
import numpy as np
import os


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

t0_true = 50.0


# ============================================================
# ORBITAL / XALLARAP GEOMETRY
# ============================================================

phi_true = 0.0
theta_true = 0.0
lambda_xi_fixed = 0.5 * np.pi


# ============================================================
# PHYSICAL SYSTEM
# ============================================================

M1 = 2.0
M2 = 1.0
rEhat = 5.0


# ============================================================
# PERIOD GRID
#
# IMPORTANT:
# Keep the same absolute P grid for every tE.
#
# Therefore all tE scans explore the same physical binary
# systems (same P and same xi_rel), while P/tE changes.
# ============================================================

N_P = 60

P_grid = np.logspace(
    1,
    5,
    N_P,
)


# ============================================================
# u0 GRID
# ============================================================

N_u0 = 200

u0_grid = np.logspace(
    -2,
    1,
    N_u0,
)


# ============================================================
# tE GRID
#
# Event timescales in days.
#
# Edit this list as desired.
# ============================================================

tE_grid = np.array(
    [
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
        750.0,
        1000.0,
    ],
    dtype=float,
)


# ------------------------------------------------------------
# Alternative:
#
# tE_grid = np.logspace(
#     np.log10(10.0),
#     np.log10(1000.0),
#     20,
# )
# ------------------------------------------------------------


# ============================================================
# TIME SAMPLING
# ============================================================

N_TIME = 10_000

WINDOW_tE = 3.5


# ============================================================
# PARALLELIZATION
# ============================================================

MAX_WORKERS = min(
    12,
    os.cpu_count() or 1,
)


# ============================================================
# OUTPUT ROOT
#
# Use a new directory so that results from the old tE=150
# calculation are not mixed with this experiment.
# ============================================================

home_path = os.path.expanduser("~")

root_directory = os.path.join(
    home_path,
    "binary_source",
    "results",
    "scan_many_tE_200x200",
)


# ============================================================
# HELPER: tE -> DIRECTORY LABEL
# ============================================================

def format_tE_label(tE):
    """
    Examples
    --------
    150.0 -> '150'
    37.5  -> '37p5'
    """

    label = f"{float(tE):g}"

    return label.replace(
        ".",
        "p",
    )


# ============================================================
# CHECK EXISTING FILE
#
# Useful when restarting a partially completed scan.
# ============================================================

def existing_file_is_compatible(
    filename,
    u0_true,
    tE_true,
):
    """
    Check that an existing file really belongs to the
    requested grid.

    This avoids silently reusing results from a different
    resolution or parameter set.
    """

    try:

        with np.load(
            filename,
            allow_pickle=False,
        ) as d:

            if "P_grid" not in d.files:
                return False

            if "truth" not in d.files:
                return False

            P_existing = d[
                "P_grid"
            ].astype(float)

            truth = d[
                "truth"
            ].astype(float)


        # ----------------------------------------------------
        # P grid
        # ----------------------------------------------------

        if len(P_existing) != len(P_grid):
            return False

        if not np.allclose(
            P_existing,
            P_grid,
            rtol=1e-12,
            atol=0.0,
        ):
            return False


        # ----------------------------------------------------
        # Truth vector begins with
        #
        # [t0, u0, tE, ...]
        # ----------------------------------------------------

        if len(truth) < 3:
            return False


        if not np.isclose(
            truth[0],
            t0_true,
        ):
            return False


        if not np.isclose(
            truth[1],
            u0_true,
        ):
            return False


        if not np.isclose(
            truth[2],
            tE_true,
        ):
            return False


        return True


    except Exception:

        return False


# ============================================================
# SINGLE (tE, u0) WORKER
#
# Each worker performs the complete P grid for one pair
#
#       (tE, u0)
#
# and writes one NPZ.
# ============================================================

def run_single_u0(task):

    (
        k,
        u0_true,
        tE_true,
        directory,
    ) = task


    # ========================================================
    # OUTPUT FILE
    # ========================================================

    out_name = os.path.join(
        directory,
        f"scan_kepler_u0_{k:03d}.npz",
    )


    # ========================================================
    # RESUME SUPPORT
    # ========================================================

    if os.path.exists(
        out_name
    ):

        if existing_file_is_compatible(
            out_name,
            u0_true,
            tE_true,
        ):

            return {
                "index":
                    k,

                "u0":
                    u0_true,

                "tE":
                    tE_true,

                "file":
                    out_name,

                "status":
                    "already_exists",
            }

        else:

            raise RuntimeError(
                "\nExisting file is incompatible with "
                "the requested scan:\n"
                f"{out_name}"
            )


    # ========================================================
    # TIME GRID
    #
    # Centered on t0.
    #
    # This is especially important when comparing many tE.
    # ========================================================

    t = np.linspace(

        t0_true
        - WINDOW_tE * tE_true,

        t0_true
        + WINDOW_tE * tE_true,

        N_TIME,
    )


    # ========================================================
    # RUN P GRID
    # ========================================================

    run_grid_and_save_npz_kepler(

        out_npz_path=out_name,

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        t=t,

        # ----------------------------------------------------
        # Microlensing parameters
        # ----------------------------------------------------

        t0_true=t0_true,

        u0_true=u0_true,

        tE_true=tE_true,

        # ----------------------------------------------------
        # Orbital geometry
        # ----------------------------------------------------

        phi_true=phi_true,

        i_true=float(
            lambda_xi_fixed
        ),

        qflux_true=0,

        theta_true=theta_true,

        # ----------------------------------------------------
        # Physical binary
        # ----------------------------------------------------

        M1_Msun=M1,

        M2_Msun=M2,

        rEhat_AU=rEhat,

        # ----------------------------------------------------
        # Period grid
        # ----------------------------------------------------

        P_grid=P_grid,

        # ----------------------------------------------------
        # Flux
        # ----------------------------------------------------

        msource_true=24.0,

        mtotal_true=24.0,

        # ----------------------------------------------------
        # Kepler-consistent orbital amplitude
        #
        # Internally the quantity currently stored as
        # xiE_of_P corresponds physically to xi_rel.
        # ----------------------------------------------------

        override_xiE=None,

        set_flux_from_truth_photometry=True,

        # ----------------------------------------------------
        # Metrics on magnification
        # ----------------------------------------------------

        rms_on_magnification=True,
    )


    return {

        "index":
            k,

        "u0":
            u0_true,

        "tE":
            tE_true,

        "file":
            out_name,

        "status":
            "finished",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CREATE ROOT DIRECTORY
    # ========================================================

    os.makedirs(
        root_directory,
        exist_ok=True,
    )


    # ========================================================
    # CREATE ONE DIRECTORY FOR EACH tE
    # ========================================================

    directories = {}


    for tE_true in tE_grid:

        label = format_tE_label(
            tE_true
        )


        directory = os.path.join(
            root_directory,
            f"scan_u0_tE{label}",
        )


        os.makedirs(
            directory,
            exist_ok=True,
        )


        directories[
            float(tE_true)
        ] = directory


    # ========================================================
    # BUILD ALL TASKS
    #
    # Number of tasks:
    #
    #       N_tE * N_u0
    #
    # Each task contains N_P fits.
    # ========================================================

    tasks = []


    for tE_true in tE_grid:

        directory = directories[
            float(tE_true)
        ]


        for k, u0_true in enumerate(
            u0_grid
        ):

            tasks.append(
                (
                    k,
                    float(u0_true),
                    float(tE_true),
                    directory,
                )
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    N_tE = len(
        tE_grid
    )


    N_tasks = len(
        tasks
    )


    N_models = (
        N_tE
        * N_u0
        * N_P
    )


    print()

    print("=" * 90)

    print(
        "MULTI-tE BINARY-SOURCE / PSPL SCAN"
    )

    print("=" * 90)


    print(
        f"N_tE     = {N_tE}"
    )


    print(
        f"N_u0     = {N_u0}"
    )


    print(
        f"N_P      = {N_P}"
    )


    print(
        f"N_tasks  = {N_tasks}"
    )


    print(
        f"Total grid cells / fits = "
        f"{N_models:,}"
    )


    print(
        f"N_time   = {N_TIME}"
    )


    print(
        f"Workers  = {MAX_WORKERS}"
    )


    print()


    print(
        "tE values [days]:"
    )


    print(
        tE_grid
    )


    print()


    print(
        "P range [days]:"
    )


    print(
        f"{P_grid.min():.4g}"
        f" -- "
        f"{P_grid.max():.4g}"
    )


    print()


    print(
        "u0 range:"
    )


    print(
        f"{u0_grid.min():.4g}"
        f" -- "
        f"{u0_grid.max():.4g}"
    )


    print()


    print(
        "Root directory:"
    )


    print(
        root_directory
    )


    print("=" * 90)

    print()


    # ========================================================
    # PARALLEL EXECUTION
    #
    # A single ProcessPool is used for all tE values.
    #
    # This allows the scheduler to keep all workers busy
    # instead of creating/destroying a pool for every tE.
    # ========================================================

    with ProcessPoolExecutor(

        max_workers=MAX_WORKERS,

        mp_context=mp.get_context(
            "spawn"
        ),

    ) as executor:


        futures = {

            executor.submit(
                run_single_u0,
                task,
            ):
            task

            for task in tasks
        }


        # ====================================================
        # COUNTERS
        # ====================================================

        n_finished = 0

        n_existing = 0

        n_failed = 0


        # ====================================================
        # COLLECT RESULTS
        # ====================================================

        for completed, future in enumerate(

            as_completed(
                futures
            ),

            start=1,

        ):


            task = futures[
                future
            ]


            (
                k,
                u0_true,
                tE_true,
                directory,
            ) = task


            try:

                result = (
                    future.result()
                )


                if (
                    result["status"]
                    == "finished"
                ):

                    n_finished += 1


                elif (
                    result["status"]
                    == "already_exists"
                ):

                    n_existing += 1


                print(

                    f"["
                    f"{completed:04d}"
                    f"/"
                    f"{N_tasks:04d}"
                    f"] "

                    f"tE="
                    f"{result['tE']:7.2f} d   "

                    f"u0="
                    f"{result['u0']:.5f}   "

                    f"{result['status']}"
                )


            except Exception as error:

                n_failed += 1


                print(

                    f"[ERROR] "

                    f"tE="
                    f"{tE_true:.2f} d, "

                    f"k="
                    f"{k:03d}, "

                    f"u0="
                    f"{u0_true:.5f}: "

                    f"{error}"
                )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 90)

    print(
        "SCAN FINISHED"
    )

    print("=" * 90)


    print(
        f"Finished       : "
        f"{n_finished}"
    )


    print(
        f"Already existed: "
        f"{n_existing}"
    )


    print(
        f"Failed         : "
        f"{n_failed}"
    )


    print(
        f"Total tasks    : "
        f"{N_tasks}"
    )


    print()


    print(
        "Results:"
    )


    for tE_true in tE_grid:

        print(

            f"tE="
            f"{tE_true:7.2f} d"
            f"  ->  "
            f"{directories[float(tE_true)]}"
        )


    print("=" * 90)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
