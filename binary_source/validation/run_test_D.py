# ============================================================
# run_D_validation_tests.py
#
# Tests para validar D_BSPL-PSPL
#
# TEST 1:
#   Convergencia con N_points
#
# TEST 2:
#   Convergencia con la ventana temporal W tE
#
# TEST 3:
#   Límite físico q_M -> 0
#
# TEST 4:
#   Usar el scan existente en u0 para comprobar:
#
#       D ~ RMS(residual) / RMS(signal)
#
#   y encontrar automáticamente dos eventos con
#   RMS casi idéntico pero D muy diferente.
#
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
import pandas as pd

import os
import glob


# ============================================================
# IMPORTANTE
# ============================================================
#
# En degeneracy_fit.py asegurate de que cualquier ejemplo
# ejecutable que haya al final esté dentro de:
#
# if __name__ == "__main__":
#     ...
#
# De lo contrario, con multiprocessing + spawn cada worker
# volverá a ejecutar ese código.
#
# ============================================================


# ============================================================
# PARÁMETROS FIDUCIALES
# ============================================================

t0_true = 50.0
tE_true = 150.0

phi_true = 0.0
theta_true = 0.0
i_true = 0.5 * np.pi

qflux_true = 0.0

M1 = 2.0
M2 = 1.0

Mtot = M1 + M2

rEhat = 5.0

msource_true = 24.0
mtotal_true = 24.0


# ============================================================
# MULTIPROCESSING
# ============================================================

MAX_WORKERS = min(
    12,
    os.cpu_count() or 1,
)


# ============================================================
# OUTPUT
# ============================================================

home = os.path.expanduser("~")

output_root = os.path.join(
    home,
    "binary_source",
    "results",
    "D_validation",
)

os.makedirs(
    output_root,
    exist_ok=True,
)


# ============================================================
# CONFIGURACIÓN DE LOS TESTS
# ============================================================


# ------------------------------------------------------------
# TEST 1: convergencia en N
# ------------------------------------------------------------

N_VALUES = np.array([
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
])


# Usamos varios sistemas para no demostrar convergencia
# solamente en un caso particular.

CONVERGENCE_CASES = [

    {
        "name": "case_A",
        "u0": 0.03,
        "P_over_tE": 0.3,
    },

    {
        "name": "case_B",
        "u0": 0.10,
        "P_over_tE": 1.0,
    },

    {
        "name": "case_C",
        "u0": 1.00,
        "P_over_tE": 5.0,
    },

]


W_N_TEST = 3.5


# ------------------------------------------------------------
# TEST 2: convergencia con la ventana
# ------------------------------------------------------------

WINDOW_VALUES = np.array([
    1.0,
    1.5,
    2.0,
    3.5,
    5.0,
    7.0,
    10.0,
])


# Mantener aproximadamente constante la densidad temporal.

N_REFERENCE = 10000
W_REFERENCE = 3.5


# ------------------------------------------------------------
# TEST 3: q_M -> 0
# ------------------------------------------------------------

Q_GRID = np.logspace(
    -5,
    0,
    45,
)


P_OVER_TE_Q_TEST = np.array([
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
])


U0_Q_TEST = 0.1

N_Q_TEST = 5000
W_Q_TEST = 3.5


# ------------------------------------------------------------
# TEST 4: scan existente en u0
# ------------------------------------------------------------

U0_SCAN_DIRECTORY = os.path.join(
    home,
    "binary_source",
    "results",
    f"scan_u0_tE{int(tE_true)}",
)


# Ancho del bin en log10(RMS) para buscar dos eventos
# aproximadamente con el mismo RMS.
#
# 0.02 dex ~ diferencia máxima de ~4.7%.

SAME_RMS_BIN_WIDTH_DEX = 0.02


# ============================================================
# D Y ESCALAS L2
# ============================================================

def compute_D_diagnostics(
    t,
    A_truth,
    A_fit,
):

    t = np.asarray(
        t,
        dtype=float,
    )

    A_truth = np.asarray(
        A_truth,
        dtype=float,
    )

    A_fit = np.asarray(
        A_fit,
        dtype=float,
    )


    valid = (
        np.isfinite(t)
        &
        np.isfinite(A_truth)
        &
        np.isfinite(A_fit)
    )


    t = t[valid]
    A_truth = A_truth[valid]
    A_fit = A_fit[valid]


    if len(t) < 2:

        return {
            "D": np.nan,
            "RMS_residual": np.nan,
            "RMS_signal": np.nan,
            "L2_residual": np.nan,
            "L2_signal": np.nan,
        }


    order = np.argsort(t)

    t = t[order]
    A_truth = A_truth[order]
    A_fit = A_fit[order]


    residual = (
        A_truth
        -
        A_fit
    )


    signal = (
        A_truth
        -
        1.0
    )


    numerator = np.trapezoid(
        residual**2,
        x=t,
    )


    denominator = np.trapezoid(
        signal**2,
        x=t,
    )


    if (
        denominator <= 0
        or
        not np.isfinite(denominator)
    ):

        D = np.nan

    else:

        D = np.sqrt(
            numerator
            /
            denominator
        )


    RMS_residual = np.sqrt(
        np.mean(
            residual**2
        )
    )


    RMS_signal = np.sqrt(
        np.mean(
            signal**2
        )
    )


    duration = (
        t[-1]
        -
        t[0]
    )


    if duration > 0:

        L2_residual = np.sqrt(
            numerator
            /
            duration
        )

        L2_signal = np.sqrt(
            denominator
            /
            duration
        )

    else:

        L2_residual = np.nan
        L2_signal = np.nan


    return {

        "D":
            float(D),

        "RMS_residual":
            float(RMS_residual),

        "RMS_signal":
            float(RMS_signal),

        "L2_residual":
            float(L2_residual),

        "L2_signal":
            float(L2_signal),

    }


# ============================================================
# LEER UNA CORRIDA NPZ
# ============================================================

def read_metrics_from_npz(
    filename,
):

    with np.load(
        filename,
        allow_pickle=False,
    ) as d:

        t = d[
            "t"
        ].astype(float)

        P_grid = d[
            "P_grid"
        ].astype(float)

        success = d[
            "SUCCESS"
        ].astype(bool)

        if (
            "A_truth_grid" not in d.files
            or
            "A_fit_grid" not in d.files
        ):

            raise KeyError(
                f"\n{filename}\n"
                "does not contain A_truth_grid/A_fit_grid.\n"
                "These validation tests require store_curves=True."
            )


        A_truth_grid = d[
            "A_truth_grid"
        ].astype(float)

        A_fit_grid = d[
            "A_fit_grid"
        ].astype(float)


        RMS_saved = (

            d["RMS"].astype(float)

            if "RMS" in d.files

            else np.full(
                len(P_grid),
                np.nan,
            )

        )


        D_saved = (

            d["D"].astype(float)

            if "D" in d.files

            else np.full(
                len(P_grid),
                np.nan,
            )

        )


    rows = []


    for j, P in enumerate(P_grid):

        if not success[j]:
            continue


        metrics = compute_D_diagnostics(

            t=t,

            A_truth=
            A_truth_grid[
                j
            ],

            A_fit=
            A_fit_grid[
                j
            ],

        )


        rows.append({

            "jP":
                j,

            "P":
                float(P),

            "D":
                metrics[
                    "D"
                ],

            "D_saved":
                float(
                    D_saved[j]
                ),

            "RMS_residual":
                metrics[
                    "RMS_residual"
                ],

            "RMS_saved":
                float(
                    RMS_saved[j]
                ),

            "RMS_signal":
                metrics[
                    "RMS_signal"
                ],

            "L2_residual":
                metrics[
                    "L2_residual"
                ],

            "L2_signal":
                metrics[
                    "L2_signal"
                ],

        })


    return rows


# ============================================================
# CREAR GRID TEMPORAL
# ============================================================

def make_time_grid(
    t0,
    tE,
    W,
    N,
):

    # Ventana centrada explícitamente en t0.

    return np.linspace(

        t0
        -
        W * tE,

        t0
        +
        W * tE,

        int(N),

    )


# ============================================================
# WORKER PARA TESTS DE CONVERGENCIA
# ============================================================

def run_single_validation(task):

    (
        test_name,
        case_name,
        u0,
        P_over_tE,
        W,
        N_points,
        out_name,

    ) = task


    P = (
        P_over_tE
        *
        tE_true
    )


    if not os.path.exists(
        out_name
    ):

        t = make_time_grid(

            t0=t0_true,

            tE=tE_true,

            W=W,

            N=N_points,

        )


        run_grid_and_save_npz_kepler(

            out_npz_path=
            out_name,

            t=t,

            t0_true=
            t0_true,

            u0_true=
            float(u0),

            tE_true=
            tE_true,

            phi_true=
            phi_true,

            i_true=
            float(i_true),

            qflux_true=
            qflux_true,

            theta_true=
            theta_true,

            M1_Msun=
            M1,

            M2_Msun=
            M2,

            rEhat_AU=
            rEhat,

            P_grid=
            np.array(
                [P],
                dtype=float,
            ),

            msource_true=
            msource_true,

            mtotal_true=
            mtotal_true,

            override_xiE=
            None,

            set_flux_from_truth_photometry=
            True,

            rms_on_magnification=
            True,

            store_curves=
            True,

        )


    rows = read_metrics_from_npz(
        out_name
    )


    if len(rows) != 1:

        raise RuntimeError(
            f"Expected one successful P in {out_name}"
        )


    row = rows[0]


    row.update({

        "test":
            test_name,

        "case":
            case_name,

        "u0":
            float(u0),

        "P_over_tE":
            float(P_over_tE),

        "W":
            float(W),

        "N_points":
            int(N_points),

        "file":
            out_name,

    })


    return row


# ============================================================
# WORKER DEL TEST q_M -> 0
# ============================================================

def run_q_validation(task):

    (
        k,
        q,
        out_name,

    ) = task


    # --------------------------------------------------------
    # Mantener masa total fija
    # --------------------------------------------------------

    M1_q = (
        Mtot
        /
        (1.0 + q)
    )


    M2_q = (
        q
        *
        M1_q
    )


    P_grid_q = (
        P_OVER_TE_Q_TEST
        *
        tE_true
    )


    if not os.path.exists(
        out_name
    ):

        t = make_time_grid(

            t0=t0_true,

            tE=tE_true,

            W=W_Q_TEST,

            N=N_Q_TEST,

        )


        run_grid_and_save_npz_kepler(

            out_npz_path=
            out_name,

            t=t,

            t0_true=
            t0_true,

            u0_true=
            U0_Q_TEST,

            tE_true=
            tE_true,

            phi_true=
            phi_true,

            i_true=
            float(i_true),

            qflux_true=
            qflux_true,

            theta_true=
            theta_true,

            M1_Msun=
            M1_q,

            M2_Msun=
            M2_q,

            rEhat_AU=
            rEhat,

            P_grid=
            P_grid_q,

            msource_true=
            msource_true,

            mtotal_true=
            mtotal_true,

            override_xiE=
            None,

            set_flux_from_truth_photometry=
            True,

            rms_on_magnification=
            True,

            store_curves=
            True,

        )


    rows = read_metrics_from_npz(
        out_name
    )


    output_rows = []


    for row in rows:

        row.update({

            "q_M":
                float(q),

            "M1":
                float(M1_q),

            "M2":
                float(M2_q),

            "P_over_tE":
                float(
                    row["P"]
                    /
                    tE_true
                ),

            "file":
                out_name,

        })


        output_rows.append(
            row
        )


    return output_rows


# ============================================================
# TEST 1
#
# CONVERGENCIA EN N
# ============================================================

def test_N_convergence():

    directory = os.path.join(
        output_root,
        "N_convergence",
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )


    tasks = []


    for case in CONVERGENCE_CASES:

        for N in N_VALUES:

            out_name = os.path.join(

                directory,

                (
                    f"{case['name']}"
                    f"_N{int(N):06d}.npz"
                ),

            )


            tasks.append((

                "N_convergence",

                case[
                    "name"
                ],

                float(
                    case["u0"]
                ),

                float(
                    case["P_over_tE"]
                ),

                float(
                    W_N_TEST
                ),

                int(N),

                out_name,

            ))


    rows = []


    with ProcessPoolExecutor(

        max_workers=
        MAX_WORKERS,

        mp_context=
        mp.get_context(
            "spawn"
        ),

    ) as executor:


        futures = {

            executor.submit(
                run_single_validation,
                task,
            ):
            task

            for task in tasks

        }


        for i, future in enumerate(

            as_completed(
                futures
            ),

            start=1,

        ):

            task = futures[
                future
            ]


            try:

                row = future.result()

                rows.append(
                    row
                )


                print(

                    f"[N] "
                    f"{i:03d}/{len(tasks):03d}  "
                    f"{row['case']}  "
                    f"N={row['N_points']:6d}  "
                    f"D={row['D']:.5e}"

                )


            except Exception as error:

                print(
                    "[ERROR N]",
                    task,
                    error,
                )


    df = pd.DataFrame(
        rows
    )


    outfile = os.path.join(
        output_root,
        "N_convergence.csv",
    )


    df.to_csv(
        outfile,
        index=False,
    )


    return df


# ============================================================
# TEST 2
#
# CONVERGENCIA EN VENTANA
# ============================================================

def test_window_convergence():

    directory = os.path.join(
        output_root,
        "window_convergence",
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )


    tasks = []


    for case in CONVERGENCE_CASES:

        for W in WINDOW_VALUES:


            N_points = int(
                np.round(
                    N_REFERENCE
                    *
                    W
                    /
                    W_REFERENCE
                )
            )


            out_name = os.path.join(

                directory,

                (
                    f"{case['name']}"
                    f"_W{W:.1f}"
                    f"_N{N_points:06d}.npz"
                ),

            )


            tasks.append((

                "window_convergence",

                case[
                    "name"
                ],

                float(
                    case["u0"]
                ),

                float(
                    case["P_over_tE"]
                ),

                float(W),

                int(
                    N_points
                ),

                out_name,

            ))


    rows = []


    with ProcessPoolExecutor(

        max_workers=
        MAX_WORKERS,

        mp_context=
        mp.get_context(
            "spawn"
        ),

    ) as executor:


        futures = {

            executor.submit(
                run_single_validation,
                task,
            ):
            task

            for task in tasks

        }


        for i, future in enumerate(

            as_completed(
                futures
            ),

            start=1,

        ):

            try:

                row = future.result()

                rows.append(
                    row
                )


                print(

                    f"[W] "
                    f"{i:03d}/{len(tasks):03d}  "
                    f"{row['case']}  "
                    f"W={row['W']:4.1f}  "
                    f"D={row['D']:.5e}"

                )


            except Exception as error:

                print(
                    "[ERROR W]",
                    error,
                )


    df = pd.DataFrame(
        rows
    )


    outfile = os.path.join(
        output_root,
        "window_convergence.csv",
    )


    df.to_csv(
        outfile,
        index=False,
    )


    return df


# ============================================================
# TEST 3
#
# q_M -> 0
# ============================================================

def test_q_limit():

    directory = os.path.join(
        output_root,
        "q_limit",
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )


    tasks = []


    for k, q in enumerate(
        Q_GRID
    ):

        out_name = os.path.join(

            directory,

            f"q_{k:03d}.npz",

        )


        tasks.append((

            k,

            float(q),

            out_name,

        ))


    rows = []


    with ProcessPoolExecutor(

        max_workers=
        MAX_WORKERS,

        mp_context=
        mp.get_context(
            "spawn"
        ),

    ) as executor:


        futures = {

            executor.submit(
                run_q_validation,
                task,
            ):
            task

            for task in tasks

        }


        for i, future in enumerate(

            as_completed(
                futures
            ),

            start=1,

        ):

            task = futures[
                future
            ]


            try:

                new_rows = (
                    future.result()
                )

                rows.extend(
                    new_rows
                )


                print(

                    f"[q] "
                    f"{i:03d}/{len(tasks):03d}  "
                    f"q={task[1]:.4e}"

                )


            except Exception as error:

                print(
                    "[ERROR q]",
                    task,
                    error,
                )


    df = pd.DataFrame(
        rows
    )


    outfile = os.path.join(
        output_root,
        "q_limit.csv",
    )


    df.to_csv(
        outfile,
        index=False,
    )


    return df


# ============================================================
# TEST 4A
#
# LEER SCAN EXISTENTE EN u0
#
# Comprobar:
#
# D ~ RMS_residual / RMS_signal
# ============================================================

def analyze_existing_u0_scan():

    pattern = os.path.join(

        U0_SCAN_DIRECTORY,

        "scan_kepler_u0_*.npz",

    )


    files = sorted(
        glob.glob(
            pattern
        )
    )


    if len(files) == 0:

        raise FileNotFoundError(
            pattern
        )


    rows = []


    for i, filename in enumerate(
        files,
        start=1,
    ):


        with np.load(
            filename,
            allow_pickle=False,
        ) as d:


            if "truth" in d.files:

                truth = d[
                    "truth"
                ].astype(float)

                u0 = float(
                    truth[1]
                )

                tE_file = float(
                    truth[2]
                )

            else:

                raise KeyError(
                    f"No truth in {filename}"
                )


        file_rows = (
            read_metrics_from_npz(
                filename
            )
        )


        for row in file_rows:

            row.update({

                "u0":
                    u0,

                "tE":
                    tE_file,

                "P_over_tE":
                    row["P"]
                    /
                    tE_file,

                "file":
                    filename,

            })


            # -----------------------------------------------
            # Aproximación RMS/RMS
            # -----------------------------------------------

            if (
                row[
                    "RMS_signal"
                ] > 0
            ):

                row[
                    "D_from_RMS_ratio"
                ] = (

                    row[
                        "RMS_residual"
                    ]

                    /

                    row[
                        "RMS_signal"
                    ]

                )

            else:

                row[
                    "D_from_RMS_ratio"
                ] = np.nan


            # -----------------------------------------------
            # Identidad exacta usando la misma norma integral
            # -----------------------------------------------

            if (
                row[
                    "D"
                ] > 0
            ):

                row[
                    "event_scale_from_D"
                ] = (

                    row[
                        "L2_residual"
                    ]

                    /

                    row[
                        "D"
                    ]

                )

            else:

                row[
                    "event_scale_from_D"
                ] = np.nan


            rows.append(
                row
            )


        print(

            f"[u0 existing] "
            f"{i:03d}/{len(files):03d}  "
            f"u0={u0:.5f}"

        )


    df = pd.DataFrame(
        rows
    )


    outfile = os.path.join(
        output_root,
        "u0_normalization_test.csv",
    )


    df.to_csv(
        outfile,
        index=False,
    )


    return df


# ============================================================
# TEST 4B
#
# ENCONTRAR DOS EVENTOS CON RMS SIMILAR Y D MUY DIFERENTE
# ============================================================

def find_same_RMS_pair(
    df,
):

    work = df.copy()


    valid = (

        np.isfinite(
            work["RMS_residual"]
        )

        &

        np.isfinite(
            work["D"]
        )

        &

        (
            work["RMS_residual"]
            >
            0
        )

        &

        (
            work["D"]
            >
            0
        )

    )


    work = work.loc[
        valid
    ].copy()


    work[
        "logRMS"
    ] = np.log10(

        work[
            "RMS_residual"
        ]

    )


    work[
        "logD"
    ] = np.log10(

        work[
            "D"
        ]

    )


    # --------------------------------------------------------
    # Agrupar en bins estrechos de RMS
    # --------------------------------------------------------

    min_log_rms = (
        work[
            "logRMS"
        ].min()
    )


    work[
        "RMS_bin"
    ] = np.floor(

        (
            work[
                "logRMS"
            ]

            -
            min_log_rms

        )

        /

        SAME_RMS_BIN_WIDTH_DEX

    ).astype(int)


    best_pair = None
    best_delta_logD = -np.inf


    for _, group in work.groupby(
        "RMS_bin"
    ):


        if len(group) < 2:
            continue


        i_min = group[
            "logD"
        ].idxmin()


        i_max = group[
            "logD"
        ].idxmax()


        delta_logD = (

            group.loc[
                i_max,
                "logD"
            ]

            -

            group.loc[
                i_min,
                "logD"
            ]

        )


        if (
            delta_logD
            >
            best_delta_logD
        ):

            best_delta_logD = (
                delta_logD
            )

            best_pair = (

                work.loc[
                    i_min
                ],

                work.loc[
                    i_max
                ],

            )


    if best_pair is None:

        raise RuntimeError(
            "No same-RMS pair found."
        )


    row_A, row_B = best_pair


    print()
    print("=" * 80)

    print("SAME-RMS / DIFFERENT-D PAIR")

    print("=" * 80)

    print()

    print("CASE A")

    print(
        f"u0       = {row_A['u0']:.8g}"
    )

    print(
        f"P/tE     = {row_A['P_over_tE']:.8g}"
    )

    print(
        f"RMS      = {row_A['RMS_residual']:.8e}"
    )

    print(
        f"D        = {row_A['D']:.8e}"
    )

    print()

    print("CASE B")

    print(
        f"u0       = {row_B['u0']:.8g}"
    )

    print(
        f"P/tE     = {row_B['P_over_tE']:.8g}"
    )

    print(
        f"RMS      = {row_B['RMS_residual']:.8e}"
    )

    print(
        f"D        = {row_B['D']:.8e}"
    )

    print()

    print(
        "RMS ratio B/A = "
        f"{row_B['RMS_residual']/row_A['RMS_residual']:.4f}"
    )

    print(
        "D ratio B/A   = "
        f"{row_B['D']/row_A['D']:.4e}"
    )

    print(
        "Delta log10 D = "
        f"{best_delta_logD:.4f} dex"
    )

    print("=" * 80)


    # ========================================================
    # Guardar las curvas de ambos casos
    # ========================================================

    pair_data = {}


    for label, row in [

        ("A", row_A),

        ("B", row_B),

    ]:


        filename = row[
            "file"
        ]

        jP = int(
            row[
                "jP"
            ]
        )


        with np.load(
            filename,
            allow_pickle=False,
        ) as d:

            pair_data[
                f"t_{label}"
            ] = d[
                "t"
            ].astype(float)


            pair_data[
                f"A_truth_{label}"
            ] = d[
                "A_truth_grid"
            ][
                jP
            ].astype(float)


            pair_data[
                f"A_fit_{label}"
            ] = d[
                "A_fit_grid"
            ][
                jP
            ].astype(float)


        pair_data[
            f"u0_{label}"
        ] = float(
            row[
                "u0"
            ]
        )


        pair_data[
            f"P_over_tE_{label}"
        ] = float(
            row[
                "P_over_tE"
            ]
        )


        pair_data[
            f"RMS_{label}"
        ] = float(
            row[
                "RMS_residual"
            ]
        )


        pair_data[
            f"D_{label}"
        ] = float(
            row[
                "D"
            ]
        )


        pair_data[
            f"RMS_signal_{label}"
        ] = float(
            row[
                "RMS_signal"
            ]
        )


    outfile = os.path.join(
        output_root,
        "same_RMS_different_D_pair.npz",
    )


    np.savez_compressed(
        outfile,
        **pair_data,
    )


    print(
        "\nSaved pair:"
    )

    print(
        outfile
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)

    print(
        "D_BSPL-PSPL VALIDATION TESTS"
    )

    print(
        f"workers = {MAX_WORKERS}"
    )

    print(
        f"output  = {output_root}"
    )

    print("=" * 80)


    # ========================================================
    # 1. Sampling convergence
    # ========================================================

    print(
        "\nTEST 1: N convergence\n"
    )

    test_N_convergence()


    # ========================================================
    # 2. Window convergence
    # ========================================================

    print(
        "\nTEST 2: window convergence\n"
    )

    test_window_convergence()


    # ========================================================
    # 3. Physical q -> 0 limit
    # ========================================================

    print(
        "\nTEST 3: q_M -> 0\n"
    )

    test_q_limit()


    # ========================================================
    # 4. Existing u0 scan
    # ========================================================

    print(
        "\nTEST 4: normalization using existing u0 scan\n"
    )

    df_u0 = (
        analyze_existing_u0_scan()
    )


    find_same_RMS_pair(
        df_u0
    )


    print()
    print("=" * 80)

    print(
        "ALL VALIDATION TESTS FINISHED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
