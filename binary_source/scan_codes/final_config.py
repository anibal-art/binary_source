"""
Common configuration for publication-quality final BSPL--PSPL scans.

The result directory is automatically tied to the Git commit used
when the scan is executed.
"""

from pathlib import Path
import os
import subprocess

import numpy as np


# ============================================================
# Repository / provenance
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_git_state():

    try:

        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short=12",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        dirty = bool(
            status.strip()
        )

    except Exception:

        # No git on the compute node: the submitting shell passes the state,
        # BINARY_SOURCE_GIT_COMMIT=$(git rev-parse --short=12 HEAD) and
        # BINARY_SOURCE_GIT_DIRTY=$(git status --porcelain | wc -l).
        commit = os.environ.get("BINARY_SOURCE_GIT_COMMIT", "unknown")
        dirty = (commit == "unknown"
                 or os.environ.get("BINARY_SOURCE_GIT_DIRTY", "1").strip() not in ("0", ""))

    return commit, dirty


CODE_COMMIT, CODE_DIRTY = get_git_state()


# ============================================================
# Output
# ============================================================

RUN_TAG = (
    f"final_{CODE_COMMIT}"
    if not CODE_DIRTY
    else f"final_{CODE_COMMIT}_dirty"
)

RESULTS_ROOT = (
    REPO_ROOT
    / "results"
    / RUN_TAG
)


# ============================================================
# Standard final numerical resolution
# ============================================================

FINAL_N_2D = 300
FINAL_N_1D = 400

FINAL_N_TIME = 10_000


# ============================================================
# Photocenter / perturbative-limit scan
# ============================================================

FINAL_N_PHOTOCENTER_QM = 200
FINAL_N_PHOTOCENTER_XI = 120

FINAL_XI_OVER_U0 = np.logspace(
    -4,
    0,
    FINAL_N_PHOTOCENTER_XI,
)


# ============================================================
# Parallelism
# ============================================================

N_CPU = os.cpu_count() or 1

MAX_WORKERS = max(
    1,
    min(
        12,
        N_CPU - 1
        if N_CPU > 1
        else 1,
    ),
)

# Override (e.g. the CPUs allocated by SLURM; os.cpu_count() counts the
# whole node).
MAX_WORKERS = int(os.environ.get("BINARY_SOURCE_MAX_WORKERS", MAX_WORKERS))


def final_output_dir(name):
    """
    Return/create one output directory for a final experiment.
    """

    path = (
        RESULTS_ROOT
        / str(name)
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path
