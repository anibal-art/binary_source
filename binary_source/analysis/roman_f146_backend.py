#!/usr/bin/env python3
"""
Roman F146 noise backend for the binary-source Roman experiment.

This module deliberately provides ONLY the F146 single-epoch uncertainty
needed by the BSPL -> PSPL Asimov/LRT analysis.

Magnitude convention
--------------------
The binary_source Roman experiment uses F146 AB magnitudes.

Therefore we call RomanF146Noise.sigma_mag_ab() directly.

We intentionally do NOT call apply_roman_f146_photometry() here because
that production roman_rubin helper starts from Vega photometry and also
implements saturation/SNR-based photometric filtering.  Those additional
selection effects are scientifically useful, but introducing them at this
stage would change more than the noise model and would prevent a clean
old-vs-Pandeia comparison.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_ROMAN_RUBIN_ROOT = (
    Path.home() / "roman_rubin"
)


def get_roman_rubin_root() -> Path:
    root = Path(
        os.environ.get(
            "ROMAN_RUBIN_ROOT",
            str(DEFAULT_ROMAN_RUBIN_ROOT),
        )
    ).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"roman_rubin repository not found: {root}"
        )

    return root


def roman_rubin_git_commit(root=None):
    if root is None:
        root = get_roman_rubin_root()

    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:
        return None


_NOISE_MODEL = None


def get_noise_model():
    global _NOISE_MODEL

    if _NOISE_MODEL is not None:
        return _NOISE_MODEL

    root = get_roman_rubin_root()

    if str(root) not in sys.path:
        sys.path.insert(
            0,
            str(root),
        )

    from stellar_population.noise_models.roman_f146 import (
        RomanF146Noise,
    )

    _NOISE_MODEL = RomanF146Noise(
        detector="median",
        background_level="medium",
        systematic_floor_frac=0.0,
    )

    return _NOISE_MODEL


def sigma_f146_pandeia_ab(magnitude_ab):
    """
    Roman F146 single-epoch magnitude uncertainty from the
    roman_rubin Pandeia R2026.1 model.

    Parameters
    ----------
    magnitude_ab : scalar or array
        F146 AB magnitude.

    Returns
    -------
    sigma_mag : ndarray
        Magnitude uncertainty.
    """

    mag = np.asarray(
        magnitude_ab,
        dtype=float,
    )

    if np.any(~np.isfinite(mag)):
        raise ValueError(
            "Non-finite F146 AB magnitude."
        )

    model = get_noise_model()

    sigma = np.asarray(
        model.sigma_mag_ab(mag),
        dtype=float,
    )

    if np.any(
        ~np.isfinite(sigma)
        | (sigma <= 0.0)
    ):
        bad = mag[
            ~np.isfinite(sigma)
            | (sigma <= 0.0)
        ]

        raise ValueError(
            "Pandeia F146 model returned invalid "
            f"uncertainties for magnitudes {bad[:10]}"
        )

    return sigma


def provenance():
    model = get_noise_model()
    root = get_roman_rubin_root()

    return {
        "backend": "roman_rubin_pandeia",
        "roman_rubin_root": str(root),
        "roman_rubin_commit": roman_rubin_git_commit(
            root
        ),
        "noise_class": (
            "stellar_population.noise_models."
            "roman_f146.RomanF146Noise"
        ),
        "grid_path": str(
            getattr(
                model,
                "grid_path",
                "",
            )
        ),
        "metadata_path": str(
            getattr(
                model,
                "metadata_path",
                "",
            )
        ),
        "magnitude_system": "AB",
        "detector": "median",
        "background_level": "medium",
        "systematic_floor_frac": 0.0,
    }


if __name__ == "__main__":

    print(provenance())

    for mag in [
        19.0,
        21.0,
        23.0,
    ]:
        print(
            f"F146_AB={mag:4.1f}  "
            f"sigma="
            f"{float(sigma_f146_pandeia_ab(mag)):.10e}"
        )
