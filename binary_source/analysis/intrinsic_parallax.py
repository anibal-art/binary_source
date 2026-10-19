"""Ideal/noiseless parallax evaluators used only for intrinsic degeneracy maps. Observational Roman fitting is delegated to roman_rubin."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from pyLIMA.models import PSPL_model
from binary_source.analysis import intrinsic_fit_tools as base
THIS_FILE = Path(__file__).resolve()
DEFAULT_RR_ROOTS = (
    THIS_FILE.parents[3] / 'roman_rubin',
    Path('/share/storage3/rubin/microlensing/romanrubin/roman_rubin'),
    Path('/export/storage3/rubin/microlensing/romanrubin/roman_rubin'),
)

def resolve_roman_ephemerides(path=None):
    """
    Resolve the CURRENT Roman ephemeris from the roman_rubin repo.

    Priority:
      1. explicit path
      2. ROMAN_EPHEMERIDES environment variable
      3. ROMAN_RUBIN_ROOT/ephemerides/Roman_positions.npy
      4. known CHE roman_rubin roots
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    env_eph = os.environ.get('ROMAN_EPHEMERIDES')
    if env_eph:
        candidates.append(Path(env_eph))
    for env_name in (
        'ROMAN_RUBIN_REPO',
        'ROMAN_RUBIN_ROOT',
    ):
        env_root = os.environ.get(env_name)
        if env_root:
            candidates.append(
                Path(env_root) / 'ephemerides' / 'Roman_positions.npy'
            )
    for root in DEFAULT_RR_ROOTS:
        candidates.append(root / 'ephemerides' / 'Roman_positions.npy')
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    attempted = '\n'.join((str(x) for x in candidates))
    raise FileNotFoundError('Could not resolve Roman ephemerides. Attempted:\n' + attempted)

def _load_ephemerides_cached(path_string):
    path = Path(path_string)
    eph = np.asarray(np.load(path), dtype=float)
    if eph.ndim != 2 or eph.shape[1] != 4:
        raise ValueError(f'Roman ephemerides must have shape (N, 4): [JD, RA, Dec, distance_AU]. Got {eph.shape}.')
    if not np.all(np.isfinite(eph)):
        raise ValueError('Roman ephemerides contain non-finite values.')
    if np.any(np.diff(eph[:, 0]) <= 0.0):
        raise ValueError('Roman ephemeris JD column must be strictly increasing.')
    return eph

def load_roman_ephemerides(path=None):
    resolved = resolve_roman_ephemerides(path)
    eph = _load_ephemerides_cached(str(resolved))
    return (resolved, eph)

def build_roman_parallax_event(t, ephemerides_path=None, name='Roman_parallax'):
    """
    Build the same Roman event geometry used by the LRT, but attach the
    externally maintained Roman/L2 ephemeris before constructing a
    pyLIMA Full-parallax model.
    """
    t = np.asarray(t, dtype=float)
    ev, tel = base.roman.build_truth_event(t)
    tel.name = str(name)
    path, eph = load_roman_ephemerides(ephemerides_path)
    if getattr(tel, 'location', None) != 'Space':
        raise RuntimeError("Roman parallax telescope must have location='Space'.")
    tel.spacecraft_name = 'L2'
    tel.spacecraft_positions = {'astrometry': [], 'photometry': eph}
    return (ev, tel, path)

def _plain(value):
    return np.asarray(getattr(value, 'value', value), dtype=float)

def _check_magnification(A, label):
    A = _plain(A)
    if np.any(~np.isfinite(A)):
        raise RuntimeError(f'{label}: non-finite magnification.')
    if np.any(A <= 0.0):
        raise RuntimeError(f'{label}: non-positive magnification.')
    return A

class PSPLParallaxEvaluator:

    def __init__(self, t, t0_reference, ephemerides_path=None):
        self.t = np.asarray(t, dtype=float)
        self.t0_reference = float(t0_reference)
        self.event, self.telescope, self.ephemerides_path = build_roman_parallax_event(self.t, ephemerides_path=ephemerides_path, name='Roman_PSPL_parallax')
        self.model = PSPL_model.PSPLmodel(self.event, blend_flux_parameter='ftotal', parallax=['Full', self.t0_reference], double_source=['None', self.t0_reference])
        self.model.define_model_parameters()

    def magnification(self, t0, u0, tE, piEN, piEE):
        params = [float(t0), float(u0), float(tE), float(piEN), float(piEE), 1.0, 1.0]
        py_params = self.model.compute_pyLIMA_parameters(params)
        A = self.model.model_magnification(self.telescope, py_params)
        return _check_magnification(A, 'PSPL+parallax')

class PhysicalCircularBSPLParallax:

    def __init__(self, t, t0_reference, ephemerides_path=None):
        self.t = np.asarray(t, dtype=float)
        self.t0_reference = float(t0_reference)
        self.event, self.telescope, self.ephemerides_path = build_roman_parallax_event(self.t, ephemerides_path=ephemerides_path, name='Roman_BSPL_parallax')
        self.model = PSPL_model.PSPLmodel(self.event, blend_flux_parameter='ftotal', parallax=['Full', self.t0_reference], double_source=['Circular', self.t0_reference])
        self.model.define_model_parameters()

    def magnification(self, t0, u0, tE, piEN, piEE, omega, q_mass=None, q_flux=None, theta=None, phi=None, inclination=None):
        fid = levels.fiducial_values()
        if q_mass is None:
            q_mass = fid['q_mass']
        if q_flux is None:
            q_flux = fid['q_flux']
        if theta is None:
            theta = fid['theta']
        if phi is None:
            phi = fid['phi']
        if inclination is None:
            inclination = fid['inclination']
        omega = float(omega)
        if omega <= 0.0:
            raise ValueError('omega must be positive.')
        P_days = 2.0 * np.pi / omega
        a_rel_AU = float(base.roman.a_from_P_kepler_days(P_days, base.roman.MTOT_MSUN))
        xi_rel = a_rel_AU / base.roman.REHAT_AU
        xi_para = xi_rel * np.cos(float(theta))
        xi_perp = xi_rel * np.sin(float(theta))
        params = [float(t0), float(u0), float(tE), float(piEN), float(piEE), float(xi_para), float(xi_perp), float(omega), float(phi), float(inclination), float(q_mass), float(q_flux), 1.0, 1.0]
        py_params = self.model.compute_pyLIMA_parameters(params)
        A = self.model.model_magnification(self.telescope, py_params)
        A = _plain(A)
        A = A / (1.0 + float(q_flux))
        return _check_magnification(A, 'BSPL+parallax')
