# Roman F146 Pandeia update checkpoint

Date: 2026-10-16

## Purpose

This checkpoint updates the Roman observational weighting used by the
BSPL -> PSPL Asimov experiment without changing the underlying
microlensing truth model.

The goal at this stage is deliberately narrow:

- preserve the existing BSPL physical family;
- preserve the Roman timestamps and fit window;
- preserve the PSPL fitting procedure;
- preserve the F146 AB magnitude convention;
- replace only the historical effective F146 uncertainty prescription
  by the Pandeia-based `RomanF146Noise` model from the sibling
  `roman_rubin` repository.

This is the first step toward regenerating the Roman grid and then
building a calibrated BSPL-vs-PSPL likelihood-ratio test.

## Repositories

`binary_source` starting HEAD:

    7a7d540feaebbadc48463fdf2a7113d65f95e87e

`roman_rubin` reference HEAD:

    dd60771691b9b32f8997bcea0fffa472246042c9

The Pandeia backend is loaded from:

    stellar_population.noise_models.roman_f146.RomanF146Noise

with the production grid:

    roman_rubin/photometry/data/roman_f146_pandeia_2026p1.csv

and metadata:

    roman_rubin/photometry/data/roman_f146_pandeia_2026p1.json

Configuration used here:

- detector: median
- background: medium
- systematic fractional floor: 0
- input magnitudes: F146 AB

## Scope of the code change

New adapter:

    binary_source/analysis/roman_f146_backend.py

Updated analysis:

    binary_source/analysis/roman_bspl_pspl_asimov.py

The analysis now supports two in-process noise backends:

    legacy
    pandeia

The historical backend is retained for reproducibility.

The Pandeia backend calls `sigma_mag_ab()` directly because the
current `binary_source` Roman Asimov experiment is already defined in
F146 AB magnitudes.

At this checkpoint we intentionally do NOT call the full
`apply_roman_f146_photometry()` production helper, because that would
also introduce Vega-to-AB conversion, saturation/SNR selection, and
epoch rejection. Those effects will be considered separately after the
noise-only update is reproduced.

## Direct noise comparison

Single-epoch uncertainties:

| F146 AB | legacy sigma_mag | Pandeia sigma_mag |
|---:|---:|---:|
| 19 | 2.7267419246e-03 | 3.3060188572e-03 |
| 21 | 9.3883263351e-03 | 9.0815549333e-03 |
| 23 | 4.6877022959e-02 | 3.2286602280e-02 |

The two prescriptions therefore do not differ by a constant scale.

## Controlled Asimov A/B test

Configuration:

- tE = 30 d
- u0 = 0.1
- P/tE = 1
- P = 30 d
- F146 AB = 19, 21, 23
- qM = 0.5
- qf = 0
- projected Einstein radius = 5 AU
- parallax disabled
- identical Roman times
- 8390 Roman epochs
- identical BSPL truth magnification
- identical noiseless `mag_truth`

For all three source magnitudes:

    max |mag_truth_legacy - mag_truth_pandeia| = 0

Therefore the observed differences below are caused only by the
photometric weighting and the consequent re-optimization of the PSPL
fit.

### Results

| F146 | Delta chi2 legacy | Delta chi2 Pandeia | ratio new/old |
|---:|---:|---:|---:|
| 19 | 5.2497053622e+06 | 1.6891687038e+06 | 0.32176448 |
| 21 | 6.7598998144e+05 | 5.1576116230e+05 | 0.76297161 |
| 23 | 4.4377804939e+04 | 5.9443608559e+04 | 1.33948961 |

Roman-weighted mismatch:

| F146 | D_Roman_eff legacy | D_Roman_eff Pandeia |
|---:|---:|---:|
| 19 | 5.3146084619e-02 | 5.8596998813e-02 |
| 21 | 4.9028196862e-02 | 5.1157576809e-02 |
| 23 | 4.3754295764e-02 | 4.8579535767e-02 |

Best-fit PSPL parameters also change because the weighting changes.

For F146=19:

    legacy   u0=0.09481060  tE=31.37683 d
    Pandeia  u0=0.10318711  tE=29.41844 d

For F146=21:

    legacy   u0=0.09217187  tE=32.06530 d
    Pandeia  u0=0.09398759  tE=31.52969 d

For F146=23:

    legacy   u0=0.08844765  tE=33.24538 d
    Pandeia  u0=0.09185106  tE=32.16230 d

## Interpretation

The updated Roman weighting cannot be reproduced by applying one global
rescaling to the old Delta chi2 grid.

The noise difference is magnitude dependent, and the best-fitting PSPL
solution itself changes when the weights change.

Consequently the full Roman Asimov grid must be recomputed with the
Pandeia backend.

The current Delta chi2 is naturally interpretable as an Asimov
likelihood-ratio statistic:

    T_Asimov
      = chi2_PSPL,best - chi2_BSPL,true
      = Delta chi2_Asimov

because the noiseless BSPL Asimov dataset has chi2_BSPL,true = 0.

## Deliberately unchanged at this checkpoint

The following have NOT yet been changed:

- Roman timestamp prescription;
- 8390-epoch high-cadence selection;
- event-centered +/-3.5 tE fitting window;
- BSPL physical parameterization;
- PSPL fitter;
- parallax treatment;
- Roman/L2 ephemerides;
- saturation filtering;
- S/N-based epoch rejection;
- source blending prescription;
- intrinsic BSPL grid.

In particular, parallax remains disabled. Therefore the updated
Roman/L2 ephemerides in `roman_rubin` do not enter this noise-only
comparison.

## Next steps

1. Expose `legacy` / `pandeia` through the command-line interface.
2. Recompute exactly the existing matched Roman grid:
   - 133 u0 nodes with u0 <= 1;
   - 60 physical periods;
   - F146 = 19, 21, 23;
   - total 23,940 PSPL fits.
3. Store the new production separately from the historical result.
4. Compare old and Pandeia grids cell by cell.
5. Regenerate the Roman manuscript figure with the Pandeia result.
6. Reinterpret the resulting Delta chi2 grid as `T_Asimov`.
7. Build a Monte-Carlo LRT calibration under H0=PSPL and H1=BSPL to
   obtain a false-positive-controlled threshold and detection power.

## Status

Checkpoint passed.

The Pandeia backend imports successfully, the production grid and
metadata are resolved, and the controlled A/B test isolates the
photometric-noise update from all other aspects of the Roman
simulation.
