# Roman BSPL LRT catalog runner

This note documents the stable interface between `binary_source` and the
canonical `roman_rubin` backend for the Roman binary-source hypothesis tests.

## Scientific model families

Exactly three observational fit families are used:

- `PSPL`: PSPL without parallax.
- `PSPL_PI`: PSPL with Full parallax.
- `BSPL`: circular binary-source PSPL without parallax.

There is **no BSPL+parallax model** in this project.

Every noisy numerical dataset is fitted by all three families. The expensive
BSPL fit is therefore shared by the two tests.

### Test A

\[
H_0^{(A)}=\mathrm{PSPL},\qquad
H_1^{(A)}=\mathrm{BSPL}.
\]

\[
\Delta\chi_A^2
=
\chi^2_{\rm PSPL}
-
\chi^2_{\rm BSPL},
\qquad
T_A=\max(0,\Delta\chi_A^2).
\]

The raw signed value must always be saved.

### Test B

\[
H_0^{(B)}=\mathrm{PSPL+parallax},\qquad
H_1^{(B)}=\mathrm{BSPL}.
\]

\[
T_B
=
\chi^2_{\rm PSPL+\pi}
-
\chi^2_{\rm BSPL}.
\]

`T_B` is signed and is **not truncated**. Positive values favor BSPL; negative
values favor PSPL+parallax. No Wilks p-value is used by the scientific
calibration.

## Truth subsets

The runner accepts:

- `--truth-kind h0a`: PSPL truth with parallax disabled. This is the
  \(\pi_E=0\) null subset for Test A.
- `--truth-kind h0b`: PSPL truth with Full parallax enabled. This is the null
  family for Test B.
- `--truth-kind h1`: circular BSPL truth with parallax disabled. The same H1
  realization supplies both `T_A` and `T_B`.

Truth is used only to simulate the dataset.

## Critical anti-truth-leakage rule

The runner **never** builds fit initial guesses from the catalog truth. Fit
starts and bounds are supplied externally through a `binary_source` policy
JSON.

This is intentional. The fit policy is part of the implemented statistic and
must later be calibrated under H0 exactly as used under H1.

The policy also supplies one **common data-derived fit reference**:

```json
"fit_reference": {
  "t0": ...,
  "u0": ...,
  "tE": ...
}
```

This reference is identical for PSPL, PSPL+parallax and BSPL and is used to
construct the `event_params` passed to `fit_lc.fit_rubin_roman`. Individual
starts may have different `t0`, `u0` and `tE`, but the reference epoch used by
the model machinery is common. This prevents the parallax reference epoch from
silently depending on truth or on a particular start.

## Canonical roman_rubin calls

The runner preserves the tested integration pattern:

1. Read a materialized catalog event with
   `catalog.event_reader.load_event_from_catalog`.
2. Simulate exactly once with
   `functions_roman_rubin.simulate_event_for_fit`.
3. Extract the numerical Roman light curve once with
   `functions_roman_rubin.extract_lightcurves_for_fit`.
4. Fit copies of that same numerical array through
   `fit_lc.fit_rubin_roman`.
5. Select the minimum chi-square over the start bank for each model family.
6. Construct `T_A` and `T_B`.

The runner checks that the W149 data hash remains unchanged while all three fit
banks are executed.

The canonical versioned ephemeris is

`roman_rubin/ephemerides/Roman_positions_2027_2032_periodic_v1.npy`

with SHA256

`eeb7bcf5300d2f3d4be3595898170975b9da20b5d47a26791af0c5258f307d58`.

## Separation of responsibilities

`roman_rubin` owns:

- Roman cadence and ephemerides;
- photometric uncertainty/noise machinery;
- pyLIMA Event/model construction;
- native `pyLIMA.fits.TRF_fit.TRFfit` fitting through
  `fit_lc.fit_rubin_roman`.

`binary_source` owns:

- the H0/H1 scientific catalogs;
- the three hypothesis families above;
- feature extraction from the observed light curve;
- deterministic candidate/start banks;
- explicit physical fit bounds;
- model-minimum selection;
- `T_A`, `T_B`, H0 calibration and power estimation.

## Fit-policy JSON contract

The runner deliberately does **not** freeze a start policy yet. It receives a
JSON document with one block per fit family:

```json
{
  "schema_version": "roman_bspl_lrt_fit_policy_v1",
  "fit_reference": {
    "t0": 0.0,
    "u0": 0.0,
    "tE": 0.0
  },
  "optimizer_options": {
    "xtol": 1e-8,
    "ftol": 1e-8,
    "gtol": 1e-8,
    "max_nfev": 50000,
    "x_scale": "jac"
  },
  "PSPL": {
    "fit_parallax": false,
    "fit_double_source": false,
    "bounds": {},
    "starts": [
      {
        "label": "feature_start_0",
        "initial_guess": {}
      }
    ]
  },
  "PSPL_PI": {
    "fit_parallax": true,
    "fit_double_source": false,
    "bounds": {},
    "starts": [
      {
        "label": "feature_plus_parallax_0",
        "initial_guess": {}
      }
    ]
  },
  "BSPL": {
    "fit_parallax": false,
    "fit_double_source": true,
    "bounds": {},
    "starts": [
      {
        "label": "feature_plus_bspl_0",
        "initial_guess": {}
      }
    ]
  }
}
```

The empty objects above are placeholders documenting the schema only. They are
not a production policy.

The next development step is to create the feature-based policy generator that
fills these starts from the observed light curve, never from truth.

## Intended repository location

```text
binary_source/
├── binary_source/
│   └── analysis/
│       ├── roman_native_lrt_no_parallax.py
│       └── roman_bspl_lrt_catalog_runner.py
└── docs/
    └── roman_bspl_lrt_catalog_runner.md
```

The existing `roman_native_lrt_no_parallax.py` remains useful as the lower-level
BSPL/native-fit validation driver. This catalog runner documents the
production-facing orchestration: one catalog realization, three fits, two test
statistics.

## Example invocation

Once a fit-policy JSON exists:

```bash
export ROMAN_RUBIN_REPO=/share/storage3/rubin/microlensing/romanrubin/roman_rubin
export PYTHONPATH="$ROMAN_RUBIN_REPO:$PWD:${PYTHONPATH:-}"

python -u \
  binary_source/analysis/roman_bspl_lrt_catalog_runner.py \
  --catalog /path/to/catalog.parquet \
  --catalog-event-index 308 \
  --truth-kind h0b \
  --fit-policy /path/to/fit_policy.json \
  --output /tmp/catalog_308_lrt.json
```

For production, the catalog adapter must provide the exact `custom_system`
fields expected by `roman_rubin`. The already validated event-308 example is
the reference behavior for that interface.
