# Run configurations

One script per experiment; each run is a JSON file here, in a folder named
after the script. Never copy a script to change its parameters: add a config.

- `scan_intrinsic_xallarap_vs_pspl_parallax/` for
  `decision_tests/scan_intrinsic_xallarap_vs_pspl_parallax.py --config ...`
  (`tE30.json` is the manuscript grid).
- `control_classical_jerk_parallax/` for
  `decision_tests/control_classical_jerk_parallax.py --config ...`

Each run writes the configuration it used next to its outputs when the script
supports it (`run_config.json`).
