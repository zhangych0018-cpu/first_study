# SWS Bayesian Optimization

## DSG W-Band SWS Problem

This repository now exposes a single user-facing pipeline for a W-band double-staggered grating slow-wave structure based on Ren et al., 2025 IEEE TED, "A W-Band Backward Wave Oscillator Based on Carbon Nanotube Cold Cathode".

The optimization stack remains:

- LHS / Sobol / local perturbation initial design
- independent GP / `ModelListGP`
- Matern-5/2 ARD kernel
- constrained qNEHVI
- mock / CST dual backend
- failed simulation logging and recovery
- baselines, ablations, robust optimization, and multifidelity calibration

## Relationship to Ren 2025

The current optimization problem focuses on the cold-structure DSG geometry and cold-test style indicators.

- Structure family: double-staggered grating (DSG)
- Target band: `96-110 GHz`
- Nominal target frequency: `100 GHz`
- Target mode: `TM21-like`

CNT cold cathode gun variables and PIC hot-test variables are not default optimization variables in this stage.

## DSG Parameters

The main design vector is:

- `W`: total transverse width / grating width
- `P`: period length
- `T`: beam tunnel height
- `G`: grating thickness
- `H`: grating height

Reference design:

```text
W = 3.2 mm
P = 0.8 mm
T = 0.3 mm
G = 0.2 mm
H = 0.6 mm
```

## User-Facing Entry Points

The repository now keeps only the DSG mainline as the public run path.

Mock demo:

```bash
cd C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization
conda activate sws_predict_env
python scripts/run_mock_demo.py
```

Generic BO entry:

```bash
python scripts/run_bo.py
```

Result analysis:

```bash
python scripts/analyze_results.py
```

Baselines:

```bash
python scripts/run_baselines.py
```

Ablation:

```bash
python scripts/run_ablation.py --quick
```

Initial data generation:

```bash
python scripts/generate_initial_data.py
```

## Default Output Directories

- Mock demo: `data/results/dsg_mock_demo`
- Generic BO run: `data/results/dsg_default_run`
- CST run: `data/results/dsg_cst_run`
- Baselines: `data/results/dsg_baselines`
- Ablation: `data/results/dsg_ablation`

## Configuration Files

The public configs are now:

- `config/config.yaml`
- `config/mock_config.yaml`
- `config/cst_config.yaml`

All three are DSG-first.

## Real CST Integration Checklist

Before switching from mock to CST, confirm:

1. The machine is Windows.
2. CST Studio Suite is installed and licensed.
3. Python can access `pywin32` / `win32com`.
4. A parameterized DSG CST template exists.
5. Template parameters match `W, P, T, G, H`.
6. CST exports the following files for each run:
   - `dispersion_tm21.txt`
   - `dispersion_fundamental.txt`
   - `kc_tm21.txt`
   - `kc_fundamental.txt`
   - `sparameters.txt`
7. Each simulation can write into an independent `run_id` directory.
8. Frequency units and column names match parser expectations.
9. Timeout and retry settings are agreed.
10. CST stdout / stderr / COM error logs should be preserved.

Run the CST backend with:

```bash
python scripts/run_bo.py --backend cst --config config/cst_config.yaml
```

## DSG Postprocessing File Format

Expected files:

- `dispersion_tm21.txt`
- `dispersion_fundamental.txt`
- `kc_tm21.txt`
- `kc_fundamental.txt`
- `sparameters.txt`

The DSG parser computes:

- `Kc_TM21_mean`
- `sync_error`
- `vp_TM21_std`
- `mode_ratio`
- `f_TM21_ghz`
- `f_fund_ghz`
- `ohmic_loss_mean`
- `S11_max`

The parser checks for:

- missing files
- empty files
- missing columns
- NaN / non-numeric values
- frequency unit conversion to GHz
- empty working-band slices
- target-frequency neighborhood failures
- mode-ratio computation failures

Parser tests can be run with:

```bash
python -m pytest -q tests/test_dsg_postprocessing.py tests/test_postprocessing.py
```

## TM21-Like Mode Competition

The workflow explicitly tracks target-mode and fundamental-mode competition using:

- `mode_ratio`
- `f_TM21_ghz`
- `f_fund_ghz`

The default BO loop still uses the main three objectives plus the S11 constraint. `mode_ratio` is recorded for downstream filtering, diagnostics, and later constraint extensions.

## Important Limitation

Mock data is only for software verification, interface validation, and workflow testing.

**Mock results cannot support physical conclusions, paper conclusions, or device-performance claims.**

Any real scientific conclusion must come from CST full-wave results, cold-test data, or measured experimental data.
