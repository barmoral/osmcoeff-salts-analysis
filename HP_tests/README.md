# tests/

Fast, self-contained checks for the HP analysis code. No cluster, no real
trajectories, no network — synthetic systems are built in memory with answers
you can work out by hand, so a failure means the code changed.

## Run

```bash
conda activate osmotic-analysis
pip install pytest            # once
cd <project root>             # the folder holding HP_analysis_replicates.py
pytest                        # ~9 s, 65 tests
pytest -v                     # names of everything that ran
pytest tests/test_profiles.py # one file
pytest -k fold                # one topic
```

`pytest.ini` at the project root sets `testpaths = tests` and silences the
MDAnalysis import warnings.

## Layout

| file | covers |
|---|---|
| `conftest.py` | fixtures: synthetic Universes (uniform / Gaussian / different box length) and a real `salt_data_*.py` built by `build_salt_data.py` |
| `test_helpers.py` | filename and k-tag conventions, molar↔molal, osmotic physics limits |
| `test_profiles.py` | folding, binning, conservation, profile widths |
| `test_salt_data.py` | generated-file schema and the design equations behind k, δz and N |
| `test_run_analysis.py` | `run_analysis()` end to end on synthetic trajectories, plus every guard |

## What the fixtures give you

- **`uniform_universe`** — ions spread evenly. Exact answers: flat folded
  profile, integral = `N_PAIRS`, concentration = N/(A·Lz).
- **`gaussian_universe`** — ions in a Gaussian of exactly the width `k_HP`
  implies, σ = 1/√(2K). The measured peak should equal `cmax_ideal`.
- **`offcentre_universe`** — same Gaussian in a 14.4 nm box, used to prove the
  fold centre is read from the trajectory rather than hard-coded.
- **`salt_data_module`** — runs `build_salt_data.py` for real and imports the
  result, so generator and consumer are tested together.

## Tests that exist because something broke

Each of these reproduces a real failure. Keep them.

| test | the bug |
|---|---|
| `test_fold_centre_comes_from_the_trajectory` | `distance=72` folded a 100 Å box about 72 Å — no error, wrong profile |
| `test_count_and_molarity_profiles_share_a_grid` | two binning formulas → `yerr (144,) vs y (100,)` |
| `test_dz_nm_actually_controls_the_bins` | `dz_nm` was accepted and ignored |
| `test_hist_to_molar_matches_the_profile_path` | bootstrap normalisation was 1.45× high with the mol/L conversion inverted |
| `test_small_quantities_survived_serialisation` | `round(v, 8)` turned `mass_water_kg` (~1e-22) into 0.0 |
| `test_molarity_is_derived_not_copied_molality` | stored molarity ≈ molality, 10–23% wrong at high concentration |
| `test_osm_experimental_uses_the_temperature_it_is_given` | experimental curve computed at 300 K while the run was 298.15 K |
| `test_wrong_k_tag_in_directory_name_raises` | analysing a k=0.68 run with a salt_data that says 1.5567 |
| `test_wrong_water_count_raises` | pointing at trajectories from a different box |

## Adding a test

Reach for an existing fixture rather than a new one, and prefer an assertion
with a closed-form answer (an integral, a limit, a ratio) over a number copied
from a previous run — the latter locks in whatever was there, bug included.

If a formula is hard to test because it is buried in a closure, lift it to a
module-level function. `hist_to_molar` was extracted from
`bootstrap_histograms` for exactly that reason, and the normalisation bug is
now a one-line test instead of a multi-minute end-to-end run.

## Before you push a change

```bash
pytest && python run_HP_analysis.py    # PROFILES_ONLY=True for the second
```
