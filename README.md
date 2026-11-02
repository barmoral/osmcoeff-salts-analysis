# osmcoeff-salts-analysis

Three workspaces. Pick one and work inside it; all read the same (heavy) data through links.

| folder | what it is |
|---|---|
| `original_analysis/` | the OLD method, frozen: `HP_analysis_replicates.py`, its run scripts, `HP_tests/`, plot scripts. Git tag `old-method` marks the code that produced the original results. |
| `FBP_analysis/` | flat-bottom-potential analysis (ported from `openff-dev/FBP_membrane`): `FBP_analysis_replicates.py`, `run_scripts/`, `FBP_tests/`, `results/<case>/` |
| `thermo_analysis/` | the NEW method: `HP_analysis_replicates_thermo.py`, `HP_tests_thermo/`, `run_scripts/`, `results/<case>/{new,legacy_check}/`, `refpaper_mods/` (cluster scripts, reference papers; not versioned) |

Shared data (never copied, never versioned): `HP_results/` (trajectories + the old analysis outputs, side by side),
`FBP_results/`, `structures/` (salt_data files, packmol inputs, pdbs; tracked in git).
Each workspace holds `HP_results` and `structures` as links to `../`. Recreate them with `bash make_links.sh`.

## Run (from inside a workspace)
    cd original_analysis && python run_HP_analysis_paperbox.py        # writes next to the trajectories, as before
    cd thermo_analysis   && python run_scripts/run_HP_analysis_paperbox.py              # -> results/<case>/new/
    cd thermo_analysis   && HP_LEGACY=1 python run_scripts/run_HP_analysis_paperbox.py  # -> results/<case>/legacy_check/

`legacy_check` runs the new file with the OLD A and box-average molality: it should reproduce the old numbers, so any
difference between `new` and `legacy_check` comes only from the fixes.

## Tests
    cd original_analysis && pytest            # HP_tests/
    cd thermo_analysis   && pytest            # HP_tests_thermo/  (add -m "not slow" to skip the end-to-end run)
