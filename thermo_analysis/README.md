# thermo_analysis

Run everything from this folder (links `HP_results` and `structures` resolve relative to it).

- `HP_analysis_replicates_thermo.py` - the analysis (see its header for what changed vs the original).
- `run_scripts/run_HP_analysis*.py` - one per system. They read trajectories from `HP_results/<case>/...` and write ONLY
  to `results/<case>/new/` (or `legacy_check/` with `HP_LEGACY=1`). `CHECKS`, `RESERVOIR_TARGET` are optional settings.
- `run_scripts/run_HP_analysis_refpaper.py` + `literature_reference_nacl_chi0.py` - the tuned reference-paper run; draws
  the paper's chi = 0 curves via `reference=`.
- `HP_tests_thermo/` - synthetic, literature-free tests (`pytest`).
- `refpaper_mods/` - cluster scripts (dispatch, run script, N_water tuning), the notebook, the papers, and the earlier
  `_refpaper` analysis copies (superseded by the thermo module). Ignored by git.
