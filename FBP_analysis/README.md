# FBP_analysis

Flat-bottom-potential (FBP) osmotic pressure / coefficient analysis. Ported from `openff-dev/FBP_membrane/`
(`fbp_analysis.py` + `run_FBP_analysis.py`) into the reorganised layout, same shape as `thermo_analysis/`.

| path | what |
|---|---|
| `FBP_analysis_replicates.py` | the library (list of changes vs the FBP_membrane copy is in its docstring) |
| `run_scripts/run_FBP_analysis.py` | edit-and-run usage script (system, molalities, replicates, optional overrides) |
| `FBP_tests/` | unit tests on synthetic inputs (`pytest`, ~1 s) |
| `results/<case>/` | outputs; never written next to the trajectories |
| `FBP_results`, `structures` | links to the shared data (`bash ../make_links.sh`) |

## Run (from this folder)
    conda activate osmotic-analysis            # has pymbar; osmotic-analysis-hp does not (but it has pytest)
    python run_scripts/run_FBP_analysis.py     # -> results/FBP_NaCl_TIP3P_144/{final_results_*.json, ocs.png, ops.png, concprof_*.png, ...}
    pytest                                      # use the -hp env for this

Trajectories: `FBP_results/<case>/result_files_<mi>m/md<mi>m_r<i>.{pdb,xtc}` (`<mi>`: 3.5 -> `35`, 4.0 -> `4`).

## Where the numbers come from
k, delta_z, z_center, T, van't Hoff factor and ion-pair count are read from `structures/salt_data_*.py`, the
file the dispatch used. The old FBP_membrane run file hard-coded delta_z = 2.4 nm and T = 300 K; salt_data has
2.4278 nm and 298.15 K. **delta_z matters a lot**: ions stop at the real wall, so with the wrong half-width the
penetration (and the pressure) is wrong (3.5 m NaCl, 3 replicates: ~200 bar / phi ~1.2 with 2.4278 nm, ~600 bar / phi ~3.5 with 2.4 nm).
Overrides are printed as `OVERRIDE`.
