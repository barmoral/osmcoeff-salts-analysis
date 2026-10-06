# HP_tests_thermo — tests for HP_analysis_replicates_thermo.py

Run on its own (the root pytest.ini still points at the original suite, HP_tests/):

    pytest HP_tests_thermo            # ~40 s, 52 tests
    pytest HP_tests_thermo -m "not slow"

Everything is synthetic and literature-free: analytic profiles, thermodynamic identities and round trips, so it
applies to any box size, salt or water model.

| file | checks |
|---|---|
| test_units_and_A.py | A from dielectric constant and T (scaling laws, textbook water value), the nm^-3 -> M^-1/2 factor direction (regression for the old A bug), stable Debye-Hueckel term vs high precision, `pi_bar` vs the original `osmotic_pressure`, Gibbs-Duhem between the osmotic bracket and ln gamma |
| test_local_molality_and_checks.py | local molality recovers the true m and beats the box average; reservoir density; `checks` ignore/warn/strict for reservoir and box; NaN -> blank/null in the results table |
| test_thermo_roundtrip.py | eq 22 recovered exactly; kappa=0 gives no P correction; eq 23 applied; V_w limits; eq 19 mapping and `rho_w0_ref`; molal alpha round trip; mu_w two ways; Gibbs-Duhem on the reported phi / ln gamma; plots, files, reservoir stamp on every thermo plot; optional reference curves |
| test_run_analysis_options.py | `run_analysis` options on synthetic trajectories: A default / `dielectric_constant` / `A_molar` / `legacy`; checks default silent and still measured; local vs legacy molality with depleted water |
| test_full_run_e2e.py (slow) | whole pipeline incl. bootstraps: file contract, blank rows beyond the curve, reservoir density stamped on the OC/OP summary plots, thermo JSON |

Not covered: statistical accuracy of the fits (the end-to-end run uses 2 bootstraps and checks plumbing only).
