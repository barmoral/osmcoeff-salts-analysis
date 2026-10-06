"""The whole pipeline (LS fits, ML bootstrap, results table, thermo extras, plots) on a synthetic system.
Few bootstraps - this checks plumbing and file contracts, not statistics."""
import csv
import json

import matplotlib.axes
import numpy as np
import pytest

from conftest import write_case

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, salt_data_module):
    tmp = tmp_path_factory.mktemp("full")
    case = write_case(tmp, n_rep=2, n_frames=8)
    out = tmp / "out"
    texts = []
    orig = matplotlib.axes.Axes.text
    matplotlib.axes.Axes.text = lambda self, *a, **k: (texts.append(str(a[2] if len(a) > 2 else k.get("s"))),
                                                       orig(self, *a, **k))[1]
    try:
        import HP_analysis_replicates_thermo as hp
        ret = hp.run_analysis(
            salt_data_path=salt_data_module.__file__, results_dir=str(case), out_dir=str(out),
            ion1="Na", ion2="Cl", molality=3.5, N_replicates=2, n_expansion_terms=2, dz_nm=0.1,
            n_bootstraps=2, random_seed=1,
            reference=dict(label="a literature curve", molar=(1.79, 1.2, 0.2, 0.01)))
    finally:
        matplotlib.axes.Axes.text = orig
    return dict(out=out, texts=texts, ret=ret)


def test_expected_files_exist(full_run):
    names = {p.name for p in full_run["out"].iterdir()}
    for f in ("NaCl_final_results_35m.csv", "NaCl_final_results_35m.json", "NaCl_oc.png", "NaCl_op.png",
              "thermo_density_NaCl.png", "thermo_ln_gamma_NaCl.png", "thermo_mu_ion_excess_NaCl.png",
              "thermo_params_NaCl_35m.json", "thermo_results_NaCl_35m.csv", "thermo_curves_NaCl_35m.npz"):
        assert f in names, f


def test_reservoir_density_is_stamped_on_the_main_result_plots(full_run):
    stamped = [t for t in full_run["texts"] if "reservoir" in t]
    assert len(stamped) >= 2 + 8                                   # OC and OP summary + the thermo plots
    ret = full_run["ret"]
    assert ret["reservoir_density"].shape == (2,)


def test_results_table_has_blank_rows_beyond_the_curve_not_a_repeated_plateau(full_run):
    rows = list(csv.DictReader(open(full_run["out"] / "NaCl_final_results_35m.csv")))
    real = [r for r in rows if r["Osmotic Pressure"] != ""]
    blank = [r for r in rows if r["Osmotic Pressure"] == ""]
    assert real and blank
    vals = [float(r["Osmotic Pressure"]) for r in real]
    assert len(set(np.round(vals, 6))) == len(vals)                # no repeated plateau value
    assert all(float(r["Molality"]) > float(real[-1]["Molality"]) for r in blank)
    js = json.load(open(full_run["out"] / "NaCl_final_results_35m.json"))
    assert js["osmotic_pressure"][-1] is None


def test_thermo_json_records_the_reservoir_density_and_the_reference(full_run):
    js = json.load(open(full_run["out"] / "thermo_params_NaCl_35m.json"))
    assert len(js["reservoir_density_g_cm3"]) == 2
    assert js["reference"]["label"] == "a literature curve"
    assert set(js["methods"]) == {"unweighted", "weighted", "maximum_likelihood"}


def test_molar_and_molal_osmotic_pressures_agree(full_run):
    js = json.load(open(full_run["out"] / "thermo_params_NaCl_35m.json"))
    for name, m in js["methods"].items():
        assert m["rms_molalOP_minus_molarOP_bar"] < 3.0, name


def test_returned_A_is_the_molar_value(full_run, hp_mod):
    assert full_run["ret"]["A"] == pytest.approx(hp_mod.debye_huckel_A_molar(59.1, 298.15), rel=1e-12)
