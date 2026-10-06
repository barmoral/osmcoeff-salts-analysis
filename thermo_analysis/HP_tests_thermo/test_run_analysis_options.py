"""run_analysis() options on synthetic trajectories (profiles_only keeps these fast)."""
import numpy as np
import pytest

from conftest import write_case, make_universe, N_PAIRS, N_WATER, LZ

NA = 6.02214076e23


@pytest.fixture
def case(tmp_path):
    return write_case(tmp_path, n_rep=2)


@pytest.fixture
def sd(salt_data_module):
    return salt_data_module.__file__


def _run(hp, d, sd, **kw):
    kw.setdefault("profiles_only", True)
    kw.setdefault("n_expansion_terms", 2)
    return hp.run_analysis(salt_data_path=sd, results_dir=str(d), ion1="Na", ion2="Cl", molality=3.5,
                           N_replicates=2, **kw)


# ----------------------------------------------------------------------------------- A
def test_default_A_comes_from_the_dielectric_constant_with_no_unit_conversion(hp_mod, case, sd):
    out = _run(hp_mod, case, sd)
    assert hp_mod.A == pytest.approx(hp_mod.debye_huckel_A_molar(59.1, 298.15), rel=1e-12)


def test_dielectric_constant_and_A_molar_options(hp_mod, case, sd):
    _run(hp_mod, case, sd, dielectric_constant=94.0)
    assert hp_mod.A == pytest.approx(hp_mod.debye_huckel_A_molar(94.0, 298.15), rel=1e-12)
    _run(hp_mod, case, sd, A_molar=1.234)
    assert hp_mod.A == 1.234


def test_legacy_restores_the_old_unit_conversion(hp_mod, case, sd):
    _run(hp_mod, case, sd, legacy=True)
    assert hp_mod.A == pytest.approx(hp_mod.debye_huckel_A_molar(59.1, 298.15) / np.sqrt(1e24 / NA), rel=1e-12)


# ------------------------------------------------------------------------------- checks
def test_checks_default_to_ignore_and_stay_quiet_even_for_a_wrong_box(hp_mod, tmp_path, sd, capsys):
    d = write_case(tmp_path, n_rep=2, lx=2.7, ly=2.7)             # area 7.29 vs BOX 9.0
    out = _run(hp_mod, d, sd)
    text = capsys.readouterr().out
    assert "Lx*Ly" not in text and "OFF" not in text
    assert out["reservoir_density"].shape == (2,)                 # still measured and returned


def test_checks_warn_reports_a_wrong_box_and_a_wrong_reservoir(hp_mod, tmp_path, sd, capsys):
    d = write_case(tmp_path, n_rep=2, lx=2.7, ly=2.7)
    _run(hp_mod, d, sd, checks="warn", reservoir_density_target=0.80)
    text = capsys.readouterr().out
    assert "Lx*Ly" in text and "OFF" in text


def test_checks_strict_raises_for_a_wrong_box(hp_mod, tmp_path, sd):
    d = write_case(tmp_path, n_rep=2, lx=2.7, ly=2.7)
    with pytest.raises(ValueError, match="Lx\\*Ly|reservoir"):
        _run(hp_mod, d, sd, checks="strict")


def test_checks_strict_with_a_correct_box_and_no_target_passes(hp_mod, case, sd):
    assert _run(hp_mod, case, sd, checks="strict") is not None


def test_bad_checks_value_raises_before_any_work(hp_mod, case, sd):
    with pytest.raises(ValueError, match="checks must be one of"):
        _run(hp_mod, case, sd, checks="maybe")


def test_reservoir_density_is_the_water_density_of_the_box_ends(hp_mod, case, sd):
    out = _run(hp_mod, case, sd)
    expected = N_WATER * hp_mod.M_WATER_G_PER_MOL / (NA * 90.0e-21)       # g/cm3: uniform water in 90 nm^3 (1 nm^3 = 1e-21 cm^3)
    assert np.allclose(out["reservoir_density"], expected, rtol=0.02)


# ------------------------------------------------------------------------------ molality
def _depleted_water(rng, n):
    """Water thinned out at the box centre (where the ions sit): density ~ 1 - 0.4 exp(-(z-z0)^2/2)."""
    z0, out = LZ / 2, []
    while len(out) < n:
        z = rng.uniform(0, LZ, 4 * n)
        keep = rng.uniform(0, 1, 4 * n) < (1 - 0.4 * np.exp(-((z - z0) ** 2) / 2.0))
        out.extend(z[keep])
    return np.array(out[:n])


def test_local_molality_differs_from_legacy_where_the_water_is_depleted(hp_mod, tmp_path, sd):
    d = write_case(tmp_path, n_rep=2, water_z_nm=_depleted_water)
    new = _run(hp_mod, d, sd)["mean_molal"]
    old = _run(hp_mod, d, sd, legacy=True)["mean_molal"]
    ratio = new / np.where(old > 0.05, old, np.nan)
    assert ratio[0] > 1.15                                         # centre: less water per litre -> higher m
    last = np.where(old > 0.05)[0][-1]                             # outermost bin that still holds ions
    assert ratio[last] < ratio[0] - 0.1                            # the correction fades towards the reservoir


def test_local_and_legacy_molality_agree_for_uniform_water(hp_mod, case, sd):
    new = _run(hp_mod, case, sd)["mean_molal"]
    old = _run(hp_mod, case, sd, legacy=True)["mean_molal"]
    m = old > 0.05
    assert np.allclose(new[m], old[m], rtol=0.03)
