"""Pure functions: naming conventions, unit conversions, osmotic physics.

These have analytic answers, so they pin behaviour rather than reproduce it.
"""
import numpy as np
import pytest
from openmm.unit import kelvin

from conftest import T_K


# ===========================================================================
# filename / directory conventions  (contracts with the dispatch script)
# ===========================================================================
@pytest.mark.parametrize("mol,expect", [
    (3.5, ("35", "3.5")),
    (2.0, ("2", "2.0")),
    (0.5, ("05", "0.5")),
    (1.0, ("1", "1.0")),
    (6.0, ("6", "6.0")),
    (1.2, ("12", "1.2")),
])
def test_format_molality(hp_mod, mol, expect):
    """mi feeds every filename; a change here renames files silently."""
    assert hp_mod._format_molality(mol) == expect


@pytest.mark.parametrize("k,tag", [
    (1.5567, "k1p56"),
    (0.6606492479786179, "k0p661"),
    (0.68, "k0p68"),
    (0.5, "k0p5"),
    (4184.0, "k4184"),
    (2, "k2"),
])
def test_k_to_tag(hp_mod, k, tag):
    """Must match _k_to_tag in osmotic_sim_dispatch_membar_cont.py exactly.

    The analysis parses this tag out of the results directory name, so the two
    implementations are a cross-file contract. If you edit one, edit both.
    """
    assert hp_mod._k_to_tag(k) == tag


def test_k_tag_survives_full_precision(hp_mod):
    """salt_data stores k at full precision; the tag must still be stable."""
    assert hp_mod._k_to_tag(0.6606492479786179) == hp_mod._k_to_tag(0.6606)


# ===========================================================================
# molar <-> molal
# ===========================================================================
def test_molal_conversion_roundtrip(hp_mod):
    n_water, v_box_L = 3008, 90.0 * 1e-24
    c_molar = np.array([0.5, 1.0, 3.5])
    m = hp_mod.convert_profile_to_molal(c_molar, n_water, v_box_L)
    factor = (n_water * hp_mod.M_WATER_G_PER_MOL * 1e-3 / hp_mod.NA) / v_box_L
    np.testing.assert_allclose(m * factor, c_molar, rtol=1e-12)


def test_molal_close_to_molar_at_bulk_density(hp_mod):
    """Packed at ~1 g/cm3 the two scales agree to <1%; a big gap means the
    water count and the box volume disagree."""
    n_water, v_box_L = 3008, 90.0 * 1e-24
    m = hp_mod.convert_profile_to_molal(np.array([3.5]), n_water, v_box_L)[0]
    assert 3.45 < m < 3.55


def test_molal_scales_inversely_with_water(hp_mod):
    v = 90.0 * 1e-24
    m1 = hp_mod.convert_profile_to_molal(np.array([1.0]), 3000, v)[0]
    m2 = hp_mod.convert_profile_to_molal(np.array([1.0]), 6000, v)[0]
    assert m1 == pytest.approx(2 * m2, rel=1e-12)


# ===========================================================================
# osmotic physics
# ===========================================================================
def test_osmotic_coefficient_approaches_one_at_infinite_dilution(hp_mod):
    """phi -> 1 as c -> 0 for any parameter set. This is the one hard limit
    the fitted equation of state must respect."""
    hp_mod.A = 1.7964 / (1e24 / hp_mod.NA) ** 0.5      # as run_analysis sets it
    cs = np.array([1e-8, 1e-7, 1e-6])
    _, phi = hp_mod.osmotic_pressure(cs, [4.0, 0.2, 0.0], nu=2,
                                     nterms=2, T=T_K)
    assert phi[0] == pytest.approx(1.0, abs=1e-3)
    assert abs(phi[0] - 1) < abs(phi[-1] - 1)          # closer as c falls


def test_osmotic_pressure_scales_with_vant_hoff(hp_mod):
    hp_mod.A = 1.7964 / (1e24 / hp_mod.NA) ** 0.5
    cs = np.array([0.1, 0.5])
    p2, _ = hp_mod.osmotic_pressure(cs, [4.0, 0.2, 0.0], nu=2, nterms=2, T=T_K)
    p3, _ = hp_mod.osmotic_pressure(cs, [4.0, 0.2, 0.0], nu=3, nterms=2, T=T_K)
    np.testing.assert_allclose(p3, 1.5 * p2, rtol=1e-12)


def test_osm_experimental_ideal_case(hp_mod):
    """phi = 1, nu = 2, 1 mol/kg at 298.15 K.

    Pi = nu * phi * c[mol/L] * R * T, with c = m * rho_water.
    """
    res = hp_mod.osm_experimental(exp_osm_coeff=1.0, vant_hoff=2,
                                  molality=1.0, T=T_K * kelvin)
    from openmm.unit import bar
    got = res.value_in_unit(bar)
    expect = 2 * 1.0 * 0.99705 * 0.0831446261815324 * T_K
    assert got == pytest.approx(expect, rel=1e-6)


def test_osm_experimental_uses_the_temperature_it_is_given(hp_mod):
    """Regression: T used to default to 300 K while the run was at 298.15."""
    a = hp_mod.osm_experimental(1.0, 2, 1.0, T=298.15 * kelvin)
    b = hp_mod.osm_experimental(1.0, 2, 1.0, T=310.0 * kelvin)
    from openmm.unit import bar
    ratio = b.value_in_unit(bar) / a.value_in_unit(bar)
    assert ratio == pytest.approx(310.0 / 298.15, rel=1e-9)


# ===========================================================================
# small utilities
# ===========================================================================
def test_find_closest(hp_mod):
    idx, val = hp_mod.find_closest([0.1, 0.5, 1.0, 3.5], 0.9)
    assert (idx, val) == (2, 1.0)


def test_format_dict_rounds_floats(hp_mod):
    out = hp_mod.format_dict({"a": 1.23456789, "b": [2.7182818, {"c": 3.14159}]})
    assert out == {"a": 1.235, "b": [2.718, {"c": 3.142}]}
