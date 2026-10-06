"""salt_data_*.py integrity, and the design equations behind it.

These run build_salt_data.py for real and then check the file it produced, so
generator and consumer are tested together.
"""
import math

import numpy as np
import pytest

from conftest import K_GAUSS, LX, LY, LZ, T_K

R_KJ = 8.31446261815324e-3
MOL = 6.02214076e23 * 1e-24


# ===========================================================================
# schema
# ===========================================================================
def test_box_has_every_field_the_pipeline_reads(salt_data_module):
    box = salt_data_module.BOX
    for key in ("lx", "ly", "lz", "area", "z_center", "n_water",
                "mass_water_kg", "volume_L", "temperature",
                "k_HP", "delta_z_FBP", "k_FBP_wall", "molal_factor"):
        assert key in box, f"BOX['{key}'] missing"


def test_small_quantities_survived_serialisation(salt_data_module):
    """REGRESSION: round(v, 8) turned mass_water_kg (~1e-22) and volume_L into
    0.0 in the written file. Nothing crashed; downstream maths just broke."""
    box = salt_data_module.BOX
    assert box["mass_water_kg"] > 0, "mass_water_kg collapsed to zero"
    assert box["volume_L"] > 0, "volume_L collapsed to zero"
    assert box["mass_water_kg"] == pytest.approx(
        box["n_water"] * 18.01528 / 6.02214076e23 * 1e-3, rel=1e-9)


def test_lookup_returns_a_full_recipe(salt_data_module):
    row = salt_data_module.lookup("NaCl", 3.5)
    for field in ("salt", "molality", "osmotic_coefficient", "density",
                  "num_particles", "k_HP", "delta_z_FBP", "k_FBP_wall"):
        assert hasattr(row, field)
    assert row.num_particles > 0
    assert row.k_HP > 0


def test_lookup_raises_on_a_missing_row(salt_data_module):
    with pytest.raises(KeyError):
        salt_data_module.lookup("NaCl", 99.0)


def test_row_values_match_box_values(salt_data_module):
    """Rows carry copies of k and delta_z; they must not drift from BOX."""
    box = salt_data_module.BOX
    for salt, m in (("NaCl", 3.5), ("CsBr", 3.0)):
        try:
            row = salt_data_module.lookup(salt, m)
        except KeyError:
            continue
        assert row.k_HP == pytest.approx(box["k_HP"], rel=1e-12)
        assert row.delta_z_FBP == pytest.approx(box["delta_z_FBP"], rel=1e-12)


# ===========================================================================
# derived experimental quantities
# ===========================================================================
def test_molarity_is_derived_not_copied_molality(salt_data_module):
    """REGRESSION: the legacy file stored molarity ~= molality, which is only
    true at infinite dilution and is 10-23% wrong at high concentration."""
    row = salt_data_module.lookup("NaCl", 6.0)
    assert row.molarity < 0.95 * row.molality, (
        "molarity looks like a copy of molality - the solute volume is being "
        "ignored")


def test_molarity_formula(salt_data_module):
    row = salt_data_module.lookup("NaCl", 3.5)
    mm = salt_data_module.MOLAR_MASS["NaCl"]
    expect = 1000 * row.molality * row.density / (1000 + row.molality * mm)
    assert row.molarity == pytest.approx(expect, rel=1e-12)


def test_vant_hoff_factors(salt_data_module):
    vh = salt_data_module.VANT_HOFF
    assert vh["NaCl"] == 2 and vh["CsBr"] == 2
    assert vh["MgCl"] == 3 and vh["Na2SO4"] == 3


def test_osmotic_coefficients_are_physical(salt_data_module):
    for d in salt_data_module.salt_infos:
        assert 0.3 < d["osmotic_coefficient"] < 3.0, d
        assert 0.9 < d["density"] < 2.5, d


# ===========================================================================
# design equations  (these are the physics, not bookkeeping)
# ===========================================================================
def test_k_matches_the_edge_tolerance_criterion(salt_data_module):
    """k_HP = 2RT * (-ln(edge_tol)) / (Lz/2)^2, from the box alone."""
    box = salt_data_module.BOX
    K = -math.log(box["edge_tol"]) / (box["lz"] / 2) ** 2
    assert box["k_HP"] == pytest.approx(2 * R_KJ * box["temperature"] * K,
                                        rel=1e-9)


def test_delta_z_equals_the_gaussian_effective_half_width(salt_data_module):
    """Why one pdb serves HP and FBP: delta_z = int_0^inf exp(-K z^2) dz."""
    box = salt_data_module.BOX
    K = box["k_HP"] / (2 * R_KJ * box["temperature"])
    assert box["delta_z_FBP"] == pytest.approx(0.5 * math.sqrt(math.pi / K),
                                               rel=1e-9)


def test_particle_count_hits_the_target_molality(salt_data_module):
    """Plug N and k back into the Gaussian normalisation; recover the target."""
    box = salt_data_module.BOX
    for m in (1.0, 2.0, 3.5):
        try:
            row = salt_data_module.lookup("NaCl", m)
        except KeyError:
            continue
        K = row.k_HP / (2 * R_KJ * box["temperature"])
        erf_c = math.erf(math.sqrt(K) * box["lz"] / 2)
        c0_nm3 = (row.num_particles / box["area"]) * math.sqrt(K / math.pi) / erf_c
        c0_molal = (c0_nm3 / MOL) / box["molal_factor"]
        assert c0_molal == pytest.approx(m, rel=0.02), \
            f"N={row.num_particles} gives {c0_molal:.3f} mol/kg, wanted {m}"


def test_particle_count_is_linear_in_molality(salt_data_module):
    a = salt_data_module.lookup("NaCl", 1.0).num_particles
    b = salt_data_module.lookup("NaCl", 2.0).num_particles
    assert b == pytest.approx(2 * a, abs=2)


def test_particle_count_is_salt_independent(salt_data_module):
    """N depends on box + target only. Two salts at the same molality must
    need the same number of pairs."""
    try:
        a = salt_data_module.lookup("NaCl", 3.5).num_particles
        b = salt_data_module.lookup("CsBr", 3.5).num_particles
    except KeyError:
        pytest.skip("both salts not present at 3.5 m")
    assert a == b


def test_paper_box_reproduces_hosseini(salt_data_module):
    """3 x 3 x 10 nm at edge_tol 3.9e-4 must give the paper's k and N."""
    box = salt_data_module.BOX
    if not (abs(box["lz"] - 10.0) < 1e-9 and abs(box["area"] - 9.0) < 1e-9):
        pytest.skip("fixture is not the paper box")
    assert box["k_HP"] == pytest.approx(1.5567, abs=5e-4)
    assert salt_data_module.lookup("NaCl", 3.5).num_particles == 60
