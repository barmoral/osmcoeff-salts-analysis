"""Profile machinery: folding, binning, conservation, normalisation.

Every test here has an exact answer computable by hand, so a failure is a
behaviour change, not statistical noise.
"""
import numpy as np
import pytest

from conftest import K_GAUSS, LX, LY, LZ, N_PAIRS

MOL = 6.02214076e23 * 1e-24          # mol/L -> nm^-3


# ===========================================================================
# folding
# ===========================================================================
def test_fold_centre_comes_from_the_trajectory(hp_mod, offcentre_universe):
    """REGRESSION: distance used to default to 72 A, which folds a 100 A box
    about the wrong point and silently produces a non-monotonic profile.

    A Gaussian centred at Lz/2 must fold to a distribution peaked at 0.
    """
    u = offcentre_universe
    ions = u.select_atoms("resname NA CL")
    z = hp_mod.get_ion_array(u, ions)

    assert z.min() >= 0.0
    assert z.max() <= u.trajectory.ts.dimensions[2] / 10.0 / 2 + 1e-9

    h, edges = np.histogram(z, bins=12, range=(0, 7.2))
    assert h.argmax() <= 1, "folded profile does not peak at z0"
    assert h[0] > h[-1] * 5, "folded profile is not decaying"


def test_fold_centre_can_still_be_overridden(hp_mod, gaussian_universe):
    u = gaussian_universe
    ions = u.select_atoms("resname NA CL")
    auto = hp_mod.get_ion_array(u, ions)
    same = hp_mod.get_ion_array(u, ions, distance=LZ * 10 / 2)
    np.testing.assert_allclose(auto, same)


def test_fold_returns_one_value_per_ion_per_frame(hp_mod, gaussian_universe):
    u = gaussian_universe
    ions = u.select_atoms("resname NA CL")
    z = hp_mod.get_ion_array(u, ions)
    assert z.size == ions.n_atoms * len(u.trajectory)


# ===========================================================================
# binning
# ===========================================================================
@pytest.mark.parametrize("dz", [0.1, 0.05, 0.025])
def test_dz_nm_actually_controls_the_bins(hp_mod, uniform_universe, dz):
    """REGRESSION: dz_nm was accepted and ignored; the bin count was pinned to
    int(Lz*10) so asking for finer bins changed nothing."""
    u = uniform_universe
    ag = u.select_atoms("resname NA")
    centres, _w, _c = hp_mod.count_profile_z_fixed_Lz(u, ag, dz)
    assert len(centres) == round((LZ / 2) / dz)


def test_count_and_molarity_profiles_share_a_grid(hp_mod, uniform_universe):
    """REGRESSION: the two used different formulas, so c_0 and the bootstrap
    profiles had different lengths -> 'yerr (144,) vs y (100,)'."""
    u = uniform_universe
    ag = u.select_atoms("resname NA")
    for dz in (0.05, 0.025):
        z_cnt, _w, _c = hp_mod.count_profile_z_fixed_Lz(u, ag, dz)
        z_mol, _cm = hp_mod.molarity_profile_fixed_Lz(u, ag, dz)
        assert len(z_cnt) == len(z_mol)
        np.testing.assert_allclose(z_cnt, z_mol)


def test_profiles_require_dz(hp_mod, uniform_universe):
    """No default: a forgotten argument must fail loudly, not silently use 0.05."""
    u = uniform_universe
    ag = u.select_atoms("resname NA")
    with pytest.raises(TypeError):
        hp_mod.count_profile_z_fixed_Lz(u, ag)


# ===========================================================================
# conservation — the strongest structural check
# ===========================================================================
def test_count_profile_conserves_ions(hp_mod, gaussian_universe):
    """Summing the folded per-species profile must return every ion."""
    u = gaussian_universe
    for resname in ("NA", "CL"):
        ag = u.select_atoms(f"resname {resname}")
        _z, _w, counts = hp_mod.count_profile_z_fixed_Lz(u, ag, 0.05)
        assert counts.sum() == pytest.approx(N_PAIRS, abs=1e-9)


def test_molarity_profile_integrates_to_the_right_number(hp_mod,
                                                         uniform_universe):
    """Uniform ions: integral of c(z) over the folded half-box, times the two
    slabs each bin represents, must give N_pairs.

        sum_i c_i * (2 * A * w_i) = N_pairs
    """
    u = uniform_universe
    A = LX * LY
    tot = 0.0
    for resname in ("NA", "CL"):
        ag = u.select_atoms(f"resname {resname}")
        _z, c = hp_mod.molarity_profile_fixed_Lz(u, ag, 0.05)
        w = (LZ / 2) / len(c)
        tot += (c * MOL).sum() * 2 * A * w
    assert tot / 2 == pytest.approx(N_PAIRS, rel=0.02)


def test_uniform_ions_give_a_flat_profile(hp_mod, uniform_universe):
    """Uniform in z -> flat after folding, at N/(A*Lz)."""
    u = uniform_universe
    ag = u.select_atoms("resname NA")
    _z, c = hp_mod.molarity_profile_fixed_Lz(u, ag, 0.25)
    expect_M = (N_PAIRS / (LX * LY * LZ)) / MOL
    assert np.median(c) == pytest.approx(expect_M, rel=0.15)
    assert c.std() / c.mean() < 0.35          # flat within Poisson noise


# ===========================================================================
# cross-section and shape
# ===========================================================================
def test_cross_section_area(hp_mod, uniform_universe):
    assert hp_mod.get_average_cross_section_area(uniform_universe) == \
        pytest.approx(LX * LY, rel=1e-9)


def test_gaussian_profile_has_the_width_k_implies(hp_mod, gaussian_universe):
    """The restraint sets sigma = 1/sqrt(2K); recover it from the profile."""
    u = gaussian_universe
    ions = u.select_atoms("resname NA CL")
    z = hp_mod.get_ion_array(u, ions)
    sigma_expect = 1.0 / np.sqrt(2 * K_GAUSS)
    # folded half-normal: E[z] = sigma * sqrt(2/pi)
    sigma_got = z.mean() / np.sqrt(2 / np.pi)
    assert sigma_got == pytest.approx(sigma_expect, rel=0.10)


def test_fold_profile_is_identity(hp_mod):
    z = np.linspace(0, 5, 11)
    y = np.exp(-z)
    zo, yo = hp_mod.fold_profile(z, y)
    np.testing.assert_allclose(zo, z)
    np.testing.assert_allclose(yo, y)
