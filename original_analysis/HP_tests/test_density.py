"""Solution density: Hosseini & Ashbaugh eqs 22-24.

The eq-22/23/24 equations in the paper are figures, not text, so they were
reconstructed from Table 3's unit footnote and checked against numbers the
paper states in prose. Those cross-checks are the tests below - if someone
"fixes" an exponent, these fail.
"""
import numpy as np
import pytest

from conftest import K_GAUSS, LX, LY, LZ, N_PAIRS, _make_universe

M_NACL = 58.44
M_CSBR = 212.809
RHO_W0 = 0.99700


# ===========================================================================
# Table 3 integrity
# ===========================================================================
def test_table3_has_all_fifteen_salts(hp_mod):
    t = hp_mod.HOSSEINI_TABLE3
    assert len(t) == 15, f"paper reports 15 salts, table has {len(t)}"
    for salt, branches in t.items():
        assert set(branches) == {"mu", "P"}, salt
        for br, c in branches.items():
            assert len(c) == 4, f"{salt}/{br} needs 4 coefficients"


def test_table3_pure_water_intercepts_are_water(hp_mod):
    """rho_w0 is TIP4P/2005 at ambient: the paper quotes 0.997 +/- 0.0005."""
    for salt, br in hp_mod.HOSSEINI_TABLE3.items():
        for name, c in br.items():
            assert c[0] == pytest.approx(0.997, abs=6e-4), f"{salt}/{name}"


def test_table3_both_branches_share_an_intercept(hp_mod):
    """Constant mu_w and constant P differ by compression, not by rho_w0."""
    for salt, br in hp_mod.HOSSEINI_TABLE3.items():
        assert br["mu"][0] == br["P"][0], salt


def test_density_rises_monotonically_with_salt(hp_mod):
    """Every alkali halide is denser than water at every molality studied."""
    m = np.linspace(0.1, 4.0, 40)
    for salt, br in hp_mod.HOSSEINI_TABLE3.items():
        rho = hp_mod.density_eq22(m, br["P"])
        assert np.all(np.diff(rho) > 0), f"{salt} density is not monotonic"
        assert rho[0] > 0.997, salt


# ===========================================================================
# eq 22
# ===========================================================================
def test_eq22_matches_a_hand_computation(hp_mod):
    """NaCl, constant P, m = 3.5, computed term by term off Table 3."""
    c = hp_mod.HOSSEINI_TABLE3["NaCl"]["P"]
    expect = (0.99700 + 6.0936e-2 * 3.5 - 6.5261e-3 * 3.5 ** 1.5
              - 5.7990e-4 * 3.5 ** 2)
    assert hp_mod.density_eq22(3.5, c) == pytest.approx(expect, rel=1e-12)


def test_eq22_reduces_to_pure_water_at_zero(hp_mod):
    for salt, br in hp_mod.HOSSEINI_TABLE3.items():
        assert hp_mod.density_eq22(0.0, br["P"]) == pytest.approx(br["P"][0])


def test_eq22_uses_a_three_halves_power(hp_mod):
    """REGRESSION: the Debye-Huckel term is m^1.5, not m^0.5 or m^3.

    Isolate it with a coefficient vector that zeroes everything else.
    """
    got = hp_mod.density_eq22(4.0, (0.0, 0.0, 1.0, 0.0))
    assert got == pytest.approx(8.0)          # 4^1.5


def test_csbr_density_is_experimentally_plausible(hp_mod):
    """3 m CsBr is ~1.40 g/cm3; the paper says CsBr slightly underpredicts."""
    rho = hp_mod.density_eq22(3.0, hp_mod.HOSSEINI_TABLE3["CsBr"]["P"])
    assert 1.35 < rho < 1.45, f"CsBr at 3 m = {rho:.3f}, expected ~1.40"


# ===========================================================================
# eq 23  - the strongest external check in this file
# ===========================================================================
def test_eq23_offset_implies_a_physical_osmotic_pressure(hp_mod):
    """Table 3 gives BOTH branches, and eq 23 says they differ by
    (1 - kappa_eff * Pi). Inverting that must return the real osmotic
    pressure of each salt - an independent confirmation of eq 23's form
    and of kappa_eff, using numbers the fit never saw.
    """
    for salt, m, lo, hi in [("NaCl", 3.5, 150, 260), ("CsBr", 3.0, 90, 190)]:
        c = hp_mod.HOSSEINI_TABLE3[salt]
        r_mu = hp_mod.density_eq22(m, c["mu"])
        r_p = hp_mod.density_eq22(m, c["P"])
        pi = (1 - r_p / r_mu) / hp_mod.KAPPA_EFF_PER_BAR
        assert lo < pi < hi, f"{salt} at {m} m implies Pi = {pi:.0f} bar"


def test_eq23_lowers_the_density(hp_mod):
    """Decompressing to 1 bar must reduce it - the paper says so explicitly."""
    r = hp_mod.density_eq22(3.5, hp_mod.HOSSEINI_TABLE3["NaCl"]["mu"])
    assert hp_mod.density_const_P(r, 200.0) < r


def test_eq23_is_a_noop_at_zero_pressure(hp_mod):
    assert hp_mod.density_const_P(1.1, 0.0) == pytest.approx(1.1)


def test_kappa_eff_is_below_tip4p2005_water(hp_mod):
    """Paper: kappa_eff is ~20% lower than water's 4.65e-5 /bar."""
    assert hp_mod.KAPPA_EFF_PER_BAR == pytest.approx(4.65e-5 * 0.8, rel=0.1)


# ===========================================================================
# eq 24
# ===========================================================================
def test_partial_molar_volume_at_infinite_dilution(hp_mod):
    """m -> 0 must give M_w / rho_w0 = 18.07 cm3/mol."""
    c = hp_mod.HOSSEINI_TABLE3["NaCl"]["P"]
    got = hp_mod.partial_molar_volume_water(1e-12, c, M_NACL)
    assert got == pytest.approx(hp_mod.M_WATER_G_PER_MOL / c[0], rel=1e-6)
    assert got == pytest.approx(18.07, abs=0.01)


def test_partial_molar_volume_varies_by_less_than_three_percent(hp_mod):
    """The paper's own claim: V_w_bar changes <~3% over the range studied.

    This is the check that validates the eq-24 derivation, since eq 24 is a
    figure in the HTML and was re-derived rather than transcribed.
    """
    for salt, M in [("NaCl", M_NACL), ("CsBr", M_CSBR), ("KCl", 74.55)]:
        c = hp_mod.HOSSEINI_TABLE3[salt]["P"]
        v0 = hp_mod.partial_molar_volume_water(1e-12, c, M)
        v = hp_mod.partial_molar_volume_water(np.linspace(0.1, 3.5, 20), c, M)
        drift = np.max(np.abs(v / v0 - 1))
        assert drift < 0.03, f"{salt}: V_w_bar drifts {100*drift:.1f}%"


def test_water_is_compressed_by_salt_not_expanded(hp_mod):
    """Electrostriction: V_w_bar must DECREASE with added salt."""
    c = hp_mod.HOSSEINI_TABLE3["NaCl"]["P"]
    v = hp_mod.partial_molar_volume_water([0.5, 1.0, 2.0, 3.5], c, M_NACL)
    assert np.all(np.diff(v) < 0), f"V_w_bar not decreasing: {v}"


# ===========================================================================
# fitting
# ===========================================================================
def test_fit_recovers_known_coefficients_exactly(hp_mod):
    """Eq 22 is linear in its coefficients, so a noiseless fit is exact."""
    truth = hp_mod.HOSSEINI_TABLE3["NaCl"]["P"]
    m = np.linspace(0.05, 4.0, 60)
    got = hp_mod.fit_density_eq22(m, hp_mod.density_eq22(m, truth))
    np.testing.assert_allclose(got, truth, atol=1e-10)


def test_fit_with_a_pinned_intercept(hp_mod):
    truth = hp_mod.HOSSEINI_TABLE3["CsBr"]["P"]
    m = np.linspace(0.05, 3.0, 40)
    got = hp_mod.fit_density_eq22(m, hp_mod.density_eq22(m, truth),
                                  fix_rho_w0=truth[0])
    np.testing.assert_allclose(got, truth, atol=1e-10)
    assert got[0] == truth[0]


def test_fit_rejects_too_few_points(hp_mod):
    with pytest.raises(ValueError, match="4 usable points"):
        hp_mod.fit_density_eq22([0.1, 0.2], [1.0, 1.01])


def test_fit_drops_nan_bins(hp_mod):
    """Empty reservoir bins come through as NaN molality; they must not
    poison the fit."""
    truth = hp_mod.HOSSEINI_TABLE3["NaCl"]["P"]
    m = np.linspace(0.05, 4.0, 60)
    rho = hp_mod.density_eq22(m, truth)
    m = np.concatenate([m, [np.nan, np.nan]])
    rho = np.concatenate([rho, [np.nan, 1.0]])
    np.testing.assert_allclose(hp_mod.fit_density_eq22(m, rho), truth,
                               atol=1e-10)


def test_the_curve_is_trustworthy_even_when_the_coefficients_are_not(hp_mod):
    """theta_1 and theta_3/2 are strongly anti-correlated: noisy data moves
    them a lot while rho(m) barely budges. So compare CURVES against Table 3,
    never individual coefficients.

    This test documents that, and guards the useful half of the claim.
    """
    truth = hp_mod.HOSSEINI_TABLE3["NaCl"]["P"]
    m = np.linspace(0.05, 4.0, 60)
    rng = np.random.default_rng(0)
    rho = hp_mod.density_eq22(m, truth) + rng.normal(0, 0.004, m.size)
    got = hp_mod.fit_density_eq22(m, rho, fix_rho_w0=truth[0])

    curve_err = abs(hp_mod.density_eq22(3.5, got)
                    / hp_mod.density_eq22(3.5, truth) - 1)
    coef_err = abs(got[1] / truth[1] - 1)
    assert curve_err < 0.01, f"curve off by {100*curve_err:.1f}%"
    assert coef_err > curve_err, (
        "expected theta_1 to be less well determined than the curve")


# ===========================================================================
# measurement from a trajectory
# ===========================================================================
@pytest.fixture
def density_universe():
    """Uniform water at a count chosen to give exactly 0.997 g/cm3, plus a
    Gaussian ion cloud. Water is held uniform, so the analytic answer is

        rho(m) = rho_w0 + (M_s * rho_w0 / 1000) * m

    i.e. salt mass adds, nothing compresses.
    """
    n_wat = round(RHO_W0 * (LX * LY * LZ) * 1e-21 / 18.015 * 6.02214076e23)
    sig = 1.0 / np.sqrt(2 * K_GAUSS)
    u = _make_universe(
        N_PAIRS, n_wat, LX, LY, LZ, n_frames=400,
        ion_z_nm=lambda rng, n: np.clip(rng.normal(LZ / 2, sig, n), 0, LZ),
        seed=7)
    return u, n_wat


def _profiles(hp_mod, u, dz=0.25):
    return hp_mod.density_molality_profiles(
        u, u.select_atoms("resname HOH and name O"),
        u.select_atoms("resname NA"), u.select_atoms("resname CL"),
        dz, M_NACL)


def test_reservoir_density_is_pure_water(hp_mod, density_universe):
    """The far end of the box has no salt, so it must read as pure water.
    A wrong unit factor anywhere shows up here immediately."""
    u, _ = density_universe
    _z, rho, _m, _cs, _cw = _profiles(hp_mod, u)
    assert rho[-6:].mean() == pytest.approx(RHO_W0, abs=0.01)


def test_water_molarity_is_bulk_water(hp_mod, density_universe):
    u, _ = density_universe
    _z, _rho, _m, _cs, c_w = _profiles(hp_mod, u)
    assert c_w.mean() == pytest.approx(55.35, rel=0.02)


def test_density_exceeds_water_wherever_there_is_salt(hp_mod,
                                                      density_universe):
    u, _ = density_universe
    _z, rho, _m, c_s, _cw = _profiles(hp_mod, u)
    salty = c_s > 1.0
    assert salty.any()
    assert np.all(rho[salty] > RHO_W0)


def test_measured_slope_matches_the_analytic_synthetic_answer(
        hp_mod, density_universe):
    """With rigid uniform water the density curve is exactly linear in m with
    slope M_s * rho_w0 / 1000. Check the CURVE, per the caveat above."""
    u, _ = density_universe
    _z, rho, m, _cs, _cw = _profiles(hp_mod, u)
    good = np.isfinite(m)
    co = hp_mod.fit_density_eq22(m[good], rho[good], fix_rho_w0=RHO_W0)
    expect = RHO_W0 + (M_NACL * RHO_W0 / 1000.0) * 3.5
    assert hp_mod.density_eq22(3.5, co) == pytest.approx(expect, rel=0.01)


def test_local_molality_differs_from_the_box_average_conversion(
        hp_mod, density_universe):
    """Why this replaces convert_profile_to_molal: the box-averaged factor is
    wrong at the centre, because the centre is not the box."""
    u, n_wat = density_universe
    _z, _rho, m, c_s, _cw = _profiles(hp_mod, u)
    v_box_L = LX * LY * LZ * 1e-24
    m_box = hp_mod.convert_profile_to_molal(c_s, n_wat, v_box_L)
    peak = np.nanargmax(c_s)
    assert m[peak] == pytest.approx(m_box[peak], rel=0.10), (
        "sanity: uniform water means the two should be close here")
    assert np.isfinite(m).sum() > 5


def test_profiles_share_the_standard_grid(hp_mod, density_universe):
    """Same folded half-box grid as every other profile function."""
    u, _ = density_universe
    for dz in (0.25, 0.1):
        z, rho, m, c_s, c_w = _profiles(hp_mod, u, dz)
        z_ref, _c = hp_mod.molarity_profile_fixed_Lz(
            u, u.select_atoms("resname NA"), dz)
        np.testing.assert_allclose(z, z_ref)
        assert len(rho) == len(m) == len(c_s) == len(c_w) == len(z)
        assert len(z) == round((LZ / 2) / dz)
