"""Units and the modified Debye-Hueckel building blocks. Pure functions, no files."""
import numpy as np
import pytest

NA = 6.02214076e23


def test_A_scales_as_dielectric_constant_to_minus_three_halves(hp_mod):
    a1 = hp_mod.debye_huckel_A_molar(59.1, 298.15)
    a2 = hp_mod.debye_huckel_A_molar(78.4, 298.15)
    assert a2 / a1 == pytest.approx((59.1 / 78.4) ** 1.5, rel=1e-12)


def test_A_scales_with_temperature(hp_mod):
    a1 = hp_mod.debye_huckel_A_molar(78.4, 298.15)
    a2 = hp_mod.debye_huckel_A_molar(78.4, 323.15)
    assert a2 / a1 == pytest.approx((298.15 / 323.15) ** 1.5, rel=1e-12)


def test_A_for_real_water_is_the_textbook_limiting_slope(hp_mod):
    """ln(gamma+-) = -1.1744 sqrt(I) for a 1:1 salt in water at 25 C (Pitzer A_phi = 0.3915, x3)."""
    assert hp_mod.debye_huckel_A_molar(78.38, 298.15) == pytest.approx(1.1744, rel=5e-3)


def test_A_in_molar_units_is_the_nm_cubed_value_divided_by_the_unit_factor(hp_mod):
    """REGRESSION for the `A /= sqrt(1e24/NA)` bug. Working in nm: A_nm = (1/2) l_B^(3/2) sqrt(8 pi), c in nm^-3.
    c[M] = c[nm^-3] * 1e24/NA, so A[M^-1/2] = A_nm / sqrt(1e24/NA). The old code applied that division to an A
    that was already in M^-1/2."""
    eps, T = 59.1, 298.15
    e, eps0, kB = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23
    lB_nm = e ** 2 / (4 * np.pi * eps0 * eps * kB * T) * 1e9
    A_nm = 0.5 * lB_nm ** 1.5 * np.sqrt(8 * np.pi)
    assert hp_mod.debye_huckel_A_molar(eps, T) == pytest.approx(A_nm / np.sqrt(1e24 / NA), rel=1e-9)


def test_g_series_matches_high_precision_direct_evaluation(hp_mod):
    x = np.array([1e-3, 1e-2, 0.04, 0.049, 0.051, 0.5, 3.0])
    xl = x.astype(np.longdouble)
    direct = ((2 * xl + xl ** 2) / (1 + xl) - 2 * np.log1p(xl)).astype(float)
    assert np.allclose(hp_mod._g(x), direct, rtol=1e-8)


def test_pi_bar_agrees_with_the_original_osmotic_pressure_function(hp_mod, monkeypatch):
    """Two independent implementations of the same modDH osmotic pressure."""
    A, B, a1, a2, T = 1.7964, 1.2, 0.2, 0.01, 298.15
    monkeypatch.setattr(hp_mod, "A", A)
    cs = np.linspace(0.05, 4.0, 30)
    op_old, _phi = hp_mod.osmotic_pressure(cs, [B, a1, a2], nu=2, nterms=2, T=T)
    assert np.allclose(hp_mod.pi_bar(cs, A, B, a1, a2, 2, T), op_old, rtol=1e-10)


def test_pi_bar_is_ideal_at_low_concentration(hp_mod):
    c = np.array([1e-4])
    ideal = 0.01 * 2 * hp_mod.R * 298.15 * c
    assert hp_mod.pi_bar(c, 1.8, 1.0, 0.1, 0.0, 2, 298.15) == pytest.approx(ideal, rel=0.05)


@pytest.mark.parametrize("A,B,a1,a2", [(1.79, 1.9, 0.09, 0.02), (1.2, 0.5, 0.4, -0.01), (1.8, 1e-3, 0.3, 0.0)])
def test_osmotic_bracket_and_ln_gamma_obey_gibbs_duhem(hp_mod, A, B, a1, a2):
    """For a 1:1 salt:  m dln(gamma+-)/dm = m dphi/dm + (phi - 1).  Any (A,B,a1,a2) must satisfy it if the two
    expressions are the same thermodynamic model."""
    m = np.linspace(0.05, 4.0, 8001)
    bracket = m + hp_mod.dh_osmotic(m, A, B) + 0.5 * a1 * m ** 2 + (2.0 / 3.0) * a2 * m ** 3     # = m*phi
    phi = bracket / m
    lng = hp_mod.ln_gamma(m, a1, a2, A, B)
    lhs = m * np.gradient(lng, m)
    rhs = m * np.gradient(phi, m) + (phi - 1)
    assert np.max(np.abs(lhs - rhs)[10:-10]) < 5e-5
