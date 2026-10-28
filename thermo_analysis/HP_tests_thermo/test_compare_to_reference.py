"""compare_to_reference.py: the reconstruction maths, on a made-up reference (no literature numbers)."""
import numpy as np
import pytest
from scipy.integrate import simpson

import compare_to_reference as cr

R, NA = 8.31446261815324, 6.02214076e23
FAKE = dict(label="made up", molar=(1.8, 1.5, 0.1, 0.02), molal=(1.79, 1.49, 0.09, 0.015),
            rho_mu=(0.997, 0.05, -0.006, -0.0001), rho_P=(0.997, 0.048, -0.0055, -0.0006))
MS, NU, T = 58.443, 2, 298.15


def test_ideal_reference_reproduces_the_analytic_gaussian(hp_mod):
    """With ln gamma = 0 the restrained profile is C0 exp(-K z^2); N = area*(NA 1e-24)*C0*sqrt(pi/K)."""
    k, area, lz, n = 1.5567, 9.0, 10.0, 60
    z = np.linspace(0, lz / 2, 401)
    c, c0 = cr.reconstruct_profile(z, k, T, n, area, lz, (0.0, 1.0, 0.0, 0.0))
    K = k * 1000.0 / (2 * R * T)
    assert c0 == pytest.approx(n / (area * NA * 1e-24 * np.sqrt(np.pi / K)), rel=2e-3)
    assert np.allclose(c, c0 * np.exp(-K * z ** 2), rtol=1e-6)


def test_reconstructed_profile_integrates_to_the_ion_count(hp_mod):
    k, area, lz, n = 2.5, 9.0, 10.0, 60
    z = np.linspace(0, lz / 2, 2001)
    c, _c0 = cr.reconstruct_profile(z, k, T, n, area, lz, FAKE["molar"])
    assert 2 * area * NA * 1e-24 * simpson(c, x=z) == pytest.approx(n, rel=2e-3)


def test_reconstructed_profile_satisfies_the_chemical_potential_balance(hp_mod):
    k, area, lz, n = 1.2, 9.0, 10.0, 60
    z = np.linspace(0.1, 4.5, 40)
    c, c0 = cr.reconstruct_profile(z, k, T, n, area, lz, FAKE["molar"])
    A, B, a1, a2 = FAKE["molar"]
    lng = lambda x: hp_mod.ln_gamma(x, a1, a2, A, B)                           # noqa: E731
    lhs = np.log(c / c0) + lng(c) - lng(c0)
    assert np.allclose(lhs, -(0.5 * k * z ** 2 * 1000.0 / (R * T)), atol=1e-8)


def test_profile_is_monotone_and_peaks_at_the_centre(hp_mod):
    z = np.linspace(0, 5, 100)
    c, c0 = cr.reconstruct_profile(z, 1.5, T, 60, 9.0, 10.0, FAKE["molar"])
    assert np.all(np.diff(c) <= 1e-12) and c[0] == pytest.approx(c0)


def test_molality_from_concentration_inverts_the_density_relation(hp_mod):
    m = np.array([0.1, 1.0, 2.5, 3.5])
    c = cr.density_to_c(m, FAKE["rho_mu"], MS)
    m_back, rho, cw = cr.profile_quantities(c, FAKE, MS)
    assert np.allclose(m_back, m, rtol=2e-3)
    assert np.allclose(rho, hp_mod.density_eq22(m, FAKE["rho_mu"]), rtol=1e-3)
    kgw = rho / (1 + m * MS / 1000.0)                                          # kg water per litre of solution
    assert np.allclose(cw, kgw * 1000.0 / hp_mod.M_WATER_G_PER_MOL, rtol=1e-3)


def test_reference_curves_are_internally_consistent(hp_mod):
    m = np.linspace(0.05, 3.5, 80)
    rc = cr.reference_curves(FAKE, MS, NU, T, m)
    assert rc["Vw"][0] == pytest.approx(hp_mod.M_WATER_G_PER_MOL / 0.997, rel=0.01)
    assert np.allclose(rc["Pi_molar"], hp_mod.pi_bar(rc["c_of_m"], *FAKE["molar"], NU, T))
    assert np.all(rc["rtlnaw"] < 0) and np.all(np.diff(rc["Pi_molar"]) > 0)
    # (no molal-vs-molar agreement check here: FAKE's two parameter sets are invented independently)
    bracket = m + hp_mod.dh_osmotic(m, *FAKE["molal"][:2]) + 0.5 * FAKE["molal"][2] * m ** 2 + (2 / 3) * FAKE["molal"][3] * m ** 3
    assert np.allclose(rc["Pi_molal"], 0.01 * NU * R * T * (hp_mod.M_WATER_G_PER_MOL / rc["Vw"]) * bracket)


def test_run_spec_parsing(hp_mod):
    label, kw = cr.parse_spec("my run=results/x/new,traj=HP_results/x,salt=structures/s.py")
    assert label == "my run" and kw == dict(folder="results/x/new", traj="HP_results/x", salt="structures/s.py")
    assert cr.parse_spec("a=b")[1] == dict(folder="b")
