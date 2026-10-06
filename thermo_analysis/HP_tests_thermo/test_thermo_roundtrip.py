"""thermo_from_profiles on analytic, self-consistent synthetic data: round trips and thermodynamic identities.
No simulation, no literature numbers - any salt / box / water model must satisfy these."""
import numpy as np
import pytest

from conftest import make_world, M_SALT, T_K

NU = 2


def _molar_method(hp, prof, B, a1, a2, A):
    c = prof["c_s"][1:]
    op = hp.pi_bar(c, A, B, a1, a2, NU, T_K)
    return dict(params=[B, a1, a2], c_molar=c, op_bar=op, oc=op / (0.01 * NU * hp.R * T_K * c))


def _run(hp, prof, method, tmp_path=None, **kw):
    kw.setdefault("make_plots", False)
    kw.setdefault("verbose", False)
    return hp.thermo_from_profiles(prof, str(tmp_path or "."), "XY", "1", M_SALT, NU, T_K, kw.pop("A", 1.79),
                                   {"ml": method}, **kw)


def test_eq22_density_coefficients_are_recovered_exactly(hp_mod):
    prof, true = make_world()
    r = _run(hp_mod, prof, _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79), kappa=0.0)
    assert np.allclose(r["coeffs_mu"], true, atol=1e-7)


def test_no_pressure_correction_when_kappa_is_zero(hp_mod):
    prof, _ = make_world()
    r = _run(hp_mod, prof, _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79), kappa=0.0)
    assert np.allclose(r["res"]["ml"]["coeffs_P"], r["coeffs_mu"], atol=1e-7)


def test_eq23_decompression_is_applied_to_the_density(hp_mod):
    prof, _ = make_world()
    meth = _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79)
    kappa = 4e-5
    r = _run(hp_mod, prof, meth, kappa=kappa)
    m = np.linspace(0.1, 3.0, 20)
    Pi = np.interp(m, prof["m"][1:], meth["op_bar"])
    expect = hp_mod.density_eq22(m, r["coeffs_mu"]) * (1 - kappa * Pi)
    got = hp_mod.density_eq22(m, r["res"]["ml"]["coeffs_P"])
    assert np.max(np.abs(got - expect)) < 3e-4                    # refit to the eq-22 form, so not exact


def test_partial_molar_volume_of_water_limits(hp_mod):
    prof, true = make_world()
    r = _run(hp_mod, prof, _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79), kappa=0.0)
    res = r["res"]["ml"]
    assert res["Vw_raw"][0] == pytest.approx(hp_mod.M_WATER_G_PER_MOL / true[0], rel=5e-3)
    assert np.max(np.abs(res["Vw"] - res["Vw_raw"])) < 3e-3         # cubic smoothing is harmless
    assert res["Vw_raw"][-1] < res["Vw_raw"][0]                     # water volume shrinks with salt here


def test_A_tilde_and_B_tilde_follow_eq19_and_the_rho_override(hp_mod):
    prof, true = make_world()
    meth = _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79)
    r = _run(hp_mod, prof, meth, kappa=0.0, A=1.79)
    assert r["res"]["ml"]["A_m"] == pytest.approx(1.79 * np.sqrt(true[0]), rel=1e-6)
    assert r["res"]["ml"]["B_m"] == pytest.approx(1.2 * np.sqrt(true[0]), rel=1e-6)
    r1 = _run(hp_mod, prof, meth, kappa=0.0, A=1.79, rho_w0_ref=1.0)
    assert r1["res"]["ml"]["A_m"] == pytest.approx(1.79) and r1["res"]["ml"]["B_m"] == pytest.approx(1.2)


def _molal_world(hp, At, Bt, al1, al2):
    """Pi(m) generated FROM a known molal model, V_w from the same density (kappa = 0)."""
    prof, true = make_world()
    s = np.sqrt(true[0])
    m_grid = np.linspace(0.02, 3.0, 200)
    Vraw = hp.partial_molar_volume_water(m_grid, true, M_SALT)
    Vw = np.polyval(np.polyfit(m_grid, Vraw, 3), m_grid)
    bracket = m_grid + hp.dh_osmotic(m_grid, At, Bt) + 0.5 * al1 * m_grid ** 2 + (2 / 3) * al2 * m_grid ** 3
    Pi = 0.01 * NU * hp.R * T_K * (hp.M_WATER_G_PER_MOL / Vw) * bracket
    c = np.interp(m_grid, prof["m"], prof["c_s"])
    meth = dict(params=[Bt / s, 0.0, 0.0], c_molar=c, op_bar=Pi, oc=Pi / (0.01 * NU * hp.R * T_K * c))
    return prof, meth, At / s, m_grid


def test_molal_alpha_round_trip(hp_mod):
    At, Bt, al1, al2 = 1.7, 1.5, 0.12, 0.015
    prof, meth, A_fit, m_grid = _molal_world(hp_mod, At, Bt, al1, al2)
    r = _run(hp_mod, prof, meth, kappa=0.0, A=A_fit)
    res = r["res"]["ml"]
    assert res["al1"] == pytest.approx(al1, rel=5e-3)
    assert res["al2"] == pytest.approx(al2, rel=2e-2)
    assert np.allclose(res["lng"], hp_mod.ln_gamma(m_grid, al1, al2, At, Bt), atol=2e-3)


def test_water_chemical_potential_two_routes_agree(hp_mod):
    """mu_w - mu_w0 = RT ln a_w from the osmotic bracket must equal -Pi * V_w."""
    prof, meth, A_fit, _ = _molal_world(hp_mod, 1.7, 1.5, 0.12, 0.015)
    res = _run(hp_mod, prof, meth, kappa=0.0, A=A_fit)["res"]["ml"]
    assert np.allclose(res["rtlnaw"], res["rtlnaw_fromPi"], rtol=2e-3, atol=1e-6)
    assert np.all(res["rtlnaw"] < 0)


def test_gibbs_duhem_holds_between_the_reported_phi_and_ln_gamma(hp_mod):
    prof, meth, A_fit, m = _molal_world(hp_mod, 1.7, 1.5, 0.12, 0.015)
    res = _run(hp_mod, prof, meth, kappa=0.0, A=A_fit)["res"]["ml"]
    phi, lng = res["phi_molal"], res["lng"]
    lhs = m * np.gradient(lng, m)
    rhs = m * np.gradient(phi, m) + (phi - 1)
    assert np.max(np.abs(lhs - rhs)[5:-5]) < 2e-3


def test_molal_description_reproduces_the_molar_osmotic_pressure(hp_mod):
    prof, _ = make_world()
    meth = _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79)
    res = _run(hp_mod, prof, meth, kappa=0.0)["res"]["ml"]
    assert np.sqrt(np.mean((res["op_molal"] - res["Pi"]) ** 2)) < 0.01 * res["Pi"].max()


def test_plots_files_and_reservoir_stamp(hp_mod, tmp_path, monkeypatch):
    import matplotlib.axes
    texts = []
    orig = matplotlib.axes.Axes.text
    monkeypatch.setattr(matplotlib.axes.Axes, "text",
                        lambda self, *a, **k: (texts.append(a[2] if len(a) > 2 else k.get("s")), orig(self, *a, **k))[1])
    prof, _ = make_world()
    meth = _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79)
    exp_m = np.array([0.5, 1.0, 2.0, 5.0])
    hp_mod.thermo_from_profiles(prof, str(tmp_path), "XY", "1", M_SALT, NU, T_K, 1.79, {"ml": meth},
                                exp_molality=exp_m, exp_oc=np.ones(4), exp_op=np.ones(4),
                                rho_res=[0.9971, 0.9975], reservoir_target=0.997, verbose=False)
    names = {p.name for p in tmp_path.iterdir()}
    for tag in ("density", "pmv_water", "op_molar_vs_molal", "op_vs_molarity", "oc_molar_vs_molal",
                "ln_gamma", "water_activity", "mu_ion_excess"):
        assert f"thermo_{tag}_XY.png" in names
    assert {"thermo_params_XY_1m.json", "thermo_curves_XY_1m.npz", "thermo_results_XY_1m.csv"} <= names
    assert sum("reservoir" in str(t) for t in texts) >= 8          # every thermo plot carries the density
    csv_text = (tmp_path / "thermo_results_XY_1m.csv").read_text().splitlines()
    assert csv_text[-1].split(",")[2] == ""                        # 5 mol/kg is beyond the 3 mol/kg box: blank


def test_optional_reference_curves_are_drawn_only_when_given(hp_mod, tmp_path):
    prof, _ = make_world()
    meth = _molar_method(hp_mod, prof, 1.2, 0.2, 0.01, 1.79)
    ref = dict(label="some literature", molar=(1.79, 1.2, 0.2, 0.01), molal=(1.78, 1.2, 0.2, 0.01),
               rho_mu=(0.997, 0.05, -0.006, 0.0), rho_P=(0.997, 0.05, -0.006, 0.0))
    r = _run(hp_mod, prof, meth, tmp_path, kappa=0.0, reference=ref, make_plots=True)
    assert r["out"]["reference"]["label"] == "some literature"
    assert _run(hp_mod, prof, meth, kappa=0.0)["out"]["reference"] is None
