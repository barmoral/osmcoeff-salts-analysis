"""Unit tests for FBP_analysis_replicates: synthetic inputs with known answers, no trajectories needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import FBP_analysis_replicates as fbp

from openmm.unit import (angstrom, bar, kelvin, kilojoule_per_mole, nanometer, mole, litre,
                         AVOGADRO_CONSTANT_NA)

SALT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "structures", "salt_data_48x48x144.py")
K_ANG = (4184.0 * kilojoule_per_mole / nanometer ** 2).in_units_of(kilojoule_per_mole / angstrom ** 2)


def test_mol_keys():
    assert fbp.mol_keys(3.5) == ("35", "3.5")
    assert fbp.mol_keys(4.0) == ("4", "4.0")
    assert fbp.mol_keys(0.5) == ("05", "0.5")


def test_walls_are_centre_minus_plus_half_width():
    lo, hi = fbp.make_walls(7.2, 2.4)
    assert lo.wall_pos[2].value_in_unit(angstrom) == pytest.approx(48.0)
    assert hi.wall_pos[2].value_in_unit(angstrom) == pytest.approx(96.0)


def test_force_is_k_times_penetration_on_the_right_side_only():
    walls = fbp.make_walls(7.2, 2.4)                       # walls at 48 and 96 A
    z = np.array([[[0, 0, 50.0], [0, 0, 45.0], [0, 0, 100.0], [0, 0, 60.0]]])   # one frame, four ions
    coords = z * angstrom
    f_lo = fbp.calculate_force_matrix_all_ions_rel_wall(coords, walls[0].wall_pos, walls[0].wall_normal, K_ANG)
    f_hi = fbp.calculate_force_matrix_all_ions_rel_wall(coords, walls[1].wall_pos, walls[1].wall_normal, K_ANG)
    k = K_ANG.value_in_unit(kilojoule_per_mole / angstrom ** 2)
    assert np.allclose(np.asarray(f_lo.value_in_unit(kilojoule_per_mole / angstrom)), [[0, 3 * k, 0, 0]])
    assert np.allclose(np.asarray(f_hi.value_in_unit(kilojoule_per_mole / angstrom)), [[0, 0, 4 * k, 0]])


def test_osmotic_series_ideal_force_gives_phi_one_and_area_cancels():
    """Choose the total wall force that equals the ideal van't Hoff pressure: phi must be exactly 1."""
    T, i, c = 298.15, 2, 1.0
    R = 0.08314                                            # bar L / (mol K)
    P_ideal = i * c * R * T                                # bar
    for area_nm2 in (20.0, 23.04, 30.0):
        A = np.full(3, area_nm2) * nanometer ** 2
        # P[bar] = F/A/N_A  ->  F = P * A * N_A   (kJ/mol/nm via bar*nm^3 = 0.06022 kJ/mol)
        F = (P_ideal * bar * area_nm2 * nanometer ** 2 * AVOGADRO_CONSTANT_NA).in_units_of(
            kilojoule_per_mole / nanometer) * np.ones(3)
        c_t = np.full(3, c) * (mole / litre)
        press, coeff = fbp.osmotic_series(F, c_t, A, vant_hoff=i, T=T * kelvin)
        assert np.allclose(press, P_ideal, rtol=1e-6)
        assert np.allclose(coeff, 1.0, rtol=1e-6)


def test_replicate_analysis_mean_and_standard_error():
    res = [{'os_press_mean': p * bar, 'os_coeff_mean': c} for p, c in [(100., 1.0), (110., 1.1), (120., 1.2)]]
    out = fbp.replicate_analysis(res, "x")
    assert out['mean_osmotic_pressure'] == pytest.approx(110.0)
    assert out['uncertainty_osmotic_pressure'] == pytest.approx(10.0 / np.sqrt(3))
    assert out['mean_osmotic_coefficient'] == pytest.approx(1.1)


def test_replicate_analysis_single_replicate_is_nan_not_error():
    out = fbp.replicate_analysis([{'os_press_mean': 100. * bar, 'os_coeff_mean': 1.0}], "x")
    assert np.isnan(out['uncertainty_osmotic_pressure']) and np.isnan(out['uncertainty_osmotic_coefficient'])


def test_rebin_conserves_total():
    src = np.linspace(0, 10, 21)
    y = np.random.default_rng(0).random(20)
    dst = np.linspace(-1, 11, 13)
    assert fbp.rebin_piecewise_constant(src, y, dst).sum() == pytest.approx(y.sum())


def test_salt_data_supplies_the_restraint_and_ion_count():
    """The run is driven by salt_data: guards against silently going back to hard-coded config values."""
    SD = fbp.load_salt_data(SALT_DATA)
    r = SD.lookup("NaCl", 3.5)
    assert r.k_FBP_wall == pytest.approx(4184.0)
    assert r.delta_z_FBP == pytest.approx(SD.BOX["z_center"] - 4.7722220347, abs=1e-6)
    assert r.num_particles == 236
    assert SD.VANT_HOFF["NaCl"] == 2


def test_prepare_context_reads_salt_data_and_flags_overrides(capsys, tmp_path):
    cfg = fbp.FBPConfig(salt_data_path=SALT_DATA, results_dir="x/result_files_{mi}m", ion1="Na", ion2="Cl",
                        center_atom1="Na", center_atom2="Cl", N_replicates=1, molalities=[3.5],
                        out_dir=str(tmp_path))
    ctx = fbp.prepare_context(cfg)
    assert ctx.T == pytest.approx(298.15) and ctx.vant_hoff == 2 and ctx.k_nm2 == pytest.approx(4184.0)
    assert ctx.delta_z == pytest.approx(2.4277779652589224)
    assert ctx.rows["35"]["Number of Particles"] == 236
    assert "OVERRIDE" not in capsys.readouterr().out

    cfg.delta_z, cfg.T = 2.4, 300.0
    ctx = fbp.prepare_context(cfg)
    assert (ctx.delta_z, ctx.T) == (2.4, 300.0)
    assert "OVERRIDE" in capsys.readouterr().out


def test_unknown_molality_lists_the_available_ones(tmp_path):
    cfg = fbp.FBPConfig(salt_data_path=SALT_DATA, results_dir="x/{mi}", ion1="Na", ion2="Cl",
                        center_atom1="Na", center_atom2="Cl", N_replicates=1, molalities=[3.33],
                        out_dir=str(tmp_path))
    with pytest.raises(KeyError, match="available"):
        fbp.prepare_context(cfg)


# ---------------------------------------------------------------- k selects the folders
def test_k_tag_matches_the_dispatch_format():
    assert fbp.k_tag(4184) == "k4184" and fbp.k_tag(2092.0) == "k2092" and fbp.k_tag(8368) == "k8368"
    assert fbp.k_tag(0.6606492479786179) == "k0p661" and fbp.k_tag(1.56) == "k1p56"


def _cfg(tmp_path, k, mols=(3.5,), nrep=2):
    return fbp.FBPConfig(salt_data_path=SALT_DATA, results_dir=str(tmp_path / "result_files_{ktag}_{mi}m"),
                         ion1="Na", ion2="Cl", center_atom1="Na", center_atom2="Cl", N_replicates=nrep,
                         molalities=list(mols), k=k, out_dir=str(tmp_path / "out"))


def _make(tmp_path, ktag, mi, nrep=2):
    d = tmp_path / f"result_files_{ktag}_{mi}m"
    d.mkdir()
    for i in range(nrep):
        for ext in ("pdb", "xtc"):
            (d / f"md{mi}m_r{i}.{ext}").write_text("")


def test_k_picks_only_that_k_folder_and_is_the_force_constant(tmp_path, capsys):
    _make(tmp_path, "k2092", "35"); _make(tmp_path, "k8368", "35")
    ctx = fbp.prepare_context(_cfg(tmp_path, 2092))
    assert ctx.ktag == "k2092" and ctx.k_nm2 == 2092
    assert ctx.rdirs["35"] == tmp_path / "result_files_k2092_35m"
    assert ctx.k_ang.value_in_unit(kilojoule_per_mole / angstrom ** 2) == pytest.approx(20.92)
    fbp.check_inputs(_cfg(tmp_path, 2092), ctx)                              # no error: files are there
    ctx8 = fbp.prepare_context(_cfg(tmp_path, 8368))
    assert ctx8.rdirs["35"] == tmp_path / "result_files_k8368_35m"


def test_outputs_go_to_one_folder_per_k_and_molality(tmp_path):
    cfg = _cfg(tmp_path, 8368, mols=(3.5, 4.0))
    ctx = fbp.prepare_context(cfg)
    assert ctx.out_dirs["35"] == tmp_path / "out" / "k8368_35m"
    assert ctx.out_dirs["4"] == tmp_path / "out" / "k8368_4m"
    assert all(d.is_dir() for d in ctx.out_dirs.values())
    cfg.out_dir = None                                                       # default: results/<case>, no k in it
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = fbp.prepare_context(cfg)
    finally:
        os.chdir(cwd)
    assert ctx.out_dirs["35"] == fbp.Path("results") / tmp_path.name / "k8368_35m"


def test_missing_k_folder_says_which_k_exist(tmp_path):
    _make(tmp_path, "k2092", "35"); _make(tmp_path, "k8368", "35")
    cfg = _cfg(tmp_path, 4184)
    with pytest.raises(FileNotFoundError, match=r"k4184.*folder not found.*k2092, k8368"):
        fbp.check_inputs(cfg, fbp.prepare_context(cfg))


def test_empty_folder_and_too_many_replicates_are_reported(tmp_path):
    (tmp_path / "result_files_k4184_35m").mkdir()
    cfg = _cfg(tmp_path, 4184)
    with pytest.raises(FileNotFoundError, match=r"4 of 4 files missing.*md35m_r0\.pdb"):
        fbp.check_inputs(cfg, fbp.prepare_context(cfg))
    _make(tmp_path, "k2092", "35", nrep=2)
    cfg = _cfg(tmp_path, 2092, nrep=3)
    with pytest.raises(FileNotFoundError, match=r"2 of 6 files missing"):
        fbp.check_inputs(cfg, fbp.prepare_context(cfg))


# ---------------------------------------------------------------- results not synced to salt_data
class _Ions(list):
    pass


def _ions(n_ions, nrep=2):
    return {"35": [list(range(n_ions)) for _ in range(nrep)]}            # len() is all resolve_pairs uses


def test_n_pairs_dict_overrides_salt_data(tmp_path, capsys):
    cfg = _cfg(tmp_path, 2092)
    cfg.n_pairs = {3.5: 233}
    ctx = fbp.prepare_context(cfg)
    assert ctx.rows["35"]["Number of Particles"] == 233
    assert "OVERRIDE (salt_data: 236)" in capsys.readouterr().out
    cfg.n_pairs = "oops"
    with pytest.raises(ValueError, match="n_pairs must be"):
        fbp.prepare_context(cfg)


def test_pairs_from_trajectory_replaces_salt_data(tmp_path, capsys):
    cfg = _cfg(tmp_path, 2092)
    cfg.n_pairs = "trajectory"
    ctx = fbp.prepare_context(cfg)
    fbp.resolve_pairs(cfg, ctx, _ions(466))                                   # 233 pairs x i=2
    assert ctx.rows["35"]["Number of Particles"] == 233
    assert "counted in the trajectory; salt_data: 236" in capsys.readouterr().out


def test_ion_count_mismatch_is_a_warning_not_a_silent_error(tmp_path, capsys):
    cfg = _cfg(tmp_path, 2092)
    ctx = fbp.prepare_context(cfg)
    fbp.resolve_pairs(cfg, ctx, _ions(466))
    out = capsys.readouterr().out
    assert "WARNING 3.5 m" in out and "466 restrained ions" in out and 'n_pairs="trajectory"' in out
    assert ctx.rows["35"]["Number of Particles"] == 236                       # unchanged unless asked
    fbp.resolve_pairs(cfg, ctx, _ions(472))
    assert "WARNING" not in capsys.readouterr().out                           # consistent: silent


def test_replicates_with_different_ion_counts_are_an_error(tmp_path):
    cfg = _cfg(tmp_path, 2092)
    ctx = fbp.prepare_context(cfg)
    with pytest.raises(ValueError, match="different numbers of restrained ions"):
        fbp.resolve_pairs(cfg, ctx, {"35": [list(range(472)), list(range(466))]})


def _wall_ions(delta, k, T=300.0, z0=7.2, n_ions=472, n_frames=400, seed=0):
    """Ions uniform between the walls with Boltzmann tails past them: the distribution check_wall_position assumes."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(0.0083144626 * T / k)
    n = n_ions * n_frames
    p_plateau = 2 * delta / (2 * delta + 2 * s * np.sqrt(np.pi / 2))
    inside = rng.random(n) < p_plateau
    z = np.where(inside, z0 + rng.uniform(-delta, delta, n),
                 z0 + rng.choice([-1.0, 1.0], n) * (delta + np.abs(rng.normal(0, s, n))))
    xyz = np.zeros((n_frames, n_ions, 3))
    xyz[..., 2] = z.reshape(n_frames, n_ions) * 10.0                          # angstrom
    return xyz * angstrom


@pytest.mark.parametrize("k", [2092.0, 4184.0, 8368.0])
def test_wall_position_estimate_recovers_delta_and_is_silent_when_right(k, capsys):
    delta = 2.4
    ions = _wall_ions(delta, k)
    est = fbp.check_wall_position(ions, 0, 7.2, delta, k, 300.0)
    assert est == pytest.approx(delta, abs=0.006)
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("k", [2092.0, 8368.0])
def test_wall_position_warns_when_delta_z_is_wrong(k, capsys):
    ions = _wall_ions(2.4, k)                                                 # run had 2.4 nm walls ...
    fbp.check_wall_position(ions, 3, 7.2, 2.4278, k, 300.0)                   # ... but salt_data's 2.4278 is used
    out = capsys.readouterr().out
    assert "WARNING replicate 3" in out and "delta_z ~ 2.4" in out


# ---------------------------------------------------------------- environment
def test_missing_pymbar_fails_immediately_and_names_the_environment(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pymbar", None)                          # makes `import pymbar` raise ImportError
    with pytest.raises(ImportError, match=r"conda activate osmotic-analysis"):
        fbp.check_pymbar()
    cfg = _cfg(tmp_path, 2092)
    with pytest.raises(ImportError, match="pymbar is not installed"):         # run() stops before touching any data
        fbp.run(cfg)
    assert not (tmp_path / "out").exists()


def test_pymbar_present_passes(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymbar", object())
    fbp.check_pymbar()
