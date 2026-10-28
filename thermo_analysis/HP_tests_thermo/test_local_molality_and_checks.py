"""Local molality converter, reservoir density, and the optional checks."""
import numpy as np
import pytest

from conftest import make_world, M_SALT


# ----------------------------------------------------------------------- local molality
def test_local_molality_recovers_the_true_molality(hp_mod):
    prof, _ = make_world()
    to_m = hp_mod.make_local_molal(prof, hp_mod.M_WATER_G_PER_MOL)
    ok = prof["c_s"] > 0.05
    assert np.allclose(to_m(prof["c_s"][ok]), prof["m"][ok], rtol=2e-3)


def test_box_average_conversion_is_worse_than_local(hp_mod):
    prof, _ = make_world()
    ok = prof["c_s"] > 0.05
    local = hp_mod.make_local_molal(prof, hp_mod.M_WATER_G_PER_MOL)(prof["c_s"][ok])
    kg_water_per_L = prof["c_w"] * hp_mod.M_WATER_G_PER_MOL / 1000.0
    box = prof["c_s"][ok] / kg_water_per_L.mean()
    err_local = np.max(np.abs(local / prof["m"][ok] - 1))
    err_box = np.max(np.abs(box / prof["m"][ok] - 1))
    assert err_box > 3 * err_local


def test_local_molality_is_finite_and_zero_at_zero_and_clips_outside_the_data(hp_mod):
    prof, _ = make_world()
    to_m = hp_mod.make_local_molal(prof, hp_mod.M_WATER_G_PER_MOL)
    out = to_m(np.array([0.0, 1e3]))
    assert out[0] == 0.0 and np.isfinite(out[1])


def test_local_molality_approaches_one_over_pure_water_density_at_low_c(hp_mod):
    prof, coeffs = make_world()
    to_m = hp_mod.make_local_molal(prof, hp_mod.M_WATER_G_PER_MOL)
    c = 0.03
    assert to_m(np.array([c]))[0] / c == pytest.approx(1.0 / coeffs[0], rel=0.01)


# ------------------------------------------------------------------------ reservoir density
def test_reservoir_density_reads_the_box_ends(hp_mod):
    prof, _ = make_world()
    z = prof["z"]
    rho = np.where(z > 3.9, 1.0420, 1.10)               # the last 1 nm of the folded profile is the reservoir
    prof["rho_reps"] = [rho, rho * 1.001]
    got = hp_mod.reservoir_density(prof)
    assert got[0] == pytest.approx(1.0420) and got[1] == pytest.approx(1.0420 * 1.001)


def test_reservoir_check_ignore_is_silent_and_never_raises(hp_mod, capsys):
    bad = hp_mod.reservoir_check([1.09, 0.99], "ignore", target=0.997)
    assert capsys.readouterr().out == "" and bad == [0, 1]


def test_reservoir_check_warn_prints_and_flags(hp_mod, capsys):
    bad = hp_mod.reservoir_check([0.9972, 1.09], "warn", target=0.997)
    out = capsys.readouterr().out
    assert bad == [1] and "OFF" in out and "WARNING" in out


def test_reservoir_check_strict_raises(hp_mod):
    with pytest.raises(ValueError, match="off target"):
        hp_mod.reservoir_check([1.09], "strict", target=0.997)


def test_reservoir_check_without_a_target_never_flags(hp_mod, capsys):
    """The target is the pure-water density of YOUR model; without one there is nothing to be 'off'."""
    assert hp_mod.reservoir_check([1.09, 0.5], "strict", target=None) == []
    assert "OFF" not in capsys.readouterr().out


def test_invalid_check_mode_raises(hp_mod):
    with pytest.raises(ValueError, match="checks must be one of"):
        hp_mod.reservoir_check([1.0], "loud")


def test_box_check_modes(hp_mod, capsys):
    assert hp_mod.box_check([9.0, 8.4], 9.0, "ignore") == [1]
    assert capsys.readouterr().out == ""
    hp_mod.box_check([9.0, 8.4], 9.0, "warn")
    assert "Lx*Ly" in capsys.readouterr().out
    with pytest.raises(ValueError):
        hp_mod.box_check([9.0, 8.4], 9.0, "strict")
    assert hp_mod.box_check([9.0, 9.1], 9.0, "strict") == []


def test_reservoir_note_reports_mean_range_and_target(hp_mod):
    s = hp_mod.reservoir_note([0.99, 1.01], target=1.0)
    assert "1.0000" in s and "0.9900" in s and "1.0100" in s and "target" in s
    assert "target" not in hp_mod.reservoir_note([1.0])


# ------------------------------------------------------------------------- results table
def test_format_dict_turns_nan_into_null_and_stays_valid_json(hp_mod):
    import json
    out = hp_mod.format_dict({"a": [1.23456, float("nan")], "b": {"c": float("nan")}})
    assert out == {"a": [1.235, None], "b": {"c": None}}
    json.dumps(out, allow_nan=False)


def test_blank_nan(hp_mod):
    assert hp_mod._blank_nan(float("nan")) == "" and hp_mod._blank_nan(1.5) == 1.5


# --------------------------------------------------------------- wrapping at the box edge
def _periodic_uniform_with_overhang(lz, edge=0.05):
    """Uniform in the periodic box, but half of the atoms within `edge` nm of a wall are reported on the
    other side of it (z < 0 or z > Lz), as happens when molecules are wrapped by their centre."""
    def f(rng, n):
        z = rng.uniform(0.0, lz, n)
        flip = rng.uniform(0, 1, n) < 0.5
        z = np.where((z < edge) & flip, z + lz, z)
        z = np.where((z > lz - edge) & flip, z - lz, z)
        return z
    return f


def test_atoms_just_outside_the_box_are_wrapped_not_dropped(hp_mod):
    """REGRESSION: molecules are wrapped by their centre, so a few atoms sit a fraction of an A beyond [0, Lz).
    The folded histogram dropped them, under-counting the outermost bin (water 'density' 0.88 there)."""
    from conftest import make_universe
    lz = 10.0
    u = make_universe(n_pairs=10, n_water=6000, lz=lz, n_frames=12, seed=5,
                      water_z_nm=_periodic_uniform_with_overhang(lz))
    water = u.select_atoms("resname HOH and name O*")
    assert (water.positions[:, 2] > lz * 10).any() or (water.positions[:, 2] < 0).any()    # the overhang is there
    _z, c_w = hp_mod.molarity_profile_fixed_Lz(u, water, 0.05)
    bulk = c_w[:-5].mean()
    assert c_w[-1] == pytest.approx(bulk, rel=0.2)             # without the wrap this bin lost ~50%


def test_ion_array_wraps_too(hp_mod):
    from conftest import make_universe
    u = make_universe(n_pairs=5, n_water=10, lz=10.0, n_frames=2, seed=1,
                      ion_z_nm=lambda rng, n: np.full(n, 10.01))                         # 0.1 A beyond the top
    z = hp_mod.get_ion_array(u, u.select_atoms("resname NA CL"))
    assert np.allclose(z, 4.99, atol=1e-4)                     # wraps to 0.1 A, folds to |0.01 - 5| nm
