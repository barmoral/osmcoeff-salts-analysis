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
