"""End-to-end tests of run_analysis() on synthetic trajectories.

Writes real md<mi>m_r<i>.pdb / .xtc files into a temp directory named the way
the dispatch names them, then calls the public entry point. This is the test
that would have caught a wrong results_dir, a mismatched salt_data file, or a
k that disagrees with the run.

profiles_only=True keeps it to a fraction of a second - the bootstraps are not
what is being tested here.
"""
import math
import shutil

import numpy as np
import pytest

from conftest import K_GAUSS, LX, LY, LZ, N_PAIRS, N_WATER, _make_universe


# --------------------------------------------------------------------------
def _write_case(tmp_path, salt_data_module, ktag="k1p56", mi="35",
                n_rep=3, n_water=N_WATER, n_pairs=N_PAIRS, lz=LZ):
    """Create <tmp>/result_files/<ktag>_<mi>m/md<mi>m_r<i>.{pdb,xtc}."""
    d = tmp_path / "result_files" / f"{ktag}_{mi}m"
    d.mkdir(parents=True, exist_ok=True)
    sigma = 1.0 / math.sqrt(2 * K_GAUSS)

    for r in range(n_rep):
        u = _make_universe(
            n_pairs, n_water, LX, LY, lz, n_frames=6,
            ion_z_nm=lambda rng, n: np.clip(rng.normal(lz / 2, sigma, n), 0, lz),
            seed=100 + r,
        )
        u.atoms.write(str(d / f"md{mi}m_r{r}.pdb"))
        with __import__("MDAnalysis").Writer(str(d / f"md{mi}m_r{r}.xtc"),
                                             u.atoms.n_atoms) as w:
            for _ts in u.trajectory:
                w.write(u.atoms)
    return d


@pytest.fixture
def case_dir(tmp_path, salt_data_module):
    return _write_case(tmp_path, salt_data_module)


@pytest.fixture
def sd_path(salt_data_module):
    return salt_data_module.__file__


# ==========================================================================
# happy path
# ==========================================================================
def test_profiles_only_runs_and_returns_expected_keys(hp_mod, case_dir, sd_path):
    out = hp_mod.run_analysis(
        salt_data_path=sd_path, results_dir=str(case_dir),
        ion1="Na", ion2="Cl", molality=3.5, N_replicates=3,
        profiles_only=True, n_expansion_terms=2, dz_nm=0.05,
    )
    for key in ("z", "c_0", "z_fit", "c_fit", "mean_molal",
                "cmax_ideal", "N_s", "LxLy"):
        assert key in out, f"missing '{key}' in the profiles_only result"


def test_measured_ion_count_matches_salt_data(hp_mod, case_dir, sd_path,
                                              salt_data_module):
    out = hp_mod.run_analysis(
        salt_data_path=sd_path, results_dir=str(case_dir),
        ion1="Na", ion2="Cl", molality=3.5, N_replicates=3,
        profiles_only=True, n_expansion_terms=2,
    )
    assert out["N_s"] == salt_data_module.lookup("NaCl", 3.5).num_particles


def test_cross_section_and_peak_are_physical(hp_mod, case_dir, sd_path):
    out = hp_mod.run_analysis(
        salt_data_path=sd_path, results_dir=str(case_dir),
        ion1="Na", ion2="Cl", molality=3.5, N_replicates=3,
        profiles_only=True, n_expansion_terms=2,
    )
    assert out["LxLy"] == pytest.approx(LX * LY, rel=1e-6)

    # The synthetic ions ARE the ideal Gaussian, so the measured peak should
    # match cmax_ideal. Average the central 0.5 nm rather than reading one
    # 0.05 nm bin: that bin holds ~2 ions/frame, i.e. ~30% Poisson noise.
    z, c = out["z"], out["c_0"]
    core = c[z < 0.5].mean()
    # analytic mean of C(0)exp(-z^2/2 sigma^2) over [0, 0.5] is 0.974 C(0)
    ratio = core / (0.974 * out["cmax_ideal"])
    assert 0.8 < ratio < 1.2, f"measured/ideal = {ratio:.2f}"


def test_figures_are_written(hp_mod, case_dir, sd_path, tmp_path):
    outdir = tmp_path / "figs"
    hp_mod.run_analysis(
        salt_data_path=sd_path, results_dir=str(case_dir), out_dir=str(outdir),
        ion1="Na", ion2="Cl", molality=3.5, N_replicates=3,
        profiles_only=True, n_expansion_terms=2,
    )
    names = {p.name for p in outdir.glob("*.png")}
    for expect in ("countsprof_NaCl.png", "concprof_NaCl.png",
                   "conc_prof_byreplicate_NaCl.png"):
        assert expect in names, f"{expect} not written; got {sorted(names)}"


def test_dz_nm_changes_the_profile_length(hp_mod, case_dir, sd_path):
    a = hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(case_dir),
                            ion1="Na", ion2="Cl", molality=3.5, N_replicates=2,
                            profiles_only=True, n_expansion_terms=2, dz_nm=0.05)
    b = hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(case_dir),
                            ion1="Na", ion2="Cl", molality=3.5, N_replicates=2,
                            profiles_only=True, n_expansion_terms=2, dz_nm=0.10)
    assert len(a["c_0"]) == pytest.approx(2 * len(b["c_0"]), abs=1)


# ==========================================================================
# guards - each of these was a real failure at some point
# ==========================================================================
def test_wrong_k_tag_in_directory_name_raises(hp_mod, tmp_path,
                                              salt_data_module, sd_path):
    """A run made with dispatch -k 0.68 lands in k0p68_35m. Analysing it with
    a salt_data that says k = 1.5567 must refuse."""
    d = _write_case(tmp_path, salt_data_module, ktag="k0p68")
    with pytest.raises(ValueError, match="k mismatch"):
        hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(d),
                            ion1="Na", ion2="Cl", molality=3.5, N_replicates=1,
                            profiles_only=True)


def test_k_override_permits_the_mismatch(hp_mod, tmp_path, salt_data_module,
                                         sd_path):
    d = _write_case(tmp_path, salt_data_module, ktag="k0p68")
    out = hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(d),
                              ion1="Na", ion2="Cl", molality=3.5,
                              N_replicates=1, profiles_only=True,
                              n_expansion_terms=2, k_override=0.68)
    assert out is not None


def test_wrong_water_count_raises(hp_mod, tmp_path, salt_data_module, sd_path):
    """Pointing at trajectories from a different box must fail loudly."""
    d = _write_case(tmp_path, salt_data_module, n_water=N_WATER + 500)
    with pytest.raises(ValueError, match="waters"):
        hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(d),
                            ion1="Na", ion2="Cl", molality=3.5, N_replicates=1,
                            profiles_only=True)


def test_missing_replicate_raises(hp_mod, case_dir, sd_path):
    """Asking for more replicates than exist must name the missing file."""
    with pytest.raises(FileNotFoundError, match=r"md35m_r3\.pdb"):
        hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(case_dir),
                            ion1="Na", ion2="Cl", molality=3.5, N_replicates=10,
                            profiles_only=True)


def test_missing_results_dir_raises(hp_mod, sd_path, tmp_path):
    with pytest.raises(FileNotFoundError):
        hp_mod.run_analysis(salt_data_path=sd_path,
                            results_dir=str(tmp_path / "nope"),
                            ion1="Na", ion2="Cl", molality=3.5, N_replicates=1,
                            profiles_only=True)


def test_molality_absent_from_salt_data_raises(hp_mod, case_dir, sd_path):
    with pytest.raises(KeyError):
        hp_mod.run_analysis(salt_data_path=sd_path, results_dir=str(case_dir),
                            ion1="Na", ion2="Cl", molality=99.0, N_replicates=1,
                            profiles_only=True)


# ==========================================================================
# bootstrap normalisation, without paying for the ML bootstrap
# ==========================================================================
def test_hist_to_molar_matches_the_profile_path(hp_mod, gaussian_universe):
    """REGRESSION: the bootstrap divided by a bogus bin-width term and inverted
    the nm^-3 -> mol/L conversion, overstating the profile by 1.45x for a 10 nm
    box. It printed as `bootstrap[0]/c_0 = 1.45`.

    The two routes to the same profile must agree:
      * molarity_profile_fixed_Lz - counts / (2 * A * w), per species
      * hist_to_molar             - density histogram of the pooled samples
    """
    u = gaussian_universe
    A = hp_mod.get_average_cross_section_area(u)
    dz = 0.05

    _z1, c1 = hp_mod.molarity_profile_fixed_Lz(u, u.select_atoms("resname NA"), dz)
    _z2, c2 = hp_mod.molarity_profile_fixed_Lz(u, u.select_atoms("resname CL"), dz)
    c_profile = 0.5 * (c1 + c2)

    lz = u.trajectory.ts.dimensions[2] / 10.0
    edges = np.linspace(0, lz / 2, len(c_profile) + 1)
    zvals = hp_mod.get_ion_array(u, u.select_atoms("resname NA CL"))
    dens, _ = np.histogram(zvals, bins=edges, density=True)
    c_hist = hp_mod.hist_to_molar(dens, N_PAIRS, A)

    m = c_profile > 1e-4
    ratio = np.median(c_hist[m] / c_profile[m])
    assert ratio == pytest.approx(1.0, abs=0.02), (
        f"hist_to_molar / molarity_profile = {ratio:.4f}; 1.45 means the old "
        f"normalisation is back")


def test_hist_to_molar_on_a_uniform_distribution(hp_mod):
    """Analytic case: N pairs uniform over the box give N/(A*Lz)."""
    n_pairs, area, lz = 60, 9.0, 10.0
    edges = np.linspace(0, lz / 2, 101)
    rng = np.random.default_rng(0)
    z = np.abs(rng.uniform(0, lz, 2 * n_pairs * 400) - lz / 2)
    dens, _ = np.histogram(z, bins=edges, density=True)
    c = hp_mod.hist_to_molar(dens, n_pairs, area)
    expect = (n_pairs / (area * lz)) / (6.02214076e23 * 1e-24)
    assert np.median(c) == pytest.approx(expect, rel=0.05)


def test_hist_to_molar_scales_correctly(hp_mod):
    d = np.array([0.2, 0.1])
    base = hp_mod.hist_to_molar(d, 60, 9.0)
    np.testing.assert_allclose(hp_mod.hist_to_molar(d, 120, 9.0), 2 * base)
    np.testing.assert_allclose(hp_mod.hist_to_molar(d, 60, 18.0), base / 2)
