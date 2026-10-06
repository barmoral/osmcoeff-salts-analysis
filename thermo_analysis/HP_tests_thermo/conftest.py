"""
Fixtures for the HP_analysis_replicates_thermo test suite.

Nothing here touches a cluster, a real trajectory or any literature number. Systems are synthetic, built in
memory with analytic answers, and the checks are physical identities (Gibbs-Duhem, round trips, limits), so the
suite applies to any box size, salt or water model. Run it on its own:

    pytest HP_tests_thermo
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
_SEARCH_DIRS = [PROJECT, PROJECT / "structures"]

import MDAnalysis as mda                                        # noqa: E402
import HP_analysis_replicates_thermo as hp                      # noqa: E402

LX = LY = 3.0
LZ = 10.0
N_PAIRS = 60
N_WATER = 3008
K_HP = 1.5567
T_K = 298.15
R_KJ = 8.31446261815324e-3
K_GAUSS = K_HP / (2 * R_KJ * T_K)


def _locate(name):
    for d in _SEARCH_DIRS:
        if (d / name).exists():
            return d / name
    return None


@pytest.fixture(scope="session")
def hp_mod():
    return hp


def make_universe(n_pairs=N_PAIRS, n_water=N_WATER, lx=LX, ly=LY, lz=LZ, n_frames=6, seed=0,
                  ion_z_nm=None, water_z_nm=None):
    """In-memory Universe: n_pairs NA + n_pairs CL + n_water 3-site HOH.

    ion_z_nm(rng, n) / water_z_nm(rng, n) -> z in nm (default: ions Gaussian about Lz/2 with the width
    k_HP implies; water uniform)."""
    rng = np.random.default_rng(seed)
    if ion_z_nm is None:
        sigma = 1.0 / np.sqrt(2 * K_GAUSS)
        ion_z_nm = lambda r, n: np.clip(r.normal(lz / 2, sigma, n), 0, lz)       # noqa: E731
    if water_z_nm is None:
        water_z_nm = lambda r, n: r.uniform(0, lz, n)                            # noqa: E731
    n_ions = 2 * n_pairs
    n_atoms = n_ions + 3 * n_water
    u = mda.Universe.empty(
        n_atoms, n_residues=n_ions + n_water,
        atom_resindex=np.concatenate([np.arange(n_ions), np.repeat(np.arange(n_ions, n_ions + n_water), 3)]),
        trajectory=True)
    names = ["NA"] * n_pairs + ["CL"] * n_pairs + ["O", "H", "H"] * n_water
    elem = ["Na"] * n_pairs + ["Cl"] * n_pairs + ["O", "H", "H"] * n_water
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", elem)
    u.add_TopologyAttr("element", elem)
    u.add_TopologyAttr("resname", ["NA"] * n_pairs + ["CL"] * n_pairs + ["HOH"] * n_water)
    u.add_TopologyAttr("resid", list(range(1, n_ions + n_water + 1)))
    frames = []
    for _ in range(n_frames):
        pos = np.empty((n_atoms, 3))
        pos[:, 0] = rng.uniform(0, lx * 10, n_atoms)
        pos[:, 1] = rng.uniform(0, ly * 10, n_atoms)
        pos[:n_ions, 2] = ion_z_nm(rng, n_ions) * 10.0
        zw = water_z_nm(rng, n_water) * 10.0                    # one z per molecule: O, H, H share it
        pos[n_ions:, 2] = np.repeat(zw, 3)
        frames.append(pos)
    u.load_new(np.array(frames), order="fac")
    for ts in u.trajectory:
        ts.dimensions = [lx * 10, ly * 10, lz * 10, 90.0, 90.0, 90.0]
    return u


def write_case(tmp_path, n_rep=2, ktag="k1p56", mi="35", **kw):
    """<tmp>/result_files/<ktag>_<mi>m/md<mi>m_r<i>.{pdb,xtc}"""
    d = tmp_path / "result_files" / f"{ktag}_{mi}m"
    d.mkdir(parents=True, exist_ok=True)
    for r in range(n_rep):
        u = make_universe(seed=100 + r, **kw)
        u.atoms.write(str(d / f"md{mi}m_r{r}.pdb"))
        with mda.Writer(str(d / f"md{mi}m_r{r}.xtc"), u.atoms.n_atoms) as w:
            for _ts in u.trajectory:
                w.write(u.atoms)
    return d


@pytest.fixture(scope="session")
def salt_data_module(tmp_path_factory):
    ref, builder = _locate("salt_reference.py"), _locate("build_salt_data.py")
    if ref is None or builder is None:
        pytest.skip("salt_reference.py / build_salt_data.py not found")
    out = tmp_path_factory.mktemp("sd") / "salt_data_test.py"
    proc = subprocess.run(
        [sys.executable, str(builder), "build", "--reference", str(ref), "--box", str(LX), str(LY), str(LZ),
         "--edge-tol", "3.9e-4", "--out", str(out)], cwd=builder.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"build_salt_data.py failed:\n{proc.stdout}\n{proc.stderr}")
    return hp.load_salt_data(str(out))


# ---- analytic "world" used by the thermo round-trip tests -------------------------------------------------
M_SALT = 58.443


def make_world(rho_coeffs=(0.997, 0.0525, -0.0061, -0.0001), m_max=3.0, n=120, Ms=M_SALT):
    """A self-consistent set of profiles from a chosen rho(m) (eq-22 form).

    kg water per L = rho/(1 + m Ms/1000)  (a litre of solution weighs rho kg),
    c_w = kgw*1000/Mw,  c_s = m * kgw.   Returns the `prof` dict thermo_from_profiles expects."""
    m = np.linspace(0.0, m_max, n)
    rho = hp.density_eq22(m, rho_coeffs)
    kgw = rho / (1 + m * Ms / 1000.0)
    c_w = kgw * 1000.0 / hp.M_WATER_G_PER_MOL
    c_s = m * kgw
    z = np.linspace(0.025, 5.0 - 0.025, n)[::-1]            # c_s decreases with z, like an HP profile
    return dict(z=z, rho=rho, c_s=c_s, c_w=c_w, m=m, rho_reps=[rho]), rho_coeffs
