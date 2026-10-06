"""
Shared fixtures.

Nothing here touches the cluster or a real trajectory. Synthetic systems are
built in memory with known answers, so a test failure means the code changed,
never that a file moved.

Layout assumed:

    <project>/
        HP_analysis_replicates.py
        build_salt_data.py
        salt_reference.py
        tests/
            conftest.py   <- this file
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# build_salt_data.py / salt_reference.py may sit at the project root or be
# filed away with the salt_data_*.py outputs. Look in both.
_SEARCH_DIRS = [PROJECT, PROJECT / "structures"]


def _locate(name):
    for d in _SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None

import MDAnalysis as mda                                        # noqa: E402
import HP_analysis_replicates as hp                             # noqa: E402


# --------------------------------------------------------------------------
# reference numbers for the synthetic system
# --------------------------------------------------------------------------
N_PAIRS = 60
N_WATER = 3008
LX = LY = 3.0        # nm
LZ = 10.0            # nm
K_HP = 1.5567        # kJ/mol/nm^2
T_K = 298.15
R_KJ = 8.31446261815324e-3
K_GAUSS = K_HP / (2 * R_KJ * T_K)        # nm^-2, ~0.314


@pytest.fixture(scope="session")
def hp_mod():
    """The analysis module under test."""
    return hp


@pytest.fixture(scope="session")
def project_dir():
    return PROJECT


def _make_universe(n_pairs, n_water, lx, ly, lz, n_frames, ion_z_nm, seed=0):
    """Build an in-memory Universe: n_pairs NA + n_pairs CL + n_water HOH.

    ion_z_nm(rng, n) -> z positions in nm for one frame. Water is uniform.
    Water is 3-site (O, H, H) so counts and selections behave like the real
    topology; positions of H are irrelevant to every function under test.
    """
    rng = np.random.default_rng(seed)
    n_ions = 2 * n_pairs
    n_wat_atoms = 3 * n_water
    n_atoms = n_ions + n_wat_atoms

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_ions + n_water,
        atom_resindex=np.concatenate([
            np.arange(n_ions),
            np.repeat(np.arange(n_ions, n_ions + n_water), 3),
        ]),
        trajectory=True,
    )
    u.add_TopologyAttr("name",
                       ["NA"] * n_pairs + ["CL"] * n_pairs
                       + ["O", "H", "H"] * n_water)
    u.add_TopologyAttr("type",
                       ["Na"] * n_pairs + ["Cl"] * n_pairs
                       + ["O", "H", "H"] * n_water)
    u.add_TopologyAttr("element",
                       ["Na"] * n_pairs + ["Cl"] * n_pairs
                       + ["O", "H", "H"] * n_water)
    u.add_TopologyAttr("resname",
                       ["NA"] * n_pairs + ["CL"] * n_pairs + ["HOH"] * n_water)
    u.add_TopologyAttr("resid", list(range(1, n_ions + n_water + 1)))

    frames = []
    for _ in range(n_frames):
        pos = np.empty((n_atoms, 3))
        pos[:, 0] = rng.uniform(0, lx * 10, n_atoms)
        pos[:, 1] = rng.uniform(0, ly * 10, n_atoms)
        pos[:n_ions, 2] = ion_z_nm(rng, n_ions) * 10.0          # nm -> A
        pos[n_ions:, 2] = rng.uniform(0, lz * 10, n_wat_atoms)
        frames.append(pos)

    u.load_new(np.array(frames), order="fac")
    for ts in u.trajectory:
        ts.dimensions = [lx * 10, ly * 10, lz * 10, 90.0, 90.0, 90.0]
    return u


@pytest.fixture
def uniform_universe():
    """Ions spread uniformly over the whole box.

    Exact answers: the folded profile is flat and integrates to N_PAIRS, and
    the concentration is N_pairs / (Lx*Ly*Lz).
    """
    return _make_universe(
        N_PAIRS, N_WATER, LX, LY, LZ, n_frames=8,
        ion_z_nm=lambda rng, n: rng.uniform(0, LZ, n),
        seed=1,
    )


@pytest.fixture
def gaussian_universe():
    """Ions in a Gaussian about z0 = Lz/2 with exactly the width k_HP implies.

    C(z) ~ exp(-K z^2), K = k/(2RT), so sigma = 1/sqrt(2K).
    """
    sigma = 1.0 / np.sqrt(2 * K_GAUSS)

    def zfun(rng, n):
        z = rng.normal(LZ / 2, sigma, n)
        return np.clip(z, 0.0, LZ)

    return _make_universe(N_PAIRS, N_WATER, LX, LY, LZ, n_frames=12,
                          ion_z_nm=zfun, seed=2)


@pytest.fixture
def offcentre_universe():
    """Ions Gaussian about z0 = Lz/2 but in a box of a DIFFERENT length.

    Used to prove the fold centre is taken from the trajectory, not hard-coded.
    """
    lz = 14.4
    sigma = 1.0 / np.sqrt(2 * K_GAUSS)

    def zfun(rng, n):
        return np.clip(rng.normal(lz / 2, sigma, n), 0.0, lz)

    return _make_universe(N_PAIRS, N_WATER, LX, LY, lz, n_frames=6,
                          ion_z_nm=zfun, seed=3)


@pytest.fixture(scope="session")
def salt_data_module(tmp_path_factory, project_dir):
    """Generate a real salt_data file with build_salt_data.py and import it.

    Exercises the generator and the consumer together: if the schema changes on
    one side, these tests fail.
    """
    ref = _locate("salt_reference.py")
    builder = _locate("build_salt_data.py")
    if ref is None or builder is None:
        pytest.skip(
            "salt_reference.py / build_salt_data.py not found in "
            + " or ".join(str(d) for d in _SEARCH_DIRS))

    out = tmp_path_factory.mktemp("sd") / "salt_data_test.py"
    proc = subprocess.run(
        [sys.executable, str(builder), "build",
         "--reference", str(ref),
         "--box", str(LX), str(LY), str(LZ),
         "--edge-tol", "3.9e-4",
         "--out", str(out)],
        cwd=builder.parent, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"build_salt_data.py failed:\n{proc.stdout}\n{proc.stderr}")
    return hp.load_salt_data(str(out))
