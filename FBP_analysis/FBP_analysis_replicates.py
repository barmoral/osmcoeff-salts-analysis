"""
FBP Analysis Replicates
=======================
Analysis library for Flat-Bottom Potential (FBP) osmotic-pressure / osmotic-coefficient calculations.

Original notebook code by Barbara Morales; refactored into library + run-file by the FBP_membrane
workspace (fbp_analysis.py / run_FBP_analysis.py). This copy lives in the reorganised project and is
laid out like thermo_analysis/ (HP_analysis_replicates_thermo.py + run_scripts/ + results/).

Nothing in this module executes on import. Drive it with run_scripts/run_FBP_analysis.py, or:

    import FBP_analysis_replicates as fbp
    fbp.run(fbp.FBPConfig(...))

Changes vs the FBP_membrane/fbp_analysis.py this was copied from
------------------------------------------------------------------
  Layout / I/O (no change to the physics)
  1. salt data is loaded BY PATH (structures/salt_data_*.py), like the HP code, instead of
     `sys.path.append('..'); from salt_data import ...`.
  2. Everything the simulation fixed is read from that salt_data file, so the analysis cannot disagree
     with the run: k_FBP_wall, delta_z_FBP, BOX['z_center'], BOX['lz'], BOX['temperature'], the ion-pair
     count per molality and the van't Hoff factor. Each can still be overridden in FBPConfig (printed as OVERRIDE).
     NOTE the old config hard-coded delta_z = 2.4 nm and T = 300 K; salt_data has 2.4278 nm and 298.15 K.
  3. Outputs go under results/<case>/<ktag>_<mi>m/ (e.g. results/FBP_NaCl_TIP3P/k4184_35m/: one folder per k and
     molality; never next to the trajectories, never into the cwd); trajectories are
     read from `results_dir`, a template with {mi} (molality key) and {ktag} (the dispatch's k tag, e.g. k4184),
     so one k at a time is read; the working directory no longer matters for output. The k given in the config
     is the force constant used in the analysis AND selects the folders, so they cannot disagree.
  3b. Results from runs that were NOT made with the current salt_data (the old FBP_membrane runs: delta_z = 2.4 nm,
     T = 300 K, a different ion-pair count) can be analysed with the overrides `delta_z`, `T` and `n_pairs`
     ("trajectory" counts the pairs in the trajectory; or {molality: pairs}). Two checks warn when the inputs
     disagree with the data: the ion count (check pairs) and the wall position (ions' extent vs delta_z).
  4. Headless: matplotlib uses Agg, every plt.show() is gone and figures are closed after saving.
  Small fixes
  5. The "Concentration (mol/L)" per-replicate plot was a count of ion pairs per bin drawn against a molality
     line; it is now labelled for what it is and the meaningless reference line is dropped. The real
     concentration profiles are the concprof_* plots.
  6. The stacked count / concentration plots were built from ALL requested molalities at once and written to the
     last molality's folder; they are now made per molality.
  7. SPW_analysis printed the VARIANCE after "+/-"; it now prints the standard error (the stored *_var keys are unchanged).
  8. replicate_analysis with a single replicate returns NaN uncertainty instead of warning on ddof=1.
  9. Mean box area is compared with salt_data (warning only).

Design notes (unchanged)
------------------------
* Walls sit at z_center -/+ delta_z (the physical slab). L_z only affects profile binning / axes.
* Osmotic pressure = (mean wall force)/area, converted to bar; the coefficient divides by the ideal
  van't Hoff pressure i*c*R*T with c the geometry-derived molarity N_pairs/(A * 2*delta_z). Both are
  evaluated per frame on equilibrated, decorrelated frames, then averaged (area cancels in phi).
* Results are plotted against MOLALITY; the pressure physics stays molarity-based.
"""

import os
import re
import sys
import json
import math
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import MDAnalysis as mda

from openmm.unit import (
    bar, mole, litre, kelvin, kilojoule_per_mole, nanometer, angstrom, molar, picosecond,
)
from openmm.unit import Quantity
from openmm.unit import AVOGADRO_CONSTANT_NA


# =============================================================================
# Physical constants (do not vary between studies -- stay out of the config)
# =============================================================================
NA = 6.02214076E23  # Avogadro's number, used by the molarity profiles


# =============================================================================
# Config: everything a user is expected to change lives here
# =============================================================================
@dataclass
class FBPConfig:
    """User-tunable inputs for one FBP analysis run. Plain numbers; units in the comments.

    Anything left as None is read from the salt_data file the simulation used.
    """
    # --- which system ---
    salt_data_path: str       # e.g. "structures/salt_data_48x48x144.py"
    results_dir: str          # trajectory folder TEMPLATE with {ktag} and {mi}, e.g. "FBP_results/FBP_NaCl_TIP3P/result_files_{ktag}_{mi}m"
    ion1: str                 # cation name as in salt_data / topology, e.g. "Na"
    ion2: str                 # anion name, e.g. "Cl"
    center_atom1: str         # central atom (element) of the cation, e.g. "Na"  (NH4 -> "N")
    center_atom2: str         # central atom (element) of the anion, e.g. "Cl"   (NO3 -> "N")
    N_replicates: int         # replicates to analyse
    molalities: list          # molalities to analyse, mol/kg (must exist in salt_data)

    # --- labels for figures only ---
    water: str = "TIP3P"
    ff_used: Optional[str] = None

    # --- overrides; None = take from salt_data ---
    k: Optional[float] = None            # FBP wall force constant, kJ/mol/nm^2      (salt_data k_FBP_wall)
    delta_z: Optional[float] = None      # slab half-width, nm                        (salt_data delta_z_FBP)
    z_center: Optional[float] = None     # slab centre, nm                            (salt_data BOX['z_center'])
    L_z: Optional[float] = None          # box z length, nm; binning / axes only      (salt_data BOX['lz'])
    T: Optional[float] = None            # kelvin                                     (salt_data BOX['temperature'])
    vantHoff: Optional[int] = None       # van't Hoff factor                          (salt_data VANT_HOFF[salt])
    n_pairs: Optional[object] = None     # ion pairs in the slab: None = salt_data; "trajectory" = count the restrained
                                         # ions in the trajectory / van't Hoff factor; or {molality: pairs}

    # --- analysis knobs ---
    dz_nm: float = 0.05                  # profile bin width, nm
    nskip: int = 100                     # pymbar equilibration-detection stride
    profiles: bool = True                # False = skip the (slow) profile plots, only osmotic values

    # --- output ---
    out_dir: Optional[str] = None        # base folder; default results/<folder holding results_dir>. Each (k, molality) gets <ktag>_<mi>m/ in it

    def __post_init__(self):
        if self.ff_used is None:
            self.ff_used = f"Sage 2.3 + {self.water}"


@dataclass
class FBPContext:
    """Values derived once from a config + salt_data and threaded through the pipeline."""
    SD: object            # the loaded salt_data module
    salt: str             # e.g. "NaCl"
    rows: dict            # mi -> dict of salt_data values for that molality
    walls: dict           # mi -> [Wall, Wall]
    k_nm2: float          # kJ/mol/nm^2
    k_ang: Quantity       # kJ/mol/angstrom^2 (MDAnalysis coordinates are in angstrom)
    delta_z: float        # nm
    z_center: float       # nm
    Lz_nm: float
    Lz_ang: float
    T: float              # K
    vant_hoff: int
    area_ref: float       # nm^2, salt_data box area
    out_dir: Path
    rdirs: dict           # mi -> trajectory folder
    resname_ions: str
    ktag: str = ""        # e.g. "k4184": how the dispatch names the folders of this k
    out_dirs: dict = field(default_factory=dict)   # mi -> <out_dir>/<ktag>_<mi>m, where that run's outputs go


@dataclass
class Wall:
    """Location and orientation of a wall."""
    wall_pos: Quantity
    wall_normal: Quantity


# =============================================================================
# Salt / experimental data
# =============================================================================
def load_salt_data(path):
    """Import a per-box salt_data_*.py by path."""
    spec = importlib.util.spec_from_file_location("salt_data_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def k_tag(k):
    """The dispatch's folder tag for a force constant (osmotic_sim_dispatch_*::_k_to_tag):
    4184 -> 'k4184', 2092.0 -> 'k2092', 0.6606 -> 'k0p661'."""
    k = float(k)
    if k.is_integer():
        return f"k{int(k)}"
    return "k" + f"{k:.3g}".replace('.', 'p').replace('-', 'm')


def mol_keys(mol):
    """(mi, mi1) string keys for a float molality: 1.0 -> ('1','1.0'); 0.5 -> ('05','0.5')."""
    if mol % 1 == 0:
        return str(int(mol)), f"{mol:.1f}"
    return f"{mol:.1f}".replace('.', ''), f"{mol:.1f}"


def salt_rows(SD, salt):
    """All salt_data rows for one salt, sorted by molality."""
    rows = [SD.SaltData(**e) if isinstance(e, dict) else e for e in SD.salt_infos]
    return sorted((r for r in rows if r.salt == salt), key=lambda r: r.molality)


def osm_experimental(exp_osm_coeff, vant_hoff, molarity=1 * (mole / litre), T=300 * kelvin):
    """Osmotic pressure for an experimental osmotic coefficient (molarity-based)."""
    R = 0.08314 * bar * litre / (mole * kelvin)
    molarity = molarity.in_units_of(mole / litre)
    osm_press_ideal = (vant_hoff * molarity * R * T).in_units_of(bar)
    return exp_osm_coeff * osm_press_ideal


# =============================================================================
# Force / geometry primitives
# =============================================================================
def filter_coords_by_wall(coords_rel_wall, wall_normal, dist_unit=angstrom):
    """Zeros out all coordinates which are not in the normal direction relative to a wall."""
    in_right_direction = np.dot(coords_rel_wall, wall_normal) > 0.0
    return np.where(in_right_direction[..., np.newaxis], coords_rel_wall, 0.0) * dist_unit


def calculate_force_matrix_all_ions_rel_wall(coords, wall_pos, wall_normal, K, dist_unit=angstrom):
    """Net harmonic force experienced by particles on one side of a wall.

    coords: (time, ion index, xyz). wall_pos: wall position vector. wall_normal: which side is penalised."""
    coords_rel_wall = coords - wall_pos
    coords_behind_wall = filter_coords_by_wall(coords_rel_wall, wall_normal=wall_normal, dist_unit=dist_unit)
    z_coords_rel_wall = coords_behind_wall[..., 2]
    # NOTE: cannot use np.abs, as this will discard unit info
    return K * abs(z_coords_rel_wall)


def make_walls(z_center_nm, delta_z_nm):
    """The two walls at z_center -/+ delta_z (angstrom, as MDAnalysis coordinates)."""
    lower, upper = (z_center_nm - delta_z_nm) * 10.0, (z_center_nm + delta_z_nm) * 10.0
    return [
        Wall(wall_pos=np.array([0.0, 0.0, lower]) * angstrom, wall_normal=np.array([0.0, 0.0, -1.0]) * angstrom),
        Wall(wall_pos=np.array([0.0, 0.0, upper]) * angstrom, wall_normal=np.array([0.0, 0.0, 1.0]) * angstrom),
    ]


# =============================================================================
# Osmotic physics (molarity-based)
# =============================================================================
def osmotic_values(mean_force_wall, molarity, cross_sectional_area, vant_hoff, T=300 * kelvin, printvals=False):
    """Osmotic pressure and coefficient from the mean wall force."""
    molarity = molarity.in_units_of(mole / litre)
    R = 0.08314 * bar * litre / (mole * kelvin)
    osm_press = mean_force_wall / cross_sectional_area
    osm_bar = osm_press.in_units_of(bar / mole)
    osm_bar = osm_bar / AVOGADRO_CONSTANT_NA

    osm_press_ideal = (vant_hoff * molarity * R * T).in_units_of(bar)
    osm_coeff = osm_bar / osm_press_ideal

    if printvals:
        print("Osmotic Pressure Observed:", osm_bar)
        print("Osmotic Pressure Ideal:", osm_press_ideal)
        print("Osmotic Coefficient:", osm_coeff)
    return osm_bar, osm_coeff


def osmotic_series(force_t, molarity_t, area_t, vant_hoff, T):
    """Vectorized sibling of osmotic_values(): per-frame osmotic pressure (bar) and coefficient,
    returned as plain numpy arrays so the statistics downstream are unit-safe. The area cancels in the
    coefficient when molarity_t and area_t are both per frame, so the coefficient is linear in the force."""
    R = 0.08314 * bar * litre / (mole * kelvin)
    molarity_t = molarity_t.in_units_of(mole / litre)

    osm_press = force_t / area_t
    osm_bar = osm_press.in_units_of(bar / mole) / AVOGADRO_CONSTANT_NA
    osm_press_ideal = (vant_hoff * molarity_t * R * T).in_units_of(bar)

    press_vals = np.asarray(osm_bar.value_in_unit(bar))
    ideal_vals = np.asarray(osm_press_ideal.value_in_unit(bar))
    return press_vals, press_vals / ideal_vals


# =============================================================================
# Per-replicate + across-replicate analysis
# =============================================================================
def SPW_analysis(times, ion_coords, molality_info, A_t, walls, K, repnum, vant_hoff, T, nskip=100):
    """Per-frame osmotic pressure / coefficient + uncorrelated-sample analysis.

    A_t and molality_info['Molarity'] are per-frame arrays. Equilibration and decorrelation are detected on
    the restraint-force timeseries; pressure and coefficient are then computed on each retained
    (uncorrelated) frame and averaged."""
    from pymbar import timeseries            # imported here so the module loads (and the unit tests run) without pymbar
    SPACER = f'\n{"-" * 50}\n'
    print(SPACER)
    _c_mean = float(molality_info['Molarity'].value_in_unit(mole / litre).mean())
    print(f"CONCENTRATION: Molality {molality_info.get('Molality')} mol/kg, "
          f"mean geometry molarity {_c_mean:.4f} mol/L, "
          f"N particles {molality_info.get('Number of Particles')}")
    print("Data for replicate #", repnum)

    force_matrices = [
        calculate_force_matrix_all_ions_rel_wall(
            ion_coords, wall_pos=wall.wall_pos, wall_normal=wall.wall_normal, K=K, dist_unit=angstrom)
        for wall in walls
    ]
    NULL_FORCE = 0.0 * force_matrices[0].unit            # right units to start the sum
    force_matrix = sum(force_matrices, start=NULL_FORCE) / len(force_matrices)
    forces_over_all_ions = force_matrix.sum(axis=1)      # per-frame total restraint force

    # Equilibration + decorrelation on the force timeseries
    forces_val = forces_over_all_ions._value
    t0, g, Neff_max = timeseries.detect_equilibration(forces_val, nskip=nskip)
    uncorr = np.array(timeseries.subsample_correlated_data(forces_val[t0:], g=g))
    sample_idx = t0 + uncorr                             # indices into the full frame axis

    F_s = forces_over_all_ions[sample_idx]
    A_s = A_t[sample_idx]
    c_s = molality_info['Molarity'][sample_idx]

    press_s, coeff_s = osmotic_series(F_s, c_s, A_s, vant_hoff=vant_hoff, T=T)
    se_denom = np.sqrt(max(Neff_max - 1, 1))             # guard tiny-sample div-by-zero

    osm_press_mean = float(press_s.mean()) * bar
    osm_press_std = float(press_s.std() / se_denom) * bar
    osm_coeff_mean = float(coeff_s.mean())
    osm_coeff_std = float(coeff_s.std() / se_denom)

    print("RESULTS:")
    print(f"Osmotic Pressure = {osm_press_mean} +/- {osm_press_std}   (standard error; t0={t0}, g={g:.1f}, "
          f"{len(sample_idx)} uncorrelated frames)")
    print(f"Osmotic Coefficient = {osm_coeff_mean} +/- {osm_coeff_std}")
    print(SPACER)

    return {
        'os_press_mean': osm_press_mean,
        'os_press_var': osm_press_std ** 2,
        'os_coeff_mean': osm_coeff_mean,
        'os_coeff_var': osm_coeff_std ** 2,
    }


def run_analysis(u, atom_group, saltdict, repnum, walls, K, vant_hoff, T, nskip=100, area_ref=None, wall_check=None):
    """Per-frame geometry: read the box at every step, so the cross-sectional area A(t) and the
    geometry-derived molarity c(t) fluctuate with the (XY-isotropic, Z-fixed) barostat. The osmotic
    pressure and coefficient are then evaluated per frame and averaged."""
    m_info = dict(saltdict)

    # One streaming pass: capture time, box, and ONLY the ion positions. Peak RAM stays ~one frame.
    times, area_ang2, ion_pos = [], [], []
    for ts in u.trajectory:
        times.append(ts.time)
        lx, ly, _lz = ts.dimensions[:3]                  # angstrom; Z fixed but read anyway
        area_ang2.append(lx * ly)
        ion_pos.append(atom_group.positions.copy())      # (n_ions, 3), ions only
    times = np.array(times) * picosecond
    A_t = np.array(area_ang2) * angstrom ** 2
    ion_coords = np.array(ion_pos) * angstrom            # (n_frames, n_ions, 3)

    if wall_check is not None:
        check_wall_position(ion_coords, repnum, **wall_check)

    if area_ref is not None:
        a_mean = float(np.mean(area_ang2)) / 100.0
        if abs(a_mean - area_ref) > 0.05 * area_ref:      # the barostat alone expands the box ~2%; 5% = a different box
            print(f"  WARNING replicate {repnum}: mean Lx*Ly = {a_mean:.3f} nm^2 vs salt_data BOX area "
                  f"{area_ref:.3f} nm^2 (>5% apart) - wrong salt_data file for these trajectories?")

    wall_sep = abs((walls[1].wall_pos - walls[0].wall_pos)[2])    # fixed (Z is fixed)

    V_mem_t = A_t * wall_sep
    n_mol = m_info['Number of Particles'] / AVOGADRO_CONSTANT_NA
    molarity_t = (n_mol / V_mem_t).in_units_of(mole / litre)      # per frame

    m_info['Molarity'] = molarity_t
    m_info['Concentration (M)'] = float(molarity_t.value_in_unit(mole / litre).mean())
    m_info['Quantity Type'] = 'molarity (per-frame, geometry-derived)'

    return SPW_analysis(times, ion_coords, m_info, A_t, walls, K,
                        repnum=repnum, vant_hoff=vant_hoff, T=T, nskip=nskip)


def replicate_analysis(results, input_name):
    """Mean and standard error over replicate results for one molality."""
    op_vals = np.array([r['os_press_mean'].value_in_unit(bar) for r in results])
    oc_vals = np.array([r['os_coeff_mean'] for r in results])
    n = len(op_vals)

    mean_op, mean_oc = op_vals.mean(), oc_vals.mean()
    if n > 1:
        op_uncertainty = op_vals.std(ddof=1) / np.sqrt(n)
        oc_uncertainty = oc_vals.std(ddof=1) / np.sqrt(n)
    else:
        op_uncertainty = oc_uncertainty = float('nan')

    print(f"----- Results standard error over {n} replicates for {input_name} molal -----")
    print(f"osmotic pressure: {mean_op:.3f} +/- {op_uncertainty:.3f}")
    print(f"osmotic coefficient: {mean_oc:.3f} +/- {oc_uncertainty:.3f}")

    return {
        'mean_osmotic_pressure': mean_op,
        'uncertainty_osmotic_pressure': op_uncertainty,
        'mean_osmotic_coefficient': mean_oc,
        'uncertainty_osmotic_coefficient': oc_uncertainty,
    }


def load_results(molalities, out_dirs):
    """Load the per-molality final_results_*.json written by this pipeline (out_dirs: mi -> folder)."""
    out = {"OP_Values": [], "OP_Errors": [], "OC_Values": [], "OC_Errors": []}
    for mol in molalities:
        mi, _ = mol_keys(mol)
        with open(Path(out_dirs[mi]) / f'final_results_{mi}m.json') as file:
            r = json.load(file)
        out["OP_Values"].append(r['mean_osmotic_pressure'])
        out["OP_Errors"].append(r['uncertainty_osmotic_pressure'])
        out["OC_Values"].append(r['mean_osmotic_coefficient'])
        out["OC_Errors"].append(r['uncertainty_osmotic_coefficient'])
    return out


# =============================================================================
# Profile helpers
# =============================================================================
def get_ion_array(u, ions):
    """Ion z-coordinates (nm) across the whole trajectory, flattened."""
    ion_array = [ions.positions[:, 2] for _ in u.trajectory]
    return np.array(ion_array).flatten() / 10


def to_float_M(x):
    """Molarity as float (mol/L), whether x is a Quantity or float."""
    try:
        return x.value_in_unit(mole / litre)
    except AttributeError:
        return float(x)


def rebin_piecewise_constant(edges_src, y_src, edges_dst):
    """Rebin piecewise-constant y(z) from edges_src to edges_dst, conserving totals."""
    widths_src = np.diff(edges_src)
    lam = np.zeros_like(y_src, dtype=float)
    valid = widths_src > 0
    lam[valid] = y_src[valid] / widths_src[valid]

    out = np.zeros(len(edges_dst) - 1, dtype=float)
    i = j = 0
    while i < len(y_src) and j < len(out):
        a, b = edges_src[i], edges_src[i + 1]
        A, B = edges_dst[j], edges_dst[j + 1]
        left, right = max(a, A), min(b, B)
        overlap = right - left
        if overlap > 0:
            out[j] += lam[i] * overlap
        if b <= B:
            i += 1
        else:
            j += 1
    return out


def count_profile_fixed_Lz(u, ag, dz_nm=0.05):
    """Time-averaged counts/bin along z using a fixed Lz from the first frame."""
    Lz_nm = u.trajectory.ts.dimensions[2] / 10.0
    edges = np.arange(-Lz_nm / 2, Lz_nm / 2 + dz_nm, dz_nm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    acc = np.zeros_like(centers, dtype=float)
    n = 0
    for ts in u.trajectory:
        mid = Lz_nm / 2.0
        z_nm = ag.positions[:, 2] / 10.0 - mid
        counts, _ = np.histogram(z_nm, bins=edges)
        acc += counts
        n += 1
    return edges, centers, acc / max(n, 1)


def molarity_profile_variable_Lz(u, ag, dz_nm=0.05):
    """Time-averaged molarity profile c(z) in mol/L, allowing Lz to vary per frame."""
    Lz_max_nm = max(ts.dimensions[2] for ts in u.trajectory) / 10.0
    zmax = Lz_max_nm / 2.0
    edges_g = np.arange(-zmax, zmax + dz_nm, dz_nm)
    centers_g = 0.5 * (edges_g[:-1] + edges_g[1:])
    n_g = len(centers_g)

    amount_sum = np.zeros(n_g)
    cover_len = np.zeros(n_g)

    for ts in u.trajectory:
        Lx_nm, Ly_nm, Lz_nm = ts.dimensions[:3] / 10.0
        mid = Lz_nm / 2.0

        edges_f = np.arange(-Lz_nm / 2.0, Lz_nm / 2.0 + dz_nm, dz_nm)
        widths_f = np.diff(edges_f)

        z_nm = ag.positions[:, 2] / 10.0 - mid
        counts, _ = np.histogram(z_nm, bins=edges_f)

        bin_vol_L = (Lx_nm * Ly_nm) * widths_f * 1e-24  # nm^3 -> L
        with np.errstate(divide='ignore', invalid='ignore'):
            c_frame = (counts / NA) / bin_vol_L          # mol/L

        i = j = 0
        while i < len(c_frame) and j < n_g:
            a, b = edges_f[i], edges_f[i + 1]
            A, B = edges_g[j], edges_g[j + 1]
            left, right = max(a, A), min(b, B)
            overlap = right - left
            if overlap > 0:
                amount_sum[j] += c_frame[i] * overlap
                cover_len[j] += overlap
            if b <= B:
                i += 1
            else:
                j += 1

    c_mean = np.zeros_like(centers_g)
    m = cover_len > 0
    c_mean[m] = amount_sum[m] / cover_len[m]
    return edges_g, centers_g, c_mean


def build_stacks(us, molalities, select_expr, per_universe_fn, dz_nm=0.05):
    """Stack (replicates x bins) of a per-universe metric across the given molalities."""
    raw = []
    Lz_max = 0.0

    for mol in molalities:
        mi, _ = mol_keys(mol)
        for u in us[mi]:
            ag = u.select_atoms(select_expr)
            edges_u, centers_u, y_u = per_universe_fn(u, ag, dz_nm)
            raw.append((edges_u, y_u))
            Lz_max = max(Lz_max, edges_u[-1] - edges_u[0])

    zmax = Lz_max / 2.0
    edges_g = np.arange(-zmax, zmax + dz_nm, dz_nm)
    centers_g = 0.5 * (edges_g[:-1] + edges_g[1:])

    stack = np.vstack([rebin_piecewise_constant(eu, yu, edges_g) for eu, yu in raw])
    return centers_g, stack


# =============================================================================
# Plotting primitives
# =============================================================================
def _finish(fig, output_path, dpi):
    fig.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_stack_and_mean(x, stack, color, label, ylabel, molar_line=None,
                        title=None, faint=0.15, output_path=None, dpi=300):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for y in stack:
        ax.plot(x, y, alpha=faint, color=color)
    ax.plot(x, stack.mean(axis=0), lw=2.6, color=color, label=f"{label} (mean)")
    if molar_line is not None:
        ax.axhline(y=molar_line, color='k', lw=2)
    ax.set_xlabel("z (nm)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    _finish(fig, output_path, dpi)


def plot_three_together(x, stacks, colors, labels, ylabel, molar_line=None,
                        title=None, faint=0.12, output_path=None, dpi=300):
    fig, ax = plt.subplots(figsize=(10, 6))
    for stack, c in zip(stacks, colors):
        for y in stack:
            ax.plot(x, y, alpha=faint, color=c)
    for stack, c, lab in zip(stacks, colors, labels):
        ax.plot(x, stack.mean(axis=0), lw=2.6, color=c, label=f"{lab} (mean)")
    if molar_line is not None:
        ax.axhline(y=molar_line, color='k', lw=2)
    ax.set_xlabel("z (nm)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    _finish(fig, output_path, dpi)


# =============================================================================
# Serialization
# =============================================================================
def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    elif isinstance(obj, Quantity):
        return {"value": round(obj.value_in_unit(obj.unit), 4), "unit": str(obj.unit)}
    elif isinstance(obj, float):
        return round(obj, 4)
    else:
        return obj


def format_dict(dictionary):
    """Round float values in a dictionary to 3 decimals."""
    return {x: (round(v, 3) if isinstance(v, float) else v) for x, v in dictionary.items()}


# =============================================================================
# Pipeline orchestration
# =============================================================================
def _pick(name, override, from_salt_data, unit=""):
    """Value to use, and a note when the user's override replaced the salt_data one."""
    if override is None:
        return from_salt_data
    tag = "" if np.isclose(override, from_salt_data) else f"  <- OVERRIDE (salt_data: {from_salt_data}{unit})"
    print(f"  {name} = {override}{unit}{tag}")
    return override


def prepare_context(cfg: FBPConfig) -> FBPContext:
    """Build the derived state (salt data, walls, dirs, unit-converted k) once."""
    SD = load_salt_data(cfg.salt_data_path)
    salt = cfg.ion1 + cfg.ion2
    rows_all = salt_rows(SD, salt)
    if not rows_all:
        raise ValueError(f"No data for {salt} in {cfg.salt_data_path}")
    by_mol = {round(r.molality, 6): r for r in rows_all}

    rows = {}
    for mol in cfg.molalities:
        r = by_mol.get(round(float(mol), 6))
        if r is None:
            raise KeyError(f"{salt} at {mol} mol/kg is not in {Path(cfg.salt_data_path).name}; "
                           f"available: {sorted(by_mol)}")
        mi, _ = mol_keys(mol)
        rows[mi] = {"Molality": r.molality, "Molarity": r.molarity, "Number of Particles": math.ceil(r.num_particles),
                    "Osmotic Coefficient": r.osmotic_coefficient, "k_FBP_wall": r.k_FBP_wall,
                    "delta_z_FBP": r.delta_z_FBP}

    if isinstance(cfg.n_pairs, dict):
        for mol, n in cfg.n_pairs.items():
            mi, mi1 = mol_keys(mol)
            if mi in rows:
                print(f"  n_pairs ({mi1} m) = {int(n)}  <- OVERRIDE (salt_data: {rows[mi]['Number of Particles']})")
                rows[mi]["Number of Particles"] = int(n)
    elif cfg.n_pairs not in (None, "trajectory"):
        raise ValueError("n_pairs must be None, 'trajectory' or a {molality: pairs} dict")

    first = next(iter(rows.values()))
    for v in rows.values():
        if v["k_FBP_wall"] != first["k_FBP_wall"] or v["delta_z_FBP"] != first["delta_z_FBP"]:
            raise ValueError("k_FBP_wall / delta_z_FBP differ between the requested molalities; run them separately.")

    BOX = SD.BOX
    print(f"FBP {salt}: salt_data {Path(cfg.salt_data_path).name}")
    k_nm2 = float(_pick("k", cfg.k, first["k_FBP_wall"], " kJ/mol/nm^2"))
    delta_z = float(_pick("delta_z", cfg.delta_z, first["delta_z_FBP"], " nm"))
    z_center = float(_pick("z_center", cfg.z_center, BOX["z_center"], " nm"))
    Lz_nm = float(_pick("L_z", cfg.L_z, BOX["lz"], " nm"))
    T = float(_pick("T", cfg.T, BOX["temperature"], " K"))
    vant_hoff = int(_pick("vantHoff", cfg.vantHoff, SD.VANT_HOFF[salt]))
    print(f"  k={k_nm2} kJ/mol/nm^2, delta_z={delta_z} nm, z_center={z_center} nm, Lz={Lz_nm} nm, "
          f"T={T} K, i={vant_hoff}")
    print(f"  walls at {z_center - delta_z:.4f} and {z_center + delta_z:.4f} nm")

    ktag = k_tag(k_nm2)
    rdirs = {mi: Path(cfg.results_dir.format(mi=mi, ktag=ktag)) for mi in rows}
    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path("results") / Path(cfg.results_dir).parent.name
    out_dirs = {mi: out_dir / f"{ktag}_{mi}m" for mi in rows}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"  reading {ktag}: " + ", ".join(str(d) for d in rdirs.values()))

    walls = {mi: make_walls(z_center, delta_z) for mi in rows}
    return FBPContext(
        SD=SD, salt=salt, rows=rows, walls=walls, k_nm2=k_nm2,
        k_ang=(k_nm2 * kilojoule_per_mole / nanometer ** 2).in_units_of(kilojoule_per_mole / angstrom ** 2),
        delta_z=delta_z, z_center=z_center, Lz_nm=Lz_nm, Lz_ang=Lz_nm * 10.0, T=T, vant_hoff=vant_hoff,
        area_ref=float(BOX["area"]), out_dir=out_dir, rdirs=rdirs, resname_ions=(cfg.ion1 + " " + cfg.ion2).upper(),
        ktag=ktag, out_dirs=out_dirs)


def load_experimental(cfg: FBPConfig, ctx: FBPContext):
    """Experimental molality (x-axis), osmotic coefficient and osmotic pressure for every molality of the salt."""
    rows = salt_rows(ctx.SD, ctx.salt)
    exp_molalities = [r.molality for r in rows]
    yexpOC = [r.osmotic_coefficient for r in rows]
    yexpOP = []
    for r in rows:
        op = osm_experimental(r.osmotic_coefficient, ctx.vant_hoff, molarity=r.molarity * molar, T=ctx.T * kelvin)
        yexpOP.append(op.value_in_unit(bar))
    return exp_molalities, yexpOC, yexpOP


def resolve_pairs(cfg: FBPConfig, ctx: FBPContext, ions):
    """Compare the restrained ions in the trajectories with the ion-pair count that will be used.

    cfg.n_pairs == "trajectory": use the count found in the trajectory (every replicate must agree).
    Otherwise a disagreement is only a warning, because the molarity (so phi) uses the pair count, not the trajectory."""
    for mol in cfg.molalities:
        mi, mi1 = mol_keys(mol)
        counts = sorted({len(g) for g in ions[mi]})
        if len(counts) != 1:
            raise ValueError(f"{mi1} m: the replicates have different numbers of restrained ions: {counts}")
        n_ions = counts[0]
        pairs_traj, rem = divmod(n_ions, ctx.vant_hoff)
        used = ctx.rows[mi]["Number of Particles"]
        if cfg.n_pairs == "trajectory":
            if rem:
                raise ValueError(f"{mi1} m: {n_ions} restrained ions is not a multiple of the van't Hoff factor "
                                 f"{ctx.vant_hoff}; give n_pairs explicitly")
            tag = "same as salt_data" if pairs_traj == used else f"salt_data: {used}"
            print(f"  n_pairs ({mi1} m) = {pairs_traj}  (counted in the trajectory; {tag})")
            ctx.rows[mi]["Number of Particles"] = pairs_traj
        elif n_ions != ctx.vant_hoff * used:
            print(f"  WARNING {mi1} m: the trajectories hold {n_ions} restrained ions but the pair count in use ({used}) "
                  f"x van't Hoff factor {ctx.vant_hoff} = {ctx.vant_hoff * used}. The molarity, and so phi, uses {used} pairs. "
                  f"If the trajectory is right, set n_pairs=\"trajectory\" (or {{{mol}: {pairs_traj}}}).")


def check_wall_position(ion_coords, repnum, z0_nm, delta_z_nm, k_nm2, T_K, tol_nm=0.01):
    """Warn if the ions' extent does not match the wall position used (a wrong delta_z shifts P by ~15 bar / 0.001 nm).

    Model: ions are uniform between the walls and fall off past each wall as exp(-d^2 / 2 s^2), s = sqrt(kT/k).
    Then the fraction of ions farther than |z-z0| = D is  s*sqrt(pi/2)*erfc((D-delta)/(s*sqrt(2))) / delta,
    so the 99.9th percentile of |z-z0| sits x*s past the wall with erfc(x/sqrt(2)) = 0.001*delta/(s*sqrt(pi/2))
    (x ~ 1.7-1.8 here). Gives delta_est = p99.9 - x*s. Warning only; precision is ~0.003 nm, the tolerance 0.01 nm."""
    from scipy.special import erfcinv
    z = ion_coords.value_in_unit(angstrom)[..., 2] / 10.0
    p999 = float(np.percentile(np.abs(z - z0_nm), 99.9))
    s = math.sqrt(0.0083144626 * T_K / k_nm2)
    x = math.sqrt(2.0) * float(erfcinv(min(0.001 * delta_z_nm / (s * math.sqrt(math.pi / 2.0)), 1.999)))
    est = p999 - x * s
    if abs(est - delta_z_nm) > tol_nm:
        print(f"  WARNING replicate {repnum}: the ions stop at |z-z0| ~ {p999:.3f} nm, which fits walls at delta_z ~ {est:.3f} nm, "
              f"but delta_z = {delta_z_nm:.4f} nm is in use. A wrong wall position shifts the pressure by ~15 bar per 0.001 nm: "
              f"check delta_z (and k) against the run.")
    return est


def check_inputs(cfg: FBPConfig, ctx: FBPContext):
    """Fail early, and say what IS there, if the pdb/xtc of the requested k / molalities / replicates are missing."""
    problems = []
    for mol in cfg.molalities:
        mi, mi1 = mol_keys(mol)
        d = ctx.rdirs[mi]
        missing = [f"md{mi}m_r{i}.{ext}" for i in range(cfg.N_replicates) for ext in ("pdb", "xtc")
                   if not (d / f"md{mi}m_r{i}.{ext}").exists()]
        if not missing:
            continue
        msg = f"{mi1} m, {ctx.ktag}: {len(missing)} of {2 * cfg.N_replicates} files missing in {d}"
        if d.is_dir():
            msg += f" (e.g. {', '.join(missing[:3])})"
        else:
            # which k's DO have this molality? (folder names are <anything>_k<k>_<mi>m)
            tags = sorted({m.group(1) for p in d.parent.glob(f"*_k*_{mi}m")
                           if (m := re.search(r"_(k[^_]+)_" + re.escape(mi) + "m$", p.name))})
            msg += f"; folder not found. k folders present for {mi1} m: {', '.join(tags) if tags else 'none'}"
        problems.append(msg)
    if problems:
        raise FileNotFoundError("Cannot read the requested results:\n  " + "\n  ".join(problems))


def load_trajectories(cfg: FBPConfig, ctx: FBPContext):
    """Open universes and ion selections for every molality/replicate.

    Trajectories are left as on-disk readers -- NOT pulled into RAM. Every downstream analysis streams
    frame-by-frame, so peak memory is ~one frame. Tradeoff: each trajectory is re-read from disk on every
    analysis pass. (MDAnalysis writes a hidden .*_offsets.npz frame index next to each XTC.)
    """
    check_inputs(cfg, ctx)
    us, ions = {}, {}
    for mol in cfg.molalities:
        mi, _ = mol_keys(mol)
        us[mi], ions[mi] = [], []
        for i in range(cfg.N_replicates):
            postfix = f'{mi}m_r{i}'
            pdb_path = ctx.rdirs[mi] / f'md{postfix}.pdb'
            xtc_path = ctx.rdirs[mi] / f'md{postfix}.xtc'
            print(f'Opening trajectory for {postfix}')
            u = mda.Universe(str(pdb_path), str(xtc_path))
            us[mi].append(u)

            if cfg.center_atom1 != cfg.center_atom2:
                ion_group = u.select_atoms(f'element {cfg.center_atom1}', f'element {cfg.center_atom2}')
            else:
                ion_group = u.select_atoms(f'element {cfg.center_atom1}')
            ions[mi].append(ion_group)
    return us, ions


def plot_ion_count_per_replicate(cfg: FBPConfig, ctx: FBPContext, us, ions):
    """Per-replicate time-averaged ion-PAIR count per z bin (all ions pooled). For concentrations see concprof_*."""
    for mol in cfg.molalities:
        mi, mi1 = mol_keys(mol)
        N_i = ctx.rows[mi]['Number of Particles']

        fig = plt.figure(figsize=(8, 5))
        for i in range(cfg.N_replicates):
            zvals_i = get_ion_array(us[mi][i], ions[mi][i])
            counts, bins = np.histogram(zvals_i, bins=int(ctx.Lz_ang / 2),
                                        weights=np.full_like(zvals_i, N_i / zvals_i.size))
            plt.plot(0.5 * (bins[1:] + bins[:-1]), counts, label=f'Replicate {i + 1}')
        plt.title(f'FBP {ctx.salt} + {cfg.water} at {mi1} m - ion pairs per bin, all replicates')
        plt.xlabel('Z Coordinate (nm)')
        plt.ylabel('Ion pairs per bin (time-averaged)')
        plt.xlim(0, ctx.Lz_nm)
        plt.legend(fontsize=6, ncol=2)
        _finish(fig, str(ctx.out_dirs[mi] / f'conc_prof_allreps_{mi}m_{ctx.salt}.png'), 150)


def plot_count_and_molarity_profiles(cfg: FBPConfig, ctx: FBPContext, us):
    """Stacked count (ions/bin) and molarity (mol/L) profiles across replicates, one set per molality."""
    dz_nm = cfg.dz_nm
    c1_col, c2_col, salt_col = 'yellowgreen', 'c', 'tab:blue'

    # When the two centre atoms are different elements, element alone separates them. When they share an
    # element (NH4NO3 -> both N) disambiguate by residue so each species is counted once.
    if cfg.center_atom1 != cfg.center_atom2:
        sel1, sel2 = f'element {cfg.center_atom1}', f'element {cfg.center_atom2}'
    else:
        sel1 = f'resname {cfg.ion1} and element {cfg.center_atom1}'
        sel2 = f'resname {cfg.ion2} and element {cfg.center_atom2}'

    _probe = next(iter(us[mol_keys(cfg.molalities[0])[0]]))
    for _s in (sel1, sel2):
        if _probe.select_atoms(_s).n_atoms == 0:
            raise ValueError(f"Selection '{_s}' matched 0 atoms. Check that the topology's residue names "
                             f"match cfg.ion1/cfg.ion2 ('{cfg.ion1}'/'{cfg.ion2}').")

    for mol in cfg.molalities:
        mi, mi1 = mol_keys(mol)
        out = ctx.out_dirs[mi]
        molar_line = to_float_M(ctx.rows[mi]['Molarity'])

        cent_c1, cnt1 = build_stacks(us, [mol], sel1, count_profile_fixed_Lz, dz_nm)
        _, cnt2 = build_stacks(us, [mol], sel2, count_profile_fixed_Lz, dz_nm)
        cnt_salt = 0.5 * (cnt1 + cnt2)
        cent_M1, M1 = build_stacks(us, [mol], sel1, molarity_profile_variable_Lz, dz_nm)
        _, M2 = build_stacks(us, [mol], sel2, molarity_profile_variable_Lz, dz_nm)
        M_salt = 0.5 * (M1 + M2)

        n1, n2, ns = cfg.ion1, cfg.ion2, f"{cfg.ion1}{cfg.ion2}"
        plot_stack_and_mean(cent_c1, cnt1, c1_col, n1, "Average ions per bin",
                            title=f"FBP {mi1} m - {n1} count profile", output_path=str(out / f'countsprof_{mi}m_{n1}.png'))
        plot_stack_and_mean(cent_c1, cnt2, c2_col, n2, "Average ions per bin",
                            title=f"FBP {mi1} m - {n2} count profile", output_path=str(out / f'countsprof_{mi}m_{n2}.png'))
        plot_stack_and_mean(cent_c1, cnt_salt, salt_col, f"{n1}+{n2}", "Average ions per bin",
                            title=f"FBP {mi1} m - total ion count profile ({n1}+{n2})",
                            output_path=str(out / f'countsprof_{mi}m_{ns}.png'))
        plot_three_together(cent_c1, [cnt1, cnt2, cnt_salt], [c1_col, c2_col, salt_col], [n1, n2, f"{n1}+{n2}"],
                            "Average ions per bin", title=f"FBP {mi1} m - count profiles for {ns}",
                            output_path=str(out / f'countsprof_{mi}m_ALL.png'))

        plot_stack_and_mean(cent_M1, M1, c1_col, n1, "Concentration (mol/L)", molar_line=molar_line,
                            title=f"FBP {mi1} m - {n1} concentration profile", output_path=str(out / f'concprof_{mi}m_{n1}.png'))
        plot_stack_and_mean(cent_M1, M2, c2_col, n2, "Concentration (mol/L)", molar_line=molar_line,
                            title=f"FBP {mi1} m - {n2} concentration profile", output_path=str(out / f'concprof_{mi}m_{n2}.png'))
        plot_stack_and_mean(cent_M1, M_salt, salt_col, ns, "Concentration (mol/L)", molar_line=molar_line,
                            title=f"FBP {mi1} m - {ns} concentration profile", output_path=str(out / f'concprof_{mi}m_{ns}.png'))
        plot_three_together(cent_M1, [M1, M2, M_salt], [c1_col, c2_col, salt_col], [n1, n2, ns],
                            "Concentration (mol/L)", molar_line=molar_line,
                            title=f"FBP {mi1} m - concentration profiles for {ns}", output_path=str(out / f'concprof_{mi}m_ALL.png'))


def compute_and_save_results(cfg: FBPConfig, ctx: FBPContext, us, ions):
    """Per-replicate osmotic values, replicate averages, and JSON output."""
    results, replicate_results = {}, {}
    for mol in cfg.molalities:
        mi, mi1 = mol_keys(mol)
        key = f"{mi}m"
        results[key] = []
        for i, (ui, ionsi) in enumerate(zip(us[mi], ions[mi])):
            print(i)
            results[key].append(run_analysis(
                ui, ionsi, ctx.rows[mi], repnum=i, walls=ctx.walls[mi], K=ctx.k_ang,
                vant_hoff=ctx.vant_hoff, T=ctx.T * kelvin, nskip=cfg.nskip, area_ref=ctx.area_ref,
                wall_check=dict(z0_nm=ctx.z_center, delta_z_nm=ctx.delta_z, k_nm2=ctx.k_nm2, T_K=ctx.T)))
        replicate_results[key] = replicate_analysis(results[key], mi1)

    for key, data in results.items():
        with open(ctx.out_dirs[key[:-1]] / f"results_{key}.json", "w") as f:
            json.dump(serialize(data), f, indent=4)
    for key, data in replicate_results.items():
        with open(ctx.out_dirs[key[:-1]] / f"final_results_{key}.json", "w") as f:
            json.dump(format_dict(data), f, indent=4)
    return results, replicate_results


def _plot_ocs_ops(cfg: FBPConfig, ctx: FBPContext, mols, exp_molalities, yexpOC, yexpOP, ocs_path, ops_path):
    """Osmotic coefficient and osmotic pressure vs MOLALITY (mol/kg) for the molalities `mols`."""
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 18, 20
    fbp_results = load_results(mols, ctx.out_dirs)
    k_title = f"{ctx.k_nm2:g} kJ/mol/nm$^2$"

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(9.0, 10.0))
    plt.plot(exp_molalities, yexpOC, marker='o', linestyle='-', color='black',
             label='Experimental (Hamer & Wu)', linewidth=3, markersize=8)
    plt.errorbar(mols, fbp_results['OC_Values'], yerr=fbp_results['OC_Errors'], fmt='o', linestyle='-',
                 color='magenta', capsize=8, label='Flat-bottom Potentials - OpenFF', linewidth=3, markersize=8)
    plt.xlim(0, max(mols) + 0.3)
    plt.xlabel("Concentration (mol/kg)", fontsize=MEDIUM_SIZE)
    plt.ylabel("Osmotic Coefficients", fontsize=MEDIUM_SIZE)
    plt.title(f'FBP - Osmotic Coefficients {ctx.salt} + {cfg.water}, k={k_title}', fontsize=BIGGER_SIZE)
    plt.legend(prop={'size': 16}, loc='upper left')
    plt.grid()
    _finish(fig, str(ocs_path), 150)

    fig = plt.figure(figsize=(10, 9))
    plt.plot(exp_molalities, yexpOP, marker='o', linestyle='-', color='black',
             label='Experimental (Hamer & Wu)', linewidth=3, markersize=8)
    plt.errorbar(mols, fbp_results['OP_Values'], yerr=fbp_results['OP_Errors'], fmt='o', linestyle='-',
                 color='magenta', capsize=8, label='Flat-bottom Potentials - OpenFF', linewidth=3, markersize=8)
    plt.xlabel("Concentration (mol/kg)", fontsize=MEDIUM_SIZE)
    plt.ylabel("Osmotic Pressure (bar)", fontsize=MEDIUM_SIZE)
    plt.title(f'FBP - Osmotic Pressure {ctx.salt} + {cfg.water}, k={k_title}', fontsize=BIGGER_SIZE)
    plt.legend()
    plt.grid()
    _finish(fig, str(ops_path), 150)


def plot_results(cfg: FBPConfig, ctx: FBPContext, exp_molalities, yexpOC, yexpOP):
    """ocs.png / ops.png in each <ktag>_<mi>m folder (that molality vs experiment); when several molalities were
    run, also ocs_<ktag>_<mi>m_<mi>m.png / ops_... with all of them in the base results folder."""
    for mol in cfg.molalities:
        mi, _ = mol_keys(mol)
        _plot_ocs_ops(cfg, ctx, [mol], exp_molalities, yexpOC, yexpOP,
                      ctx.out_dirs[mi] / "ocs.png", ctx.out_dirs[mi] / "ops.png")
    if len(cfg.molalities) > 1:
        tag = "_".join([ctx.ktag] + [f"{mol_keys(m)[0]}m" for m in cfg.molalities])
        base = next(iter(ctx.out_dirs.values())).parent
        _plot_ocs_ops(cfg, ctx, list(cfg.molalities), exp_molalities, yexpOC, yexpOP,
                      base / f"ocs_{tag}.png", base / f"ops_{tag}.png")


def check_pymbar():
    """The equilibration detection needs pymbar. It is imported lazily (so the module loads without it), which would
    otherwise fail only after the slow profile plots: check up front."""
    try:
        import pymbar  # noqa: F401
    except ImportError as e:
        raise ImportError("pymbar is not installed in this environment (python: %s).\n"
                          "  Use the environment that has it:  conda activate osmotic-analysis\n"
                          "  (osmotic-analysis-hp does not have pymbar)." % sys.executable) from e


def run(cfg: FBPConfig):
    """End-to-end pipeline for one configuration. Returns (results, replicate_results)."""
    check_pymbar()
    ctx = prepare_context(cfg)
    exp_molalities, yexpOC, yexpOP = load_experimental(cfg, ctx)
    us, ions = load_trajectories(cfg, ctx)
    resolve_pairs(cfg, ctx, ions)

    if cfg.profiles:
        plot_ion_count_per_replicate(cfg, ctx, us, ions)
        plot_count_and_molarity_profiles(cfg, ctx, us)
    out = compute_and_save_results(cfg, ctx, us, ions)
    plot_results(cfg, ctx, exp_molalities, yexpOC, yexpOP)
    print("Results in " + ", ".join(str(d) for d in ctx.out_dirs.values()))
    return out
