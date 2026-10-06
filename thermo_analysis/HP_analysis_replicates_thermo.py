"""
HP Analysis Replicates
======================
Code written by Barbara Morales and Dr. Michael Shirts to analyze harmonic potential
method calculation and simulation results of osmotic coefficients.

Converted from HP_analysis_replicates.ipynb to a standalone Python script.

Key changes vs. notebook:
  - All figures are saved to the result directory rather than displayed interactively.
  - Concentration axes that were in molar (mol/L) are now expressed in molal (mol/kg
    of water), using the number of water molecules counted directly from each trajectory.
  - A usage script is provided separately (run_HP_analysis.py).

HP_analysis_replicates_thermo.py (this copy) - changes vs HP_analysis_replicates.py
  This copy is system-independent: nothing in it is tied to one salt, box, water model or paper.
  1. A: the nm^-3 conversion `A /= sqrt(1e24/NA)` is removed (every concentration in the fits is mol/L).
     A comes from `A_molar` if given, else from the water model's `dielectric_constant` and T
     (1:1 salts); vantHoff == 3 keeps the original hard-coded 2.50. `legacy=True` restores the old A.
  2. Molality axis: profiles are converted with the LOCAL water density measured from the trajectories
     (make_local_molal) instead of one box-average factor. `legacy=True` restores the old conversion.
  3. Results table: target molalities beyond the curve are blank/NaN instead of repeating the last point.
  4. Checks (box area vs salt_data, reservoir water density): `checks="ignore"` (default) | "warn" | "strict".
     The reservoir density is ALWAYS measured and stamped on the saved summary plots (and returned).
     `reservoir_density_target` (g/cm3, YOUR water model at your T, P) is only used for warn/strict.
  5. M_WATER_G_PER_MOL 18.015 -> 18.01528 (matches salt_data).
  6. New: density (eq 22/23), V_w (eq 24), molal description, ln gamma+-, water and ion chemical
     potentials - thermo_from_profiles(); `thermo_extras=True`. Optional `reference=` draws literature curves.

Reference:
  Ashbaugh & Hosseini, JCTC 2024, https://pubs.acs.org/doi/10.1021/acs.jctc.3c00982
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import logging
logging.basicConfig(level=logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

import importlib.util
from pathlib import Path
import numpy as np
import json
import csv

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – write PNGs to disk
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, minimize, curve_fit
from scipy.integrate import simpson

from openmm.unit import (
    bar, mole, litre, kelvin, kilojoule_per_mole, nanometer,
)

import MDAnalysis as mda


# ===========================================================================
# PHYSICAL CONSTANTS
# ===========================================================================
NA  = 6.02214076e23        # Avogadro [mol⁻¹]
R   = 8.31446261815324     # Ideal gas constant [J/(mol·K)]

# Molar masses (g/mol) – used for molar→molal conversion
M_WATER_G_PER_MOL = 18.01528  # water (H₂O); was 18.015

# ---------------------------------------------------------------------------
# MODULE HELPERS
# ---------------------------------------------------------------------------

def load_salt_data(path):
    """Import a per-box salt_data_*.py by path."""
    spec = importlib.util.spec_from_file_location("salt_data_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _format_molality(mol):
    """3.5 -> ('35', '3.5');  2.0 -> ('2', '2.0')."""
    mi1 = f"{mol:.1f}"
    return (str(int(mol)) if mol % 1 == 0 else mi1.replace(".", "")), mi1


def _k_to_tag(k):
    """Same tag the dispatch uses: 1.5567 -> 'k1p56', 4184.0 -> 'k4184'."""
    if isinstance(k, int) or (isinstance(k, float) and float(k).is_integer()):
        return f"k{int(k)}"
    return "k" + f"{k:.3g}".replace(".", "p").replace("-", "m")


def select_ion(u, resname, element):
    """Ion centre atoms, optionally filtered by residue name.

    Residue names matter when both centres share an element (NH4NO3 -> N).
    """
    if resname is not None:
        return u.select_atoms(f"resname {resname} and element {element}")
    return u.select_atoms(f"element {element}")



def convert_profile_to_molal(c_molar_profile, N_water_mean, V_box_L):
    """Convert a molar concentration profile to molal.

    Parameters
    ----------
    c_molar_profile : np.ndarray
    N_water_mean    : float – mean number of TIP3P water oxygens across replicates
    V_box_L         : float – mean box volume in litres

    Returns
    -------
    np.ndarray  – molal concentration profile
    """
    c_molar_profile = np.asarray(c_molar_profile, dtype=float)
    m_water_kg = N_water_mean * M_WATER_G_PER_MOL * 1e-3 / NA
    factor = m_water_kg / V_box_L        # kg_water / L_solution
    return c_molar_profile / factor      # mol/L  /  (kg/L)  = mol/kg


# ===========================================================================
# HELPER FUNCTIONS (unchanged from notebook)
# ===========================================================================

def load_salt_info(SD, ion1, ion2):
    """Experimental reference rows for one salt, from a loaded salt_data module."""
    salt = ion1 + ion2
    entries = [SD.SaltData(**e) if isinstance(e, dict) else e for e in SD.salt_infos]
    filtered_entries = {
        f"Molality {e.molality} mol/kg": {
            "Molality": e.molality,
            "Molarity": round(e.molarity, 4),        # derived property
            "Number of Particles": e.num_particles,
            "Osmotic Coefficient": e.osmotic_coefficient,
            "Density": e.density,
        }
        for e in entries if e.salt == salt
    }
    if filtered_entries:
        return {salt: filtered_entries}
    else:
        return {"Error": f"No data found for {salt}"}


def extract_experiment_data(data, salt):
    """Return lists of molality values (mol/kg) and osmotic coefficients.

    Molality is read directly from the key name ('Molality X mol/kg') to
    avoid any molar/molal ambiguity.  The 'Molarity' field in each entry is
    NOT used here.
    """
    molality_list = []
    osmotic_coeff_list = []
    if salt in data:
        for molality_key, values in data[salt].items():
            # Key format: "Molality X mol/kg"
            molality_val = float(molality_key.split()[1])
            molality_list.append(molality_val)
            osmotic_coeff_list.append(values["Osmotic Coefficient"])
    else:
        print(f"Salt '{salt}' not found in data.")
    return molality_list, osmotic_coeff_list


def osm_experimental(exp_osm_coeff, vant_hoff, molality, T,
                     rho_water_kg_per_L=0.99705):
    """Compute experimental osmotic pressure from molality.

    Parameters
    ----------
    exp_osm_coeff      : float  -- experimental osmotic coefficient
    vant_hoff          : int    -- van't Hoff factor
    molality           : float  -- concentration in mol/kg (NOT mol/L)
    T                  : Quantity -- temperature
    rho_water_kg_per_L : float  -- density of pure water in kg/L (default 0.9965
                                   at ~300 K / 27 degC)

    Notes
    -----
    Osmotic pressure = phi * nu * c_molar * R * T
    where c_molar [mol/L] = molality [mol/kg] * rho_water [kg/L].
    """
    R_bar = (R / 100) * bar * litre / (mole * kelvin)   # 1 L*bar = 100 J
    # Convert molality -> molarity using water density at ~300 K
    c_molar = molality * rho_water_kg_per_L   # mol/kg * kg/L = mol/L
    c_molar_q = c_molar * (mole / litre)
    osm_press_ideal = vant_hoff * c_molar_q * R_bar * T
    osm_press_ideal = osm_press_ideal.in_units_of(bar)
    osm_bar = exp_osm_coeff * osm_press_ideal
    return osm_bar


def osmotic_pressure(cs, params, nu, nterms, T):
    """Calculate osmotic pressure and osmotic coefficient from fitted parameters.

    Parameters
    ----------
    cs     : array-like  – concentration profile (nm⁻³ internally)
    params : array-like  – [B, alpha1, ...] fit parameters
    nu     : int         – van't Hoff factor
    nterms : int         – number of alpha expansion terms
    T      : float       – temperature in K (default 300)
    R      : float       – gas constant in J/(mol·K) (default 8.31446261815324)
    """
    csh = np.sqrt(cs)
    B = params[0]
    Bf = 1 + B * csh
    terms = cs + (A / B ** 2) * ((2 / B) * np.log(Bf) - (2 * csh + B * cs) / Bf)
    for i in range(1, nterms + 1):
        terms += params[i] * (i / (i + 1)) * cs ** (i + 1)
    p = nu * R * T * terms
    op = p / 100
    ip = nu * R * T * cs
    return op, p / ip


def _blank_nan(x):
    return "" if x != x else x


def find_closest(lst, target):
    closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
    closest_value = lst[closest_index]
    return closest_index, closest_value


def format_dict(obj):
    if isinstance(obj, dict):
        return {k: format_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_dict(v) for v in obj]
    elif isinstance(obj, float):
        return None if obj != obj else round(obj, 3)      # NaN -> null
    else:
        return obj


def get_ion_array(u, ions, distance=None):
    """Folded |z - z0| for every ion, every frame, in nm.

    z0 defaults to the box centre read from the trajectory. The old hard-coded
    72 A folded a 100 A box about the wrong point - silently.
    """
    if distance is None:
        distance = u.trajectory.ts.dimensions[2] / 2.0      # A, box centre
    ion_array = []
    for ts in u.trajectory:
        ions_z = ions.positions[:, 2]
        ion_array.append(ions_z)
    ionz = np.abs(np.array(ion_array) - distance) / 10
    zvals = ionz.flatten()
    return zvals


def count_profile_z_fixed_Lz(u, ag, dz_nm):
    Lz_nm = u.trajectory.ts.dimensions[2] / 10.0
    nbins = max(1, int(round((Lz_nm / 2) / dz_nm)))
    edges = np.linspace(0, Lz_nm / 2, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    counts_accum = np.zeros_like(centers, dtype=float)
    n_frames = 0
    for ts in u.trajectory:
        mid = Lz_nm / 2.0
        z_nm = np.abs(ag.positions[:, 2] / 10.0 - mid)
        counts, _ = np.histogram(z_nm, bins=edges)
        counts_accum += counts
        n_frames += 1
    avg_counts = counts_accum / max(n_frames, 1)
    return centers, widths, avg_counts


def get_average_cross_section_area(u):
    LxLy = 0
    n_frames = 0
    for ts in u.trajectory:
        Lx_nm, Ly_nm, _ = ts.dimensions[:3] / 10.0
        LxLy += Lx_nm * Ly_nm
        n_frames += 1
    LxLy /= n_frames
    return LxLy


def molarity_profile_fixed_Lz(u, ag, dz_nm):
    Lz_nm = u.trajectory.ts.dimensions[2] / 10.0
    nbins = max(1, int(round((Lz_nm / 2) / dz_nm)))
    edges = np.linspace(0, Lz_nm / 2, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    c_accum = np.zeros_like(centers, dtype=float)
    n_frames = 0
    for ts in u.trajectory:
        Lx_nm, Ly_nm, _ = ts.dimensions[:3] / 10.0
        mid = Lz_nm / 2.0
        z_nm = np.abs(ag.positions[:, 2] / 10.0 - mid)
        counts, _ = np.histogram(z_nm, bins=edges)
        bin_vol_L = (Lx_nm * Ly_nm) * widths * 1e-24
        with np.errstate(divide="ignore", invalid="ignore"):
            c_frame = (counts / (2 * NA)) / bin_vol_L
            c_frame[~np.isfinite(c_frame)] = 0.0
        c_accum += c_frame
        n_frames += 1
    c_mean = c_accum / max(n_frames, 1)
    return centers, c_mean


def hist_to_molar(conc_density, n_pairs, area_nm2):
    """Folded density=True histogram -> salt concentration in mol/L.

    conc_density : np.histogram(|z - z0|, bins=edges, density=True) output.
                   Units nm^-1; integrates to 1 over the folded half-box, so
                   the bin width is already divided out - no dz term here.
    n_pairs      : ion pairs in the box (N_s)
    area_nm2     : mean Lx*Ly

    A folded bin represents TWO slabs of area `area_nm2`, and one pair is two
    ions, hence the single factor of 2. The 1e24/NA converts nm^-3 to mol/L -
    it MULTIPLIES; dividing understates the profile by 1/1.66^2.
    """
    return conc_density * n_pairs / (2 * area_nm2) * (1e24 / NA)


def fold_profile(z, y):
    """Stub – data should already be folded."""
    return z, y


# ===========================================================================
# SOLUTION DENSITY  (Hosseini & Ashbaugh, JCTC 2023, 19, 8826 - eqs 22-24)
# ===========================================================================
#
# The idea, in one sentence: an HP box already contains every concentration
# between ~0 (reservoir) and C(0) (centre), so one simulation yields the whole
# rho(m) curve - you just have to histogram the WATER as well as the ions.
#
#   rho(z) = [C_w(z) M_w + C_s(z) M_s] / 1000        g/cm^3
#   m(z)   = C_s(z) / (C_w(z) M_w / 1000)            mol salt / kg water
#
# with C in mol/L. Both come straight out of molarity_profile_fixed_Lz.
#
# The paper then fits rho against m with (eq 22)
#
#   rho(m) = rho_w0 + theta_1 m + theta_3/2 m^(3/2) + theta_2 m^2
#
# The m^(3/2) term is the Debye-Huckel limiting-law signature in the apparent
# molar volume; this is a Sohnel-Novotny style expansion. Eq 22 is LINEAR in
# the four coefficients, so the fit is plain least squares - no curve_fit, no
# initial guess, no convergence to babysit.
#
# An osmotic-force-balance box is at constant mu_w, not constant P: the salty
# middle sits at P + Pi. Eq 23 decompresses to 1 bar,
#
#   rho(m|P) = rho(m|mu_w) * (1 - kappa_eff * Pi)
#
# with kappa_eff = 3.95e-5 /bar (fitted in the paper; ~20% below TIP4P/2005's
# own compressibility, 4.65e-5 /bar). At 3.5 m NaCl, Pi ~ 200 bar, so this is
# a 0.8% correction - small, but it is the entire difference between the two
# coefficient sets in Table 3.
# ---------------------------------------------------------------------------

KAPPA_EFF_PER_BAR = 3.95e-5     # effective solution compressibility, eq 23

# Table 3: fits of eq 22 to the paper's SIMULATED densities, 25 C.
#   (rho_w0, theta_1, theta_3/2, theta_2) at constant mu_w and at constant P.
#
# WARNING - these are NOT experimental densities, and they are NOT a target for
# an unmodified Joung-Cheatham run. They come from TIP4P/2005 water with the
# cation-anion Lennard-Jones cross terms OPTIMIZED (chi != 0) against osmotic
# pressure. The chi = 0 fits live in Table S4 of the SI. Use these to see the
# shape and the scale, not to grade your own numbers.
HOSSEINI_TABLE3 = {
    #        rho_w0    theta_1      theta_3/2     theta_2
    "LiCl": {"mu": (0.99702, 1.2346e-2, -2.3566e-3,  2.7688e-4),
             "P":  (0.99702, 1.0546e-2, -2.0884e-3, -7.0290e-5)},
    "LiBr": {"mu": (0.99712, 3.9357e-2,  1.0385e-3, -1.6204e-3),
             "P":  (0.99712, 3.7330e-2,  1.6557e-3, -2.1963e-3)},
    "LiI":  {"mu": (0.99713, 6.3490e-2, -7.1690e-4, -2.2102e-3),
             "P":  (0.99713, 6.1383e-2, -1.2910e-5, -2.8867e-3)},
    "NaCl": {"mu": (0.99700, 6.2948e-2, -7.1889e-3, -4.1790e-5),
             "P":  (0.99700, 6.0936e-2, -6.5261e-3, -5.7990e-4)},
    "NaBr": {"mu": (0.99695, 9.9941e-2, -5.4726e-3, -1.0888e-3),
             "P":  (0.99695, 9.7856e-2, -4.5989e-3, -1.8555e-3)},
    "NaI":  {"mu": (0.99700, 1.3606e-1, -1.4879e-3, -3.1384e-3),
             "P":  (0.99700, 1.3343e-1,  2.3562e-4, -4.3985e-3)},
    "KCl":  {"mu": (0.99713, 4.9510e-2, -5.3635e-3, -6.2850e-4),
             "P":  (0.99713, 4.7929e-2, -5.4872e-3, -7.0140e-4)},
    "KBr":  {"mu": (0.99680, 9.9842e-2, -5.3608e-3, -1.8468e-3),
             "P":  (0.99680, 9.8248e-2, -5.3708e-3, -2.1235e-3)},
    "KI":   {"mu": (0.99698, 1.1226e-1, -6.1732e-3, -2.8402e-3),
             "P":  (0.99698, 1.1113e-1, -7.3155e-3, -2.6182e-3)},
    "RbCl": {"mu": (0.99676, 8.5821e-2, -7.0817e-3, -1.5060e-3),
             "P":  (0.99676, 8.4221e-2, -7.2535e-3, -1.5783e-3)},
    "RbBr": {"mu": (0.99716, 1.2109e-1, -9.4249e-3, -2.2284e-3),
             "P":  (0.99716, 1.1975e-1, -1.0040e-2, -2.1754e-3)},
    "RbI":  {"mu": (0.99686, 1.5398e-1, -1.4705e-2, -2.3955e-3),
             "P":  (0.99686, 1.5281e-1, -1.5714e-2, -2.1964e-3)},
    "CsCl": {"mu": (0.99736, 1.2948e-1, -8.5014e-3, -2.8228e-3),
             "P":  (0.99736, 1.2785e-1, -8.5480e-3, -2.9858e-3)},
    "CsBr": {"mu": (0.99734, 1.6846e-1, -1.3048e-2, -3.1837e-3),
             "P":  (0.99734, 1.6700e-1, -1.3468e-2, -3.2468e-3)},
    "CsI":  {"mu": (0.99679, 2.2188e-1, -6.8022e-3, -5.4566e-3),
             "P":  (0.99679, 2.2035e-1, -7.1660e-3, -5.6528e-3)},
}


def density_eq22(molality, coeffs):
    """Eq 22: rho(m) = rho_w0 + t1 m + t32 m^(3/2) + t2 m^2.

    molality : scalar or array, mol/kg
    coeffs   : (rho_w0, theta_1, theta_3/2, theta_2)

    Returns density in g/cm^3.
    """
    m = np.asarray(molality, dtype=float)
    rho_w0, t1, t32, t2 = coeffs
    return rho_w0 + t1 * m + t32 * m ** 1.5 + t2 * m ** 2


def fit_density_eq22(molality, density, weights=None, fix_rho_w0=None):
    """Least-squares fit of eq 22. Linear in the coefficients, so exact.

    molality   : array, mol/kg  (from density_molality_profiles)
    density    : array, g/cm^3
    weights    : optional per-point weights. Bins near the box edge hold few
                 ions and are noisy in m; 1/sigma or the bin ion count works.
    fix_rho_w0 : pin the intercept to a measured pure-water density instead of
                 fitting it. Recommended - your reservoir gives rho_w0 directly
                 and letting the fit chase it wastes a degree of freedom on the
                 one number you already know exactly.

    Returns (rho_w0, theta_1, theta_3/2, theta_2).
    """
    m = np.asarray(molality, dtype=float)
    y = np.asarray(density, dtype=float)
    good = np.isfinite(m) & np.isfinite(y) & (m >= 0)
    m, y = m[good], y[good]
    if m.size < 4:
        raise ValueError(f"need >=4 usable points to fit eq 22, got {m.size}")

    cols = [m, m ** 1.5, m ** 2]
    if fix_rho_w0 is None:
        design = np.column_stack([np.ones_like(m)] + cols)
    else:
        design = np.column_stack(cols)
        y = y - fix_rho_w0

    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float)[good])
        design, y = design * w[:, None], y * w

    sol, *_ = np.linalg.lstsq(design, y, rcond=None)
    return tuple(sol) if fix_rho_w0 is None else (fix_rho_w0, *sol)


def density_const_P(rho_mu, osmotic_pressure_bar, kappa_eff=KAPPA_EFF_PER_BAR):
    """Eq 23: decompress a constant-mu_w density to 1 bar.

    rho_mu               : density measured in the osmotic box, g/cm^3
    osmotic_pressure_bar : Pi at the same molality, bar
    """
    return np.asarray(rho_mu, float) * (
        1.0 - kappa_eff * np.asarray(osmotic_pressure_bar, float))


def partial_molar_volume_water(molality, coeffs, molar_mass_salt):
    """Eq 24: V_w_bar(m) in cm^3/mol, from an eq-22 fit.

    Derived from V(m) = (1000 + m M_s)/rho for a 1 kg water basis, with
    V_s_bar = dV/dm and V_w_bar = (V - m dV/dm) * M_w/1000:

        V_w_bar = M_w [ 1/rho + m (1000 + m M_s) rho' / (1000 rho^2) ]

    Sanity: at m -> 0 this is M_w/rho_w0 = 18.07 cm^3/mol, and for the paper's
    NaCl fit at 3.5 m it gives 17.70 - a 2.1% drop, matching their statement
    that V_w_bar varies by less than ~3% over the range studied.

    NOTE: I derived this from eq 22 rather than transcribing eq 24, which is a
    figure in the HTML. The limits above check out, but verify against the PDF
    before you put it in a paper.
    """
    m = np.asarray(molality, dtype=float)
    _, t1, t32, t2 = coeffs
    rho = density_eq22(m, coeffs)
    drho = t1 + 1.5 * t32 * np.sqrt(m) + 2.0 * t2 * m
    return M_WATER_G_PER_MOL * (
        1.0 / rho + m * (1000.0 + m * molar_mass_salt) * drho
        / (1000.0 * rho ** 2))


def density_molality_profiles(u, ag_water_O, ag_ion1, ag_ion2, dz_nm,
                              molar_mass_salt):
    """Folded rho(z) and m(z) from one trajectory - the measurement half.

    ag_water_O : water OXYGENS only, one per molecule. With a 4-site model be
                 careful that the M virtual site is excluded; prefer
                 `u.select_atoms("resname HOH and element O")` over a name
                 glob, and check n_atoms against BOX['n_water'].
    ag_ion1/2  : the two ion centres (select_ion output).

    Returns (z_nm, rho_g_cm3, molality, c_salt_M, c_water_M), all on the same
    folded half-box grid as every other profile in this module.

    Why this also matters for the molality AXIS: convert_profile_to_molal()
    divides the whole profile by ONE box-averaged kg-water-per-litre. In an HP
    box that factor is genuinely a function of z - the centre is 3.5 m salt and
    the ends are pure water - so the box average is wrong at both ends and only
    right somewhere in between. m(z) returned here is the local, correct one.
    """
    z, c_w = molarity_profile_fixed_Lz(u, ag_water_O, dz_nm)
    _, c_1 = molarity_profile_fixed_Lz(u, ag_ion1, dz_nm)
    _, c_2 = molarity_profile_fixed_Lz(u, ag_ion2, dz_nm)
    c_s = 0.5 * (c_1 + c_2)                      # salt formula units, mol/L

    # g/L -> g/cm^3 is the /1000; mol/L * g/mol = g/L.
    rho = (c_w * M_WATER_G_PER_MOL + c_s * molar_mass_salt) / 1000.0

    kg_water_per_L = c_w * M_WATER_G_PER_MOL / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        molality = np.where(kg_water_per_L > 1e-6, c_s / kg_water_per_L, np.nan)

    return z, rho, molality, c_s, c_w


# ===========================================================================
# THERMODYNAMIC EXTRAS  (density, V_w, molal description, ln gamma, chemical potentials)
# ===========================================================================
# Notation: molar set (A, B, a1, a2) in M^-1/2, M^-1/2, M^-1, M^-2 at constant mu_w;
# molal set (A~, B~, a~1, a~2) in m^-1/2 ... m^-2 at constant P;  A~ = A sqrt(rho_w0), B~ = B sqrt(rho_w0).
# B is not re-scanned: it is the B each method already fitted.
THERMO_COLORS = {"unweighted": "tab:blue", "weighted": "tab:orange", "maximum_likelihood": "tab:green"}
THERMO_LABELS = {"unweighted": "unweighted LS", "weighted": "weighted LS", "maximum_likelihood": "max. likelihood"}
_CHECK_MODES = ("ignore", "warn", "strict")


def _col(name):
    return THERMO_COLORS.get(name, "tab:gray")


def _lab(name):
    return THERMO_LABELS.get(name, str(name))


def _check_mode(mode):
    if mode not in _CHECK_MODES:
        raise ValueError(f"checks must be one of {_CHECK_MODES}, got {mode!r}")


# ---------------------------------------------------------------------------
# Debye-Hueckel / modDH building blocks (molar OR molal: same functional form)
# ---------------------------------------------------------------------------
def debye_huckel_A_molar(dielectric_constant, T):
    """Limiting-law slope of ln(gamma+-) for a 1:1 salt, per sqrt(mol/L).

    A = (1/2) l_B^(3/2) sqrt(8 pi N_A c),  l_B = e^2 / (4 pi eps0 eps_r kB T).
    eps_r = 59.1 (the TIP4P/2005 value used by Hosseini & Ashbaugh) at 298.15 K gives 1.7964.
    It depends on the WATER MODEL's dielectric constant, so pass the right one for your model.
    """
    e, eps0, kB = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23
    lB = e ** 2 / (4 * np.pi * eps0 * dielectric_constant * kB * T)
    return 0.5 * lB ** 1.5 * np.sqrt(8 * np.pi * NA * 1000.0)


def _g(x):
    """g(x) = (2x + x^2)/(1+x) - 2 ln(1+x), stable for tiny x (series ~ x^3/3).

    The Debye-Hueckel osmotic term is -(A/B^3) g(B sqrt(c)); the textbook form cancels
    catastrophically when a fitted B is ~0 (it can be, for poorly constrained fits).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 0.05
    xs = x[small]
    out[small] = sum(((-1) ** (n + 1)) * (1.0 - 2.0 / n) * xs ** n for n in range(3, 14))
    xl = x[~small]
    out[~small] = (2 * xl + xl ** 2) / (1 + xl) - 2 * np.log1p(xl)
    return out


def dh_osmotic(c, A, B):
    """Debye-Hueckel part of the osmotic bracket (<= 0), c in M (molar set) or m (molal set)."""
    c = np.asarray(c, dtype=float)
    return -(A / B ** 3) * _g(B * np.sqrt(c))


def pi_bar(c, A, B, a1, a2, nu, T):
    """Osmotic pressure (bar) of the molar modDH description; c in mol/L.

    Pi = 0.01 nu R T [c - (A/B^3) g(B sqrt c) + a1 c^2/2 + 2 a2 c^3/3]   (1 J/L = 0.01 bar)
    """
    c = np.asarray(c, dtype=float)
    return 0.01 * nu * R * T * (c + dh_osmotic(c, A, B) + 0.5 * a1 * c ** 2 + (2.0 / 3.0) * a2 * c ** 3)


def ln_gamma(c, a1, a2, A, B):
    """ln(gamma+-) = a1 c + a2 c^2 - A sqrt(c)/(1 + B sqrt(c))."""
    c = np.asarray(c, dtype=float)
    return a1 * c + a2 * c ** 2 - A * np.sqrt(c) / (1 + B * np.sqrt(c))


# ---------------------------------------------------------------------------
# measured profiles, reservoir density, local molality
# ---------------------------------------------------------------------------
def _water_group(u, ion_groups_u, water_selection=None):
    if water_selection:
        w = u.select_atoms(water_selection)
    else:
        w = u.select_atoms("resname SOL and name OW")
        if len(w) == 0:
            w = u.select_atoms("resname HOH and name O*")
        if len(w) == 0:
            ions = ion_groups_u[0] | ion_groups_u[1]
            w = u.select_atoms("element O").difference(ions)
    if len(w) == 0:
        raise RuntimeError("no water oxygens found for the density profile (pass water_selection=...)")
    return w


def measured_profiles(us, ion_groups, dz_nm, molar_mass_salt, water_selection=None):
    """Replicate-averaged folded profiles: z (nm), rho (g/cm3), c_s (M), c_w (M), m (mol/kg).

    Also `rho_reps`: the per-replicate rho(z), for the reservoir check.
    """
    cs_all, cw_all = [], []
    for u, grp in zip(us, ion_groups):
        w = _water_group(u, grp, water_selection)
        z, _rho, _m, c_s, c_w = density_molality_profiles(u, w, grp[0], grp[1], dz_nm, molar_mass_salt)
        cs_all.append(c_s); cw_all.append(c_w)
    c_s = np.mean(cs_all, axis=0); c_w = np.mean(cw_all, axis=0)
    rho = (c_w * M_WATER_G_PER_MOL + c_s * molar_mass_salt) / 1000.0
    m = np.where(c_w > 1e-6, c_s / (c_w * M_WATER_G_PER_MOL / 1000.0), np.nan)
    rho_reps = [(cw * M_WATER_G_PER_MOL + cs * molar_mass_salt) / 1000.0 for cs, cw in zip(cs_all, cw_all)]
    return dict(z=np.asarray(z), rho=rho, c_s=c_s, c_w=c_w, m=m, rho_reps=rho_reps)


def reservoir_density(prof, slab_nm=1.0):
    """Mean solution density (g/cm3) in the outermost `slab_nm` of the box, per replicate.

    That region is the salt-free reservoir; its density says which pressure the
    osmotic force balance is actually running at (pure water of the model at 1 atm is the
    intended state). Always computed; whether it is *enforced* is `checks=`.
    """
    z = np.asarray(prof["z"])
    sel = z >= z.max() - slab_nm
    return np.array([float(np.mean(r[sel])) for r in prof["rho_reps"]])


def reservoir_check(rho_res, mode="ignore", target=None, tol=0.005):
    """mode 'ignore' (default): silent. 'warn': print. 'strict': raise.
    `target` (g/cm3) is the pure-water density of YOUR model at the run's T and P; None = no
    target, so nothing can be 'off' and only the numbers are printed."""
    _check_mode(mode)
    rho_res = np.asarray(rho_res, dtype=float)
    bad = [] if target is None else [i for i, r in enumerate(rho_res) if abs(r - target) > tol]
    if mode == "ignore":
        return bad
    for i, r in enumerate(rho_res):
        tag = "   <-- OFF target %.4f +/- %.4f" % (target, tol) if i in bad else ""
        print(f"  reservoir water density r{i}: {r:.4f} g/cm3{tag}")
    if bad:
        msg = (f"reservoir water density is off target {target} +/- {tol} g/cm3 in replicate(s) {bad}: "
               f"Pi(C) is not at the 1 atm state point the method assumes.")
        if mode == "strict":
            raise ValueError(msg)
        print("  WARNING: " + msg)
    return bad


def box_check(areas, area_ref, mode="ignore", rtol=0.02):
    """Compare each replicate's mean Lx*Ly with the salt_data BOX area."""
    _check_mode(mode)
    bad = [i for i, a in enumerate(areas) if abs(a / area_ref - 1.0) > rtol]
    if mode == "ignore":
        return bad
    for i in bad:
        msg = (f"replicate {i}: mean Lx*Ly = {areas[i]:.3f} nm^2 vs salt_data BOX area {area_ref:.3f} nm^2 "
               f"({100 * (areas[i] / area_ref - 1):+.1f}%)")
        if mode == "strict":
            raise ValueError(msg)
        print("  WARNING: " + msg)
    return bad


def reservoir_note(rho_res, target=None):
    """One-line text for plots: the checked reservoir density."""
    r = np.asarray(rho_res, dtype=float)
    s = f"reservoir $\\rho_w$ = {r.mean():.4f} g/cm$^3$"
    if r.size > 1:
        s += f" ({r.min():.4f}–{r.max():.4f} over {r.size} replicates)"
    if target is not None:
        s += f"; target {target:.4f}"
    return s


def make_local_molal(prof, M_water):
    """c_s [mol/L] -> LOCAL molality [mol/kg] from the measured water profile.

    m = c_s / (kg water per litre at that z). The water density varies with z (ions displace
    water; compressibility), so a single box-average factor is wrong at the centre and at the
    ends. 1/(kg water per L) is smooth in c_s: fit a quadratic and clip outside the data.
    """
    cs, cw = np.asarray(prof["c_s"], float), np.asarray(prof["c_w"], float)
    ok = (cs > 0.02) & (cw > 1e-6)
    r = 1.0 / (cw * M_water / 1000.0)
    coef = np.polyfit(cs[ok], r[ok], 2)
    lo, hi = float(cs[ok].min()), float(cs[ok].max())

    def to_molal(c):
        c = np.asarray(c, dtype=float)
        return c * np.polyval(coef, np.clip(c, lo, hi))
    return to_molal


def _sorted_curve(c, y):
    c = np.asarray(c, float); y = np.asarray(y, float)
    ok = np.isfinite(c) & np.isfinite(y) & (c > 0)
    c, y = c[ok], y[ok]
    order = np.argsort(c)
    c, y = c[order], y[order]
    c, idx = np.unique(c, return_index=True)
    return c, y[idx]


# ---------------------------------------------------------------------------
# the thermodynamic calculations (pure: profiles + fit results in, curves out)
# ---------------------------------------------------------------------------
def thermo_from_profiles(prof, outdir, salt, mi, molar_mass_salt, vant_hoff, T, A_fit, methods,
                         exp_molality=None, exp_oc=None, exp_op=None, to_molal=None,
                         kappa=KAPPA_EFF_PER_BAR, rho_w0_ref=None, reference=None,
                         rho_res=None, reservoir_target=None, make_plots=True, verbose=True):
    """
    prof     : dict from measured_profiles (z, rho, c_s, c_w, m)
    methods  : {name: dict(params=[B, a1, a2], c_molar=<grid mol/L>, op_bar=<Pi on grid>, oc=<phi on grid>)}
    A_fit    : the A (M^-1/2) the fits used; the molal set uses A~ = A sqrt(rho_w0), B~ = B sqrt(rho_w0)
    kappa    : effective compressibility for eq 23 (bar^-1). The default is the value Hosseini & Ashbaugh
               fitted for TIP4P/2005; for another water model pass its own (0 = skip the P correction)
    rho_w0_ref : pure-water density used in eq 19. None -> the intercept of the eq-22 fit
    reference  : optional dict(label=..., molar=(A,B,a1,a2), molal=(A~,B~,a~1,a~2),
                 rho_mu=(4 eq-22 coeffs), rho_P=(4)) drawn as extra curves; any key may be missing
    Returns dict(out=<json-able summary>, res=<per-method curves>, coeffs_mu=..., m_grid=...).
    """
    nu = vant_hoff
    Mw = M_WATER_G_PER_MOL
    say = print if verbose else (lambda *a, **k: None)
    say("\n=== thermodynamic extras (density, V_w, molal description, ln gamma, chemical potentials) ===")

    good = np.isfinite(prof["m"]) & np.isfinite(prof["rho"]) & (prof["m"] >= 0)
    m_pts, rho_pts, cs_pts = prof["m"][good], prof["rho"][good], prof["c_s"][good]
    m_max = float(np.nanmax(m_pts))

    # eq 22 at constant mu_w: it is the measured density, independent of the osmotic fit
    coeffs_mu = fit_density_eq22(m_pts, rho_pts)
    rho_w0 = float(coeffs_mu[0])
    rho_ref = float(rho_w0_ref) if rho_w0_ref is not None else rho_w0
    say(f"  eq 22 (mu_w): rho_w0={rho_w0:.5f} g/cm3 theta1={coeffs_mu[1]:.5g} "
        f"theta3/2={coeffs_mu[2]:.5g} theta2={coeffs_mu[3]:.5g};  rho_w0 used in eq 19: {rho_ref:.5f}")

    c_srt, m_srt = _sorted_curve(cs_pts, m_pts)
    m_grid = np.linspace(0.02, m_max, 200)

    ref = reference
    if ref is not None:
        say(f"  extra reference curves: {ref.get('label', 'reference')}")
        if "molal" in ref:
            Am_r, Bm_r, a1m_r, a2m_r = ref["molal"]
            ref_bracket = m_grid + dh_osmotic(m_grid, Am_r, Bm_r) + 0.5 * a1m_r * m_grid ** 2 + (2 / 3) * a2m_r * m_grid ** 3
            ref_lng = ln_gamma(m_grid, a1m_r, a2m_r, Am_r, Bm_r)
            ref_rtlnaw = -nu * R * T * 1e-3 * (Mw / 1000.0) * ref_bracket

    out = {"rho_w0_mu": rho_w0, "rho_w0_used_eq19": rho_ref, "coeffs_mu": list(map(float, coeffs_mu)),
           "m_max": m_max, "kappa_eff_per_bar": float(kappa), "A_used": float(A_fit),
           "reservoir_density_g_cm3": None if rho_res is None else list(map(float, np.atleast_1d(rho_res))),
           "reservoir_density_target": reservoir_target,
           "reference": None if ref is None else {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v)
                                                  for k, v in ref.items()},
           "methods": {}}
    res = {}
    for name, d in methods.items():
        B, a1_m, a2_m = d["params"][0], d["params"][1], d["params"][2]
        cg, opg = _sorted_curve(d["c_molar"], d["op_bar"])
        _, ocg = _sorted_curve(d["c_molar"], d["oc"])
        # this method's Pi and phi on the LOCAL molality axis
        m_of_c = to_molal(cg) if to_molal is not None else np.interp(cg, c_srt, m_srt)
        order = np.argsort(m_of_c)
        m_of_c, opg_s, ocg_s = m_of_c[order], opg[order], ocg[order]
        Pi_m = np.interp(m_grid, m_of_c, opg_s)
        Pi_pts = np.interp(cs_pts, cg, opg)                    # Pi at each measured profile point

        # eq 23 -> constant-P density, refit with rho_w0 pinned
        rho_P_pts = density_const_P(rho_pts, Pi_pts, kappa)
        coeffs_P = fit_density_eq22(m_pts, rho_P_pts, fix_rho_w0=rho_w0)

        # partial molar volume of water: eq 24, then cubic smoothing
        Vw_raw = partial_molar_volume_water(m_grid, coeffs_P, molar_mass_salt)
        gam = np.polyfit(m_grid, Vw_raw, 3)                     # highest power first
        Vw = np.polyval(gam, m_grid)

        # molal (constant-P) Debye-Hueckel model, eq 19 mapping
        s = np.sqrt(rho_ref)
        A_m, B_m = A_fit * s, B * s
        pref = 0.01 * nu * R * T * (Mw / Vw)                   # bar per (mol/kg)
        dh = dh_osmotic(m_grid, A_m, B_m)
        resid = Pi_m / pref - (m_grid + dh)
        design = np.column_stack([0.5 * m_grid ** 2, (2.0 / 3.0) * m_grid ** 3])
        (al1, al2), *_ = np.linalg.lstsq(design, resid, rcond=None)
        bracket = m_grid + 0.5 * al1 * m_grid ** 2 + (2.0 / 3.0) * al2 * m_grid ** 3 + dh
        op_molal = pref * bracket
        phi_molal = bracket / m_grid
        lng = ln_gamma(m_grid, al1, al2, A_m, B_m)
        rtlnaw = -nu * R * T * 1e-3 * (Mw / 1000.0) * bracket        # kJ/mol  (= mu_w - mu_w^0)
        rtlnaw_fromPi = -Pi_m * Vw * 1e-4                             # bar*cm3/mol = 0.1 J/mol -> kJ/mol
        sc = np.sqrt(cg)
        mu_s_ex = -A_fit * sc / (1 + B * sc) + a1_m * cg + a2_m * cg ** 2   # mu_s^ex/RT, molar description
        rms = float(np.sqrt(np.mean((op_molal - Pi_m) ** 2)))
        say(f"  [{name}] B={B:.4g} a1={a1_m:.4g} a2={a2_m:.4g} (molar) -> B~={B_m:.4g} a~1={al1:.4g} "
            f"a~2={al2:.4g}; V_w(0)={Vw[0]:.3f} V_w({m_max:.2f})={Vw[-1]:.3f} cm3/mol; "
            f"RMS(molal OP - molar OP)={rms:.3f} bar")
        res[name] = dict(c_molar=cg, op_molar=opg, mu_s_ex=mu_s_ex, Pi=Pi_m,
                         Pi_molar_phi=np.interp(m_grid, m_of_c, ocg_s), coeffs_P=coeffs_P, Vw=Vw,
                         Vw_raw=Vw_raw, op_molal=op_molal, phi_molal=phi_molal, lng=lng,
                         rtlnaw=rtlnaw, rtlnaw_fromPi=rtlnaw_fromPi, al1=al1, al2=al2, A_m=A_m, B_m=B_m)
        out["methods"][name] = dict(
            molar_params=dict(B=float(B), alpha1=float(a1_m), alpha2=float(a2_m)),
            molal_params=dict(B_tilde=float(B_m), A_tilde=float(A_m), alpha1_tilde=float(al1), alpha2_tilde=float(al2)),
            coeffs_P=list(map(float, coeffs_P)), pmv_cubic_gamma3_to_gamma0=list(map(float, gam)),
            rms_molalOP_minus_molarOP_bar=rms)

    if not make_plots:
        return dict(out=out, res=res, coeffs_mu=coeffs_mu, m_grid=m_grid)

    note = None if rho_res is None else reservoir_note(rho_res, reservoir_target)
    lab = "" if ref is None else ref.get("label", "reference")

    def save(fig, tag):
        fn = f"{outdir}/thermo_{tag}_{salt}.png"
        fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig); say(f"  wrote {fn}")

    def stamp(ax):
        if note:
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    # 1 density
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(m_pts, rho_pts, s=8, c="0.6", label="simulation (measured, $\\mu_w$ const.)")
    mm = np.linspace(0, m_max, 200)
    ax.plot(mm, density_eq22(mm, coeffs_mu), "k-", lw=2, label="eq 22 fit, constant $\\mu_w$")
    if ref is not None and "rho_mu" in ref:
        ax.plot(mm, density_eq22(mm, ref["rho_mu"]), "r-", lw=1.5, label=f"{lab}, constant $\\mu_w$")
    if ref is not None and "rho_P" in ref:
        ax.plot(mm, density_eq22(mm, ref["rho_P"]), "r:", lw=1.5, label=f"{lab}, constant P")
    for name, r in res.items():
        ax.plot(mm, density_eq22(mm, r["coeffs_P"]), "--", c=_col(name),
                label=f"eq 22/23, constant P ({_lab(name)})")
    ax.set_xlabel("molality (mol/kg)"); ax.set_ylabel("density (g/cm$^3$)"); ax.set_title(f"{salt} solution density")
    ax.legend(fontsize=8); ax.grid(alpha=.3); stamp(ax); save(fig, "density")

    # 2 partial molar volume of water
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in res.items():
        ax.plot(m_grid, r["Vw"], c=_col(name), lw=2, label=_lab(name))
        ax.plot(m_grid, r["Vw_raw"], c=_col(name), lw=0.8, ls=":")
    ax.set_xlabel("molality (mol/kg)"); ax.set_ylabel("$\\bar V_w$ (cm$^3$/mol)")
    ax.set_title(f"{salt} partial molar volume of water (dotted: before cubic smoothing)")
    ax.legend(); ax.grid(alpha=.3); stamp(ax); save(fig, "pmv_water")

    # 3 osmotic pressure: molar vs molal description
    fig, ax = plt.subplots(figsize=(7, 5))
    if exp_molality is not None and exp_op is not None:
        ax.plot(exp_molality, exp_op, "ko-", label="experimental")
    for name, r in res.items():
        ax.plot(m_grid, r["Pi"], c=_col(name), lw=2, label=f"{_lab(name)} - molar fit (as reported)")
        ax.plot(m_grid, r["op_molal"], c=_col(name), ls="--", lw=1.5,
                label=f"{_lab(name)} - molal description")
    ax.set_xlim(0, m_max * 1.05); ax.set_xlabel("molality (mol/kg)"); ax.set_ylabel("osmotic pressure (bar)")
    ax.set_title(f"{salt} osmotic pressure: molar vs molal description"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    stamp(ax); save(fig, "op_molar_vs_molal")

    # 3b osmotic pressure vs MOLARITY (the quantity the molar fit is made in)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in res.items():
        ax.plot(r["c_molar"], r["op_molar"], c=_col(name), lw=2, label=_lab(name))
    if ref is not None and "molar" in ref:
        cc = np.linspace(0.02, max(r["c_molar"].max() for r in res.values()), 200)
        ax.plot(cc, pi_bar(cc, *ref["molar"], nu, T), "r--", lw=2, label=f"{lab} (molar)")
    ax.set_xlabel("concentration $C_s$ (mol/L)"); ax.set_ylabel("osmotic pressure (bar)")
    ax.set_title(f"{salt} osmotic pressure vs molarity"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    stamp(ax); save(fig, "op_vs_molarity")

    # 4 osmotic coefficient
    fig, ax = plt.subplots(figsize=(7, 5))
    if exp_molality is not None and exp_oc is not None:
        ax.plot(exp_molality, exp_oc, "ko-", label="experimental")
    for name, r in res.items():
        ax.plot(m_grid, r["Pi_molar_phi"], c=_col(name), lw=2, label=f"{_lab(name)} - molar fit")
        ax.plot(m_grid, r["phi_molal"], c=_col(name), ls="--", label=f"{_lab(name)} - molal description")
    if ref is not None and "molal" in ref:
        ax.plot(m_grid, ref_bracket / m_grid, "r-", lw=1.5, label=f"{lab} (molal)")
    ax.set_xlim(0, m_max * 1.05); ax.set_xlabel("molality (mol/kg)"); ax.set_ylabel("osmotic coefficient")
    ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title(f"{salt} osmotic coefficient"); stamp(ax)
    save(fig, "oc_molar_vs_molal")

    # 5 mean ionic activity coefficient
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in res.items():
        ax.plot(m_grid, r["lng"], c=_col(name), lw=2, label=_lab(name))
    if ref is not None and "molal" in ref:
        ax.plot(m_grid, ref_lng, "r-", lw=1.5, label=f"{lab} (molal)")
    ax.axhline(0, c="k", lw=.5); ax.set_xlabel("molality (mol/kg)"); ax.set_ylabel("$\\ln\\gamma_\\pm$")
    ax.set_title(f"{salt} mean ionic activity coefficient"); ax.legend(); ax.grid(alpha=.3); stamp(ax)
    save(fig, "ln_gamma")

    # 6 water chemical potential / activity
    fig, ax = plt.subplots(figsize=(7, 5))
    if exp_molality is not None and exp_oc is not None:
        em = np.asarray(exp_molality, float)
        ax.plot(np.sqrt(em), -nu * R * T * 1e-3 * (Mw / 1000.0) * em * np.asarray(exp_oc, float), "ko-",
                label="experimental")
    for name, r in res.items():
        ax.plot(np.sqrt(m_grid), r["rtlnaw"], c=_col(name), lw=2, label=f"{_lab(name)} (molal description)")
        ax.plot(np.sqrt(m_grid), r["rtlnaw_fromPi"], c=_col(name), ls=":",
                label=f"{_lab(name)}  $-\\Pi\\bar V_w$")
    if ref is not None and "molal" in ref:
        ax.plot(np.sqrt(m_grid), ref_rtlnaw, "r-", lw=1.5, label=f"{lab} (molal)")
    ax.set_xlabel("$\\sqrt{m}$ (mol/kg)$^{1/2}$"); ax.set_ylabel("$\\mu_w-\\mu_w^0$ = RT ln $a_w$ (kJ/mol)")
    ax.set_title(f"{salt} water chemical potential / activity"); ax.legend(fontsize=7); ax.grid(alpha=.3); stamp(ax)
    save(fig, "water_activity")

    # 7 ion chemical potential (excess), molar description = the quantity eq 12 is fitted to
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in res.items():
        ax.plot(r["c_molar"], r["mu_s_ex"], c=_col(name), lw=2, label=_lab(name))
    if ref is not None and "molar" in ref:
        cc = np.linspace(0.02, max(r["c_molar"].max() for r in res.values()), 200)
        Ar, Br, a1r, a2r = ref["molar"]
        ax.plot(cc, ln_gamma(cc, a1r, a2r, Ar, Br), "r--", lw=2, label=f"{lab} (molar)")
    ax.axhline(0, c="k", lw=.5); ax.set_xlabel("concentration $C_s$ (mol/L)")
    ax.set_ylabel("$\\mu_s^{ex}/RT = \\ln\\gamma_\\pm$ (molar)")
    ax.set_title(f"{salt} salt excess chemical potential (constant $\\mu_w$)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    stamp(ax); save(fig, "mu_ion_excess")

    # table at the experimental molalities (blank beyond what the box reaches)
    if exp_molality is not None:
        def _at(m, y):
            m = np.asarray(m, float)
            return np.where((m >= m_grid.min()) & (m <= m_grid.max()), np.interp(m, m_grid, y), np.nan)
        em_arr = np.asarray(exp_molality, float)
        with open(f"{outdir}/thermo_results_{salt}_{mi}m.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Molality", "Method", "Density_muw_const_g_cm3", "Density_P_const_g_cm3",
                        "Vw_cm3_mol", "ln_gamma_pm", "muw_minus_muw0_kJ_mol(RTln_aw)",
                        "Pi_molar_fit_bar", "Pi_molal_description_bar"])
            for name, r in res.items():
                cols = [_at(em_arr, density_eq22(m_grid, coeffs_mu)), _at(em_arr, density_eq22(m_grid, r["coeffs_P"])),
                        _at(em_arr, r["Vw"]), _at(em_arr, r["lng"]), _at(em_arr, r["rtlnaw"]),
                        _at(em_arr, r["Pi"]), _at(em_arr, r["op_molal"])]
                for i, mm_ in enumerate(em_arr):
                    w.writerow([f"{mm_:g}", name] + ["" if not np.isfinite(c[i]) else f"{c[i]:.6g}" for c in cols])
        say(f"  wrote {outdir}/thermo_results_{salt}_{mi}m.csv")

    np.savez(f"{outdir}/thermo_curves_{salt}_{mi}m.npz", m_grid=m_grid,
             rho_profile_m=m_pts, rho_profile=rho_pts, coeffs_mu=np.asarray(coeffs_mu),
             **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items()})
    with open(f"{outdir}/thermo_params_{salt}_{mi}m.json", "w") as fh:
        json.dump(out, fh, indent=2)
    say(f"  wrote {outdir}/thermo_params_{salt}_{mi}m.json and thermo_curves_{salt}_{mi}m.npz")
    return dict(out=out, res=res, coeffs_mu=coeffs_mu, m_grid=m_grid)


def run_thermo_extras(outdir, ion1, ion2, mi, us, ion_groups, dz_nm, molar_mass_salt, vant_hoff, T, A_fit,
                      methods, exp_molality, exp_oc, exp_op, to_molal=None, prof=None, water_selection=None,
                      **kw):
    """Measure the profiles (unless given) and run thermo_from_profiles."""
    if prof is None:
        prof = measured_profiles(us, ion_groups, dz_nm, molar_mass_salt, water_selection)
    return thermo_from_profiles(prof, outdir, f"{ion1}{ion2}", mi, molar_mass_salt, vant_hoff, T, A_fit,
                                methods, exp_molality, exp_oc, exp_op, to_molal=to_molal, **kw)


# ===========================================================================
# MAIN ANALYSIS FUNCTION
# ===========================================================================

def run_analysis(
    salt_data_path,                 # per-box salt_data_*.py the SIMULATION used
    results_dir,                    # folder with md<mi>m_r<i>.pdb / .xtc
    ion1,
    ion2,
    molality,                       # single target molality (HP = one per run)
    N_replicates,
    water: str = "water",           # label only
    ff_used: str = "",              # label only
    n_expansion_terms: int = 1,
    dz_nm: float = 0.05,
    n_bootstraps: int = 500,
    profiles_only: bool = False,
    random_seed: int = 1,
    eps: float = 1e-4,
    out_dir=None,
    resname1: str = None,
    resname2: str = None,
    center_atom1: str = None,
    center_atom2: str = None,
    k_override=None,                # only if the run used dispatch -k
    thermo_extras=True,             # density / V_w / molal description / ln gamma / chemical potentials
    legacy=False,                   # True: old A (divided by sqrt(1e24/NA)) and box-average molality
    checks="ignore",                # "ignore" | "warn" | "strict": box-area and reservoir-density checks
    reservoir_density_target=None,  # g/cm3 of pure water for YOUR model at your T,P (used by warn/strict only)
    reservoir_density_tol=0.005,    # g/cm3
    dielectric_constant=59.1,       # of the water model; sets A (59.1 = TIP4P/2005, gives A = 1.7964)
    A_molar=None,                   # M^-1/2; overrides dielectric_constant
    kappa_eff_per_bar=KAPPA_EFF_PER_BAR,  # eq 23 compressibility (default: fitted for TIP4P/2005); 0 = skip
    rho_w0_ref=None,                # pure-water density for eq 19 (None: the eq-22 intercept)
    reference=None,                 # optional dict of literature curves to draw (see thermo_from_profiles)
    water_selection=None,           # MDAnalysis selection of the water oxygens (None: SOL/OW, HOH/O*, element O)
):
    """Run the HP analysis for one salt at one molality.

    Everything physical - k, T, van't Hoff factor, box, ion count and the
    experimental phi/density reference - comes from `salt_data_path`, the same
    file the simulation used, so the analysis cannot disagree with the run.

    `results_dir` is the folder the dispatch wrote:
        <case>/result_files/<ktag>_<mi>m/md<mi>m_r<i>.pdb|.xtc
    Its k-tag is parsed and checked against salt_data.
    """
    # ------------------------------------------------------------------
    # Everything derived from salt_data + the results directory
    # ------------------------------------------------------------------
    SD  = load_salt_data(salt_data_path)
    BOX = SD.BOX
    row = SD.lookup(f"{ion1}{ion2}", molality)

    k_val = float(k_override) if k_override is not None else row.k_HP
    k     = k_val * (kilojoule_per_mole / nanometer ** 2)
    T        = BOX["temperature"]
    vantHoff = SD.VANT_HOFF[f"{ion1}{ion2}"]
    L_z      = BOX["lz"]

    resname1 = resname1 or ion1.upper()
    resname2 = resname2 or ion2.upper()
    center_atom1 = center_atom1 or ion1
    center_atom2 = center_atom2 or ion2
    resname_ions = f"{resname1} {resname2}"

    frdir  = Path(results_dir)
    outdir = Path(out_dir) if out_dir else frdir
    outdir.mkdir(parents=True, exist_ok=True)
    if not frdir.is_dir():
        raise FileNotFoundError(f"results_dir not found: {frdir}")

    mi, mi1 = _format_molality(molality)

    # The dispatch names the folder <ktag>_<mi>m. Refuse to analyse a run whose
    # k does not match the salt_data file being used to interpret it.
    dir_tag = frdir.name.split("_")[0]
    want_tag = _k_to_tag(k_val)
    if dir_tag.startswith("k") and dir_tag != want_tag:
        raise ValueError(
            f"k mismatch: results_dir '{frdir.name}' was produced with {dir_tag}, "
            f"but {'k_override' if k_override is not None else Path(salt_data_path).name}"
            f" gives {want_tag} (k = {k_val}).\n"
            f"Set k_override to the k the run actually used, or point at the "
            f"matching results directory."
        )

    # z-grid shared by every profile and by the bootstrap histograms
    nbins_common = max(1, int(round((L_z / 2) / dz_nm)))
    edges_common = np.linspace(0, L_z / 2, nbins_common + 1)

    print(f"[salt_data] {salt_data_path}")
    print(f"  {ion1}{ion2} @ {mi1} mol/kg : N={row.num_particles} pairs, "
          f"k={k_val:.4f} ({want_tag})"
          f"{'  <- OVERRIDE' if k_override is not None else ''}, "
          f"nu={vantHoff}, T={T} K")
    print(f"  box {BOX['lx']} x {BOX['ly']} x {L_z} nm, z0={BOX['z_center']} nm, "
          f"{BOX['n_water']} waters")
    print(f"  {nbins_common} z-bins, dz_eff={(L_z/2)/nbins_common:.4f} nm")
    print(f"  reading {frdir}")
    print(f"  writing {outdir}")

    # ------------------------------------------------------------------
    # Debye-Hückel A coefficient
    # ------------------------------------------------------------------
    _check_mode(checks)
    global A  # used inside calc_y / osmotic_pressure
    if A_molar is not None:
        A = float(A_molar)
    elif vantHoff == 2:
        A = debye_huckel_A_molar(dielectric_constant, T)
    elif vantHoff == 3:
        A = 2.50                       # original hard-coded value; not re-derived here
    else:
        raise ValueError(f"Unsupported vantHoff factor: {vantHoff}")
    print(f"A = {A:.4f} M^-1/2")
    if legacy:                         # old behaviour: A divided by sqrt(1e24/NA) against mol/L profiles
        conversion_factor = 1e24 / NA
        A /= conversion_factor ** 0.5

    # ------------------------------------------------------------------
    # Eq. 12 functions
    # ------------------------------------------------------------------
    def calc_y(theta, x, nterms=n_expansion_terms):
        C_max = np.max(x)
        term_1 = np.log(x / C_max)
        term_2 = -A * np.sqrt(x) / (1 + theta[0] * np.sqrt(x))
        term_3 = A * np.sqrt(C_max) / (1 + theta[0] * np.sqrt(C_max))
        y = term_1 + term_2 + term_3
        for i in range(1, nterms + 1):
            y += theta[i] * (x ** i - C_max ** i)
        return y

    def residuals(theta, x, y):
        return np.power(calc_y(theta, x) - y, 2)

    # ------------------------------------------------------------------
    # Load salt information
    # ------------------------------------------------------------------
    salt_dict = load_salt_info(SD, ion1, ion2)

    # ------------------------------------------------------------------
    # Experimental data
    # exp_concs is now in mol/kg (molality), read directly from key names.
    # ------------------------------------------------------------------
    exp_concs, yexpOC = extract_experiment_data(salt_dict, f"{ion1}{ion2}")

    yexpOP = []
    for i, m in enumerate(exp_concs):
        op_res = osm_experimental(
            exp_osm_coeff=yexpOC[i],
            vant_hoff=vantHoff,
            molality=exp_concs[i],   # mol/kg
            T=T * kelvin,
        )
        yexpOP.append(op_res.value_in_unit(op_res.unit))

    # ------------------------------------------------------------------
    # Load trajectories
    # ------------------------------------------------------------------
    us, ions_sel = [], []
    for i in range(N_replicates):
        pdb = frdir / f"md{mi}m_r{i}.pdb"
        xtc = frdir / f"md{mi}m_r{i}.xtc"
        for f_ in (pdb, xtc):
            if not f_.exists():
                raise FileNotFoundError(f"{f_} not found - check results_dir")
        u = mda.Universe(str(pdb), str(xtc))   # on-disk reader; streams below
        us.append(u)
        # de-duplicating union (|), NOT '+': when both centres share an element
        # and no resnames are given, '+' lists every atom twice.
        ions_sel.append(select_ion(u, resname1, center_atom1)
                        | select_ion(u, resname2, center_atom2))

    # ------------------------------------------------------------------
    # Count water molecules (needed for molar→molal conversion)
    # We select TIP3P water oxygens (residue name SOL or HOH, element O).
    # ------------------------------------------------------------------
    N_water_list, V_box_list = [], []
    for u in us:
        water_ag = u.select_atoms("resname SOL and name OW")
        if len(water_ag) == 0:
            water_ag = u.select_atoms("resname HOH and name O*")
        if len(water_ag) == 0:
            water_ag = u.select_atoms(f"element O and not (resname {resname_ions})")
        N_water_list.append(len(water_ag))

        dim = u.trajectory.ts.dimensions
        V_box_list.append(dim[0] / 10.0 * dim[1] / 10.0 * dim[2] / 10.0 * 1e-24)

    N_water_mean = float(np.mean(N_water_list))
    V_box_mean   = float(np.mean(V_box_list))
    print(f"  mean N_water = {N_water_mean:.0f} (salt_data: {BOX['n_water']}), "
          f"mean V_box = {V_box_mean:.4e} L")
    if abs(N_water_mean - BOX["n_water"]) > 1:
        raise ValueError(
            f"trajectory has {N_water_mean:.0f} waters, {Path(salt_data_path).name} "
            f"says {BOX['n_water']} - wrong salt_data file for these trajectories?")

    # ------------------------------------------------------------------
    # Measured water/ion profiles -> reservoir density and LOCAL molality converter
    # ------------------------------------------------------------------
    _ion_groups = [(select_ion(u_, resname1, center_atom1), select_ion(u_, resname2, center_atom2))
                   for u_ in us]
    _molar_mass_salt = SD.MOLAR_MASS[f"{ion1}{ion2}"]
    _prof = measured_profiles(us, _ion_groups, dz_nm, _molar_mass_salt, water_selection)
    _rho_res = reservoir_density(_prof)
    reservoir_check(_rho_res, checks, reservoir_density_target, reservoir_density_tol)
    _res_note = reservoir_note(_rho_res, reservoir_density_target)
    if legacy:
        def to_molal(c):
            return convert_profile_to_molal(c, N_water_mean, V_box_mean)
    else:
        to_molal = make_local_molal(_prof, M_WATER_G_PER_MOL)

    # ------------------------------------------------------------------
    # Convert experimental concentrations to molal
    # (use tabulated molality values directly from salt_dict)
    # ------------------------------------------------------------------
    # exp_concs is already in mol/kg -- exp_molals alias removed

    # ------------------------------------------------------------------
    # Concentration profiles
    # ------------------------------------------------------------------
    vnames = ["B", "alpha1", "alpha2", "alpha3", "alpha4"]

    zvals = []
    for i in range(N_replicates):
        zvals.append(get_ion_array(us[i], ions_sel[i]))
    zvals_all = np.concatenate(zvals)

    # ------------------------------------------------------------------
    # Build count + molarity profiles (per-replicate)
    # ------------------------------------------------------------------
    z_ion1_cnt, ion1_cnt = [], []
    z_ion2_cnt, ion2_cnt = [], []
    z_salt_cnt, salt_cnt = [], []

    z_ion1_M, ion1_M = [], []
    z_ion2_M, ion2_M = [], []
    z_salt_M, salt_M = [], []

    Ns_ion1_list, Ns_ion2_list, Ns_pairs_list, LxLy_list = [], [], [], []

    for u in us:
        ag1 = select_ion(u, resname1, center_atom1)
        ag2 = select_ion(u, resname2, center_atom2)

        z1, _w1, cts1 = count_profile_z_fixed_Lz(u, ag1, dz_nm)
        z2, _w2, cts2 = count_profile_z_fixed_Lz(u, ag2, dz_nm)
        z_ion1_cnt.append(z1); ion1_cnt.append(cts1)
        z_ion2_cnt.append(z2); ion2_cnt.append(cts2)
        z_salt_cnt.append(z1); salt_cnt.append(0.5 * (cts1 + cts2))

        z1M, c1M = molarity_profile_fixed_Lz(u, ag1, dz_nm)
        z2M, c2M = molarity_profile_fixed_Lz(u, ag2, dz_nm)
        z_ion1_M.append(z1M); ion1_M.append(c1M)
        z_ion2_M.append(z2M); ion2_M.append(c2M)
        z_salt_M.append(z1M); salt_M.append(0.5 * (c1M + c2M))

        Ns_ion1_list.append(ag1.n_atoms)
        Ns_ion2_list.append(ag2.n_atoms)
        Ns_pairs_list.append(min(ag1.n_atoms, ag2.n_atoms))
        LxLy_list.append(get_average_cross_section_area(u))

    def fold_stack(z_list, y_list):
        folded = [fold_profile(z, y) for z, y in zip(z_list, y_list)]
        zpos = folded[0][0]
        stack = np.vstack([yf for _, yf in folded])
        return zpos, stack, stack.mean(axis=0)

    box_check(LxLy_list, BOX["area"], checks)

    zpos_cnt_1, stack_cnt_1, mean_cnt_1 = fold_stack(z_ion1_cnt, ion1_cnt)
    zpos_cnt_2, stack_cnt_2, mean_cnt_2 = fold_stack(z_ion2_cnt, ion2_cnt)
    zpos_cnt_s, stack_cnt_s, mean_cnt_s = fold_stack(z_salt_cnt, salt_cnt)

    zpos_M_1, stack_M_1, mean_M_1 = fold_stack(z_ion1_M, ion1_M)
    zpos_M_2, stack_M_2, mean_M_2 = fold_stack(z_ion2_M, ion2_M)
    zpos_M_s, stack_M_s, mean_M_s = fold_stack(z_salt_M, salt_M)

    # ------------------------------------------------------------------
    # Plot 1 – counts/bin
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for y in stack_cnt_1: ax.plot(zpos_cnt_1, y, alpha=0.2, lw=1, color="yellowgreen")
    for y in stack_cnt_2: ax.plot(zpos_cnt_2, y, alpha=0.2, lw=1, color="c")
    for y in stack_cnt_s: ax.plot(zpos_cnt_s, y, alpha=0.2, lw=1, color="tab:blue")
    ax.plot(zpos_cnt_1, mean_cnt_1, lw=2.6, color="yellowgreen", label=f"{ion1} (mean)")
    ax.plot(zpos_cnt_2, mean_cnt_2, lw=2.6, color="c",           label=f"{ion2} (mean)")
    ax.plot(zpos_cnt_s, mean_cnt_s, lw=2.6, color="tab:blue",    label=f"{ion1}{ion2} (per-salt mean)")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Average ions per bin")
    ax.set_title(f"HP - Count profiles (per-salt) for {ion1}{ion2}")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/countsprof_{ion1}{ion2}.png", dpi=300)
    plt.close(fig)

    # Same curves as the PNG above, as data, so different boxes can be
    # overlaid without rerunning the analysis.
    np.savez(
        f"{outdir}/countsprof_{ion1}{ion2}.npz",
        z_ion1=zpos_cnt_1, mean_ion1=mean_cnt_1, std_ion1=stack_cnt_1.std(axis=0),
        z_ion2=zpos_cnt_2, mean_ion2=mean_cnt_2, std_ion2=stack_cnt_2.std(axis=0),
        z_salt=zpos_cnt_s, mean_salt=mean_cnt_s, std_salt=stack_cnt_s.std(axis=0),
        ion1=ion1, ion2=ion2, n_replicates=N_replicates,
    )

    # ------------------------------------------------------------------
    # Plot 2 – molal concentration profile
    # (converted from molar)
    # ------------------------------------------------------------------
    stack_molal_1 = np.array([to_molal(y) for y in stack_M_1])
    stack_molal_2 = np.array([to_molal(y) for y in stack_M_2])
    stack_molal_s = np.array([to_molal(y) for y in stack_M_s])
    mean_molal_1 = stack_molal_1.mean(axis=0)
    mean_molal_2 = stack_molal_2.mean(axis=0)
    mean_molal_s = stack_molal_s.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    for y in stack_molal_1: ax.plot(zpos_M_1, y, alpha=0.2, lw=1, color="yellowgreen")
    for y in stack_molal_2: ax.plot(zpos_M_2, y, alpha=0.2, lw=1, color="c")
    for y in stack_molal_s: ax.plot(zpos_M_s, y, alpha=0.2, lw=1, color="tab:blue")
    ax.plot(zpos_M_1, mean_molal_1, lw=2.6, color="yellowgreen", label=f"{ion1} (mean)")
    ax.plot(zpos_M_2, mean_molal_2, lw=2.6, color="c",           label=f"{ion2} (mean)")
    ax.plot(zpos_M_s, mean_molal_s, lw=2.6, color="tab:blue",    label=f"{ion1}{ion2} (per-salt mean)")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Concentration (mol/kg)")
    ax.set_title(f"HP - Concentration profiles (per-salt) for {ion1}{ion2} [molal]")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/concprof_{ion1}{ion2}.png", dpi=300)
    plt.close(fig)

    # Same curves as the PNG above, as data, so different boxes can be
    # overlaid (e.g. on a log axis) without rerunning the analysis.
    np.savez(
        f"{outdir}/concprof_{ion1}{ion2}.npz",
        z_ion1=zpos_M_1, mean_ion1=mean_molal_1, std_ion1=stack_molal_1.std(axis=0),
        z_ion2=zpos_M_2, mean_ion2=mean_molal_2, std_ion2=stack_molal_2.std(axis=0),
        z_salt=zpos_M_s, mean_salt=mean_molal_s, std_salt=stack_molal_s.std(axis=0),
        ion1=ion1, ion2=ion2, n_replicates=N_replicates,
    )

    # --- per-replicate concentration profiles (spot a bad replicate) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, y in enumerate(stack_molal_s):
        ax.plot(zpos_M_s, y, lw=1.2, label=f"Replicate {i + 1}")
    ax.plot(zpos_M_s, mean_molal_s, lw=2.5, ls="--", color="k", label="mean")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Concentration (mol/kg)")
    ax.set_title(f"HP {ion1}{ion2} + {water} at {mi1}m - per-replicate concentration")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/conc_prof_byreplicate_{ion1}{ion2}.png", dpi=300)
    plt.close(fig)

    # Summary
    N_s_list = []
    for i, (Nc, Nb, Np) in enumerate(zip(Ns_ion1_list, Ns_ion2_list, Ns_pairs_list)):
        print(f"Rep {i}: ion1={Nc}, ion2={Nb}, Pairs={Np}")
        N_s_list.append(Np)
    N_s = np.mean(N_s_list)
    approx_total_ions = mean_cnt_s.sum()
    print(f"Approx ions/frame from folded per-salt counts (rep-mean): ~{approx_total_ions:.2f}")
    LxLy = np.average(np.array(LxLy_list))
    print(f"Average cross-sectional area is {LxLy:.4f}")

    # Bin-occupancy diagnostic: with dz_nm small enough, most bins hold well
    # under one ion per frame on average, so their measured concentration is
    # dominated by counting (shot) noise rather than signal. This is the
    # mechanism behind the weighted eq-12 fit (and anything seeded from it,
    # including the ML optimizer) destabilising at fine bin widths - it is
    # not something dz-independent methods like the raw per-ion ML likelihood
    # suffer from directly, but they DO inherit a bad initial guess from it.
    n_bins_occ = mean_cnt_s.size
    frac_below_1 = float(np.mean(mean_cnt_s < 1.0))
    frac_below_0p1 = float(np.mean(mean_cnt_s < 0.1))
    print(f"  Bin occupancy (dz_nm={dz_nm}): {n_bins_occ} bins, "
          f"min mean count/frame={mean_cnt_s.min():.4g}, "
          f"{100*frac_below_1:.1f}% of bins <1 ion/frame, "
          f"{100*frac_below_0p1:.1f}% <0.1 ion/frame"
          f"{'  <-- WARNING: most bins are shot-noise dominated' if frac_below_1 > 0.5 else ''}")

    # ------------------------------------------------------------------
    # Build mean molal profile for fitting (from molarity then convert)
    # ------------------------------------------------------------------
    c_0 = mean_M_s               # molar mean profile (fold_stack already averaged)
    z = np.asarray(zpos_M_s)
    mask = np.isfinite(c_0) & (c_0 > eps)
    z_fit = z[mask]
    c_fit = c_0[mask]           # fitting is done in molar (Eq. 12 uses nm units internally)

    # ------------------------------------------------------------------
    # Least-squares fitting
    # ------------------------------------------------------------------
    initial_guess = [4, 0.2, 0.0, 0.0, 0.0][:n_expansion_terms + 1]

    def y_values(z_fits):
        return -(0.5 * k.value_in_unit(kilojoule_per_mole / nanometer ** 2) * z_fits ** 2) * 1000 / (R * T)

    result_unweighted_bad = least_squares(residuals, initial_guess, args=(c_fit, y_values(z_fit)))
    print(result_unweighted_bad.x)

    result_unweighted = curve_fit(
        lambda xin, *t: calc_y(t, xin),
        xdata=c_fit, ydata=y_values(z_fit),
        p0=initial_guess, sigma=None, check_finite=True,
    )
    ps = result_unweighted[0]; cov = result_unweighted[1]
    std = np.sqrt(np.diag(cov))
    print("parameters and uncertainties (unweighted)")
    for i, p in enumerate(ps):
        print(f"  {vnames[i]:^7s}: {p:6.3f} +/- {std[i]:.3f}")
    print(f"Condition number of covariance: {np.linalg.cond(cov):.3g}")

    # Plotting Eq. 12 fits
    fig, ax = plt.subplots()
    ax.plot(z_fit, y_values(z_fit), "b", label="-U(z)/RT")
    ax.plot(z_fit, np.log(c_fit / np.max(c_fit)), "g--", label="Ideal")
    ax.plot(z_fit, calc_y(result_unweighted[0], c_fit), "r--", label="Fitting eq 12")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("-U(z)/RT")
    ax.set_title("Fitting of eq. 12 - Unweighted results")
    ax.legend()
    fig.savefig(f"{outdir}/fit_eq12_unweighted_{ion1}{ion2}.png", dpi=150)
    plt.close(fig)

    # Weighted fitting
    norm_weighted = simpson(c_fit / np.max(c_fit), x=z_fit)
    print(f"norm_weighted = {norm_weighted}")

    def weighted_residuals(theta, x, y):
        return (x / (norm_weighted * np.max(x) - x)) * np.power(np.subtract(calc_y(theta, x), y), 2)

    result_weighted_bad = least_squares(weighted_residuals, initial_guess, args=(c_fit, y_values(z_fit)))

    result_weighted = curve_fit(
        lambda xin, *t: calc_y(t, xin),
        xdata=c_fit, ydata=y_values(z_fit),
        p0=initial_guess,
        sigma=np.sqrt((norm_weighted * np.max(c_fit) - c_fit) / c_fit),
        check_finite=True,
    )
    ps = result_weighted[0]; cov = result_weighted[1]
    std = np.sqrt(np.diag(cov))
    print("parameters and uncertainties (weighted)")
    for i, p in enumerate(ps):
        print(f"  {vnames[i]:^7s}: {p:.4f} +/- {std[i]:.4f}")
    print(f"Condition number of covariance: {np.linalg.cond(cov):.3g}")

    # Weighted vs unweighted residuals comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z_fit, y_values(z_fit) - np.log(c_fit / np.max(c_fit)), "g--", label="Ideal")
    ax.plot(z_fit, y_values(z_fit) - calc_y(result_unweighted[0], c_fit), "r--", label="Unweighted")
    ax.plot(z_fit, y_values(z_fit) - calc_y(result_unweighted_bad.x, c_fit), "y--", label="Prev Unweighted")
    ax.plot(z_fit, y_values(z_fit) - calc_y(result_weighted[0], c_fit), "m--", label="Weighted")
    ax.plot(z_fit, y_values(z_fit) - calc_y(result_weighted_bad.x, c_fit), "c--", label="Prev Weighted")
    ax.axhline(y=0, color="k")
    ax.set_xlabel("z (nm)"); ax.set_ylim([-0.5, 0.5]); ax.set_ylabel("-U(z)/RT")
    ax.set_title("Weighted vs Unweighted Fitting (y_values - results)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/fit_comparison_{ion1}{ion2}.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------
    np.random.seed(random_seed)

    red_U_app = lambda z_: 1000 * k.value_in_unit(kilojoule_per_mole / nanometer ** 2) * z_ * z_ / (2 * R * T)
    cz_ideal = lambda z_: np.exp(-red_U_app(z_))

    ideal_norm = np.sqrt(np.pi * R * T / (2 * 1000 * k.value_in_unit(kilojoule_per_mole / nanometer ** 2)))
    print(f"Normalisation constant: {ideal_norm:.3f}")

    cmax_ideal = N_s / (2 * LxLy * ideal_norm)
    cmax_ideal *= 1e24 / NA
    _cmax_molal = to_molal(np.array([cmax_ideal]))[0]
    _c0_meas = float(np.max(c_0))
    _c0_molal = to_molal(np.array([_c0_meas]))[0]
    print(f"Ideal peak (cmax_ideal): {cmax_ideal:.3f} M = {_cmax_molal:.3f} mol/kg")
    print(f"Measured peak (c_0):     {_c0_meas:.3f} M = {_c0_molal:.3f} mol/kg"
          f"   -> measured/ideal = {_c0_meas / cmax_ideal:.3f}")

    def logprobfunc(a, z_, cz_):
        cmax_ = np.max(cz_)
        czh = np.sqrt(cz_); cmaxh = np.sqrt(cmax_)
        terms = -red_U_app(z_)
        terms += A * ((czh / (1 + a[0] * czh)) - (cmaxh / (1 + a[0] * cmaxh)))
        for i in range(1, n_expansion_terms + 1):
            terms += -a[i] * (cz_ ** i - cmax_ ** i)
        return terms

    def czfunc(a, z_, cz_):
        cznew = np.exp(logprobfunc(a, z_, cz_))
        zunique, locs = np.unique(z_, return_index=True)
        newarea = simpson(cznew[locs], x=zunique)
        cznorm_ratio = ideal_norm / newarea
        return (cmax_ideal * cznorm_ratio) * cznew

    def converge_c(params_, z_, cz_start, niter=1000, lim=0.0001):
        if isinstance(cz_start, str) and cz_start == "ideal":
            cz_start = cmax_ideal * cz_ideal(z_)
        c_old = cz_start
        for _ in range(niter):
            c_new = czfunc(params_, z_, c_old)
            norm_ = np.sqrt(np.dot(c_new - c_old, c_new - c_old))
            c_old = c_new
            if norm_ < lim:
                break
        return czfunc(params_, z_, c_new)

    def neglliter(a, z_, cz_start, counts=None, lim=0.0001, doeval=False):
        if len(z_) != len(cz_start):
            return
        if doeval:
            lim = 1
        cz_ = converge_c(a, z_, cz_start, lim=lim)
        terms = np.log(cz_ / cmax_ideal)
        csum = np.sum(terms) if counts is None else np.dot(counts, terms)
        return -1 * csum

    def bootstrap_histograms(samples, n_boot, wtype="unweighted", init_guess=None):
        if init_guess is None:
            init_guess = initial_guess
        b_params, b_profiles = [], []
        for i in range(n_boot + 1):
            n_samples = len(samples)
            indices = np.arange(n_samples) if i == 0 else np.random.randint(0, n_samples, size=n_samples)
            bootstrap_sample = samples[indices]
            conc, bins = np.histogram(
                bootstrap_sample, bins=edges_common, density=True
            )
            # density=True already divides by the bin width. A folded bin spans
            # TWO slabs of area LxLy, and pairs = ions/2, hence the single /2.
            conc = hist_to_molar(conc, N_s, LxLy)
            zbins = 0.5 * (bins[1:] + bins[:-1])
            msk = np.isfinite(conc) & (conc > eps)
            z_f = zbins[msk]; c_f = conc[msk]
            sigma_ = None if wtype == "unweighted" else np.sqrt((norm_weighted * np.max(c_f) - c_f) / c_f)
            res = curve_fit(
                lambda xin, *t: calc_y(t, xin),
                xdata=c_f, ydata=y_values(z_f),
                p0=init_guess, sigma=sigma_,
            )
            b_params.append(res[0])
            b_profiles.append(conc)
        return np.array(b_profiles), np.array(b_params)

    def bootstrap_samples(samples, minfunc, n_boot, init_params):
        zinit = np.unique(samples)
        c_start = converge_c(init_params, zinit, cz_start="ideal")
        bootstrap_params, bootstrap_cs = [], []
        n_samples = len(samples)
        for i in range(n_boot + 1):
            indices = np.arange(n_samples) if i == 0 else np.random.randint(0, n_samples, size=n_samples)
            bootstrap_sample = samples[indices]
            zsparse, zcount = np.unique(bootstrap_sample, return_counts=True)
            c_new = np.interp(zsparse, zinit, c_start)
            result = minimize(minfunc, init_params, args=(zsparse, c_new, zcount), method="Nelder-Mead")
            print(i, result.x, result.fun, result.success)
            bootstrap_params.append(result.x)
            c_new = converge_c(result.x, zsparse, cz_start=c_new)
            c_new = np.interp(zinit, zsparse, c_new)
            bootstrap_cs.append(c_new)
        return np.array(bootstrap_cs), np.array(bootstrap_params)

    # ------------------------------------------------------------------
    # Analyse bootstrap (parameter distributions + osmotic properties)
    # ------------------------------------------------------------------
    def analyze_bootstrap(bps, bcs, averes):
        bs_mean_parameters = []
        for i in range(n_expansion_terms + 1):
            pd_ = bps[1:, i]
            std_param = np.std(pd_); mean_param = np.mean(pd_)
            bs_mean_parameters.append(mean_param)
            print(f"Param {i} = {bps[0, i]:.3f} +/- {std_param:.3f} (mean {mean_param:.4f})",
                  f" repl.-ave. {averes[0][i]:.3f} +/- {np.sqrt(averes[1][i, i]):.3f}")
            fig, ax = plt.subplots()
            ax.hist(pd_, bins="scott")
            ax.set_title(f"Bootstrap distribution - parameter {vnames[i]}")
            ax.set_xlabel(f"parameter {vnames[i]}")
            ax.set_ylabel("Count")
            fig.savefig(f"{outdir}/param_{vnames[i]}_{ion1}{ion2}.png", dpi=150)
            plt.close(fig)
        for i in range(n_expansion_terms + 1):
            for j in range(i + 1, n_expansion_terms + 1):
                fig, ax = plt.subplots()
                ax.scatter(bps[1:, i], bps[1:, j], c="m", s=10, alpha=0.3)
                ax.scatter(bps[0, i], bps[0, j], color="k", marker="*", s=50)
                ax.scatter(averes[0][i], averes[0][j], color="b", marker="*", s=50)
                ax.set_title(f"Bootstrap parameter correlation: {vnames[i]} vs {vnames[j]}")
                ax.set_xlabel(f"parameter {vnames[i]}")
                ax.set_ylabel(f"parameter {vnames[j]}")
                fig.savefig(f"{outdir}/param_corr_{vnames[i]}_{vnames[j]}_{ion1}{ion2}.png", dpi=150)
                plt.close(fig)
        return bs_mean_parameters

    def plot_osmotic_coefficients(bps, bcs, averes, avec, title, ylim):
        ops, ocs = [], []
        for i, r in enumerate(zip(bps, bcs)):
            if i != 0:
                p, c = r[0], r[1]
                op, oc = osmotic_pressure(c, p, nu=vantHoff, nterms=n_expansion_terms, T=T)
                ops.append(op); ocs.append(oc)
        ops = np.array(ops); ocs = np.array(ocs)

        opt_op, opt_oc = osmotic_pressure(bcs[0], bps[0], nu=vantHoff, nterms=n_expansion_terms, T=T)
        ave_op, ave_oc = osmotic_pressure(avec, averes[0], nu=vantHoff, nterms=n_expansion_terms, T=T)

        bounds = {"95": {"lower": 2.5, "upper": 97.5}, "1s": {"lower": 16, "upper": 84}}
        percentiles_oc = {key: np.percentile(ocs, [bounds[key]["lower"], bounds[key]["upper"]], axis=0)
                          for key in bounds}
        percentiles_op = {key: np.percentile(ops, [bounds[key]["lower"], bounds[key]["upper"]], axis=0)
                          for key in bounds}
        oc_std = np.std(ocs, axis=0); op_std = np.std(ops, axis=0)
        oc_sig = 0.5 * (percentiles_oc["1s"][1] - percentiles_oc["1s"][0])
        op_sig = 0.5 * (percentiles_op["1s"][1] - percentiles_op["1s"][0])
        alpha = np.min([1, 100.0 / len(ocs)])

        # Convert molar concentration axis → molal for plots
        bcs_0_molal = to_molal(bcs[0])
        avec_molal  = to_molal(avec)
        bcs_molal   = [to_molal(c) for c in bcs[1:]]

        # Osmotic coefficient plot (molal x-axis)
        fig, ax = plt.subplots()
        ax.set_title(f"Osmotic Coefficients: {title}")
        for c_m, oc in zip(bcs_molal, ocs):
            ax.plot(c_m, oc, lw=0.1, alpha=alpha, c="m")
        ax.errorbar(bcs_0_molal, opt_oc, yerr=oc_sig, errorevery=10, c="k", lw=1, label="all")
        ax.errorbar(avec_molal, ave_oc, yerr=oc_sig, errorevery=10, c="b", lw=1, label="repl. ave.")
        ax.set_ylim(ylim)
        ax.set_xlabel("Concentration (mol/kg)")
        ax.set_ylabel("Osmotic coefficient")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{outdir}/oc_{title.replace(' ', '_')}_{ion1}{ion2}.png", dpi=150)
        plt.close(fig)

        # Osmotic pressure plot (molal x-axis)
        fig, ax = plt.subplots()
        ax.set_title(f"Osmotic Pressures: {title}")
        for c_m, op in zip(bcs_molal, ops):
            ax.plot(c_m, op, lw=0.1, alpha=alpha, c="m")
        ax.errorbar(bcs_0_molal, opt_op, yerr=op_sig, errorevery=10, c="k", lw=1, label="all")
        ax.errorbar(avec_molal, ave_op, yerr=op_sig, errorevery=10, c="b", lw=1, label="repl.ave.")
        ax.set_xlabel("Concentration (mol/kg)")
        ax.set_ylabel("Osmotic pressure")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{outdir}/op_{title.replace(' ', '_')}_{ion1}{ion2}.png", dpi=150)
        plt.close(fig)

        results = dict(
            opt_op=opt_op, opt_oc=opt_oc, ave_op=ave_op, ave_oc=ave_oc,
            oc_std=oc_std, op_std=op_std,
            percentiles_oc=percentiles_oc, percentiles_op=percentiles_op,
        )
        return results

    # ------------------------------------------------------------------
    # Run bootstraps
    # ------------------------------------------------------------------
    if profiles_only:
        print(f"\n[profiles_only] stopping before the bootstraps. "
              f"Figures in {outdir}")
        return dict(
            z=z, c_0=c_0, z_fit=z_fit, c_fit=c_fit,
            zpos=zpos_M_s, stack_molal=stack_molal_s, mean_molal=mean_molal_s,
            counts_mean=mean_cnt_s,
            params_unweighted=result_unweighted[0],
            params_weighted=result_weighted[0],
            cmax_ideal=cmax_ideal, N_s=N_s, LxLy=LxLy,
            N_water_mean=N_water_mean, V_box_mean=V_box_mean,
            reservoir_density=_rho_res,
        )

    bprofiles_unweighted, bparams_unweighted = bootstrap_histograms(
        samples=zvals_all, n_boot=n_bootstraps, wtype="unweighted",
        init_guess=result_unweighted[0],
    )

    # bootstrap index 0 is the un-resampled data, so it must reproduce c_0
    _b0 = np.asarray(bprofiles_unweighted[0])
    if _b0.shape == np.shape(c_0):
        _m = np.isfinite(_b0) & np.isfinite(c_0) & (c_0 > eps)
        _r = _b0[_m] / c_0[_m]
        print(f"[check] bootstrap[0]/c_0: median={np.median(_r):.4f} "
              f"min={_r.min():.4f} max={_r.max():.4f}   (want 1.0000)")
    else:
        print(f"[check] SHAPE MISMATCH {_b0.shape} vs {np.shape(c_0)} "
              f"- the two z-grids disagree")
    analyze_bootstrap(
        bparams_unweighted, bprofiles_unweighted, result_unweighted
    )
    unweighted_osmotic = plot_osmotic_coefficients(
        bparams_unweighted, bprofiles_unweighted, result_unweighted, c_0,
        title="Least squares unweighted parameters", ylim=[0.9, 1.2],
    )

    bprofiles_weighted, bparams_weighted = bootstrap_histograms(
        samples=zvals_all, n_boot=n_bootstraps, wtype="weighted",
    )
    analyze_bootstrap(
        bparams_weighted, bprofiles_weighted, result_weighted
    )
    weighted_osmotic = plot_osmotic_coefficients(
        bparams_weighted, bprofiles_weighted, result_weighted, c_0,
        title="Least squares weighted parameters", ylim=[0.9, 1.2],
    )

    # ------------------------------------------------------------------
    # ML bootstrap
    # ------------------------------------------------------------------
    zsparse_all, zcount_all = np.unique(zvals_all, return_counts=True)
    newp = result_weighted[0]
    c_new_sparse_all = converge_c(newp, zsparse_all, cz_start="ideal", lim=0.001)
    results_nm = minimize(
        neglliter, newp,
        args=(zsparse_all, c_new_sparse_all, zcount_all),
        method="Nelder-Mead",
    )
    print(results_nm)
    full_opt = results_nm.x
    c_opt_sparse_all = converge_c(full_opt, zsparse_all, cz_start=c_new_sparse_all)

    bprofiles_ml, bparams_ml = bootstrap_samples(
        zvals_all, neglliter, n_boot=n_bootstraps, init_params=full_opt
    )
    stdparam = np.std(bparams_ml, axis=0)
    mxy = np.outer(stdparam, stdparam)
    covparam = np.cov(bparams_ml.T, ddof=0)
    covcorr = covparam / mxy
    print("correlation matrix"); print(covcorr)
    print(f"Condition number of covariance: {np.linalg.cond(covparam):.3g}")

    analyze_bootstrap(
        bparams_ml, bprofiles_ml, [full_opt, covparam]
    )
    ml_osmotic = plot_osmotic_coefficients(
        bparams_ml, bprofiles_ml, [full_opt], c_opt_sparse_all,
        title="Maximum Likelihood", ylim=[0.9, 1.2],
    )

    # ------------------------------------------------------------------
    # Max concentrations
    # ------------------------------------------------------------------
    maxconcs = []
    fig, ax = plt.subplots()
    for i in range(len(bprofiles_ml)):
        ax.scatter(zsparse_all, bprofiles_ml[i], s=0.1, lw=0.1, c="m")
        maxconcs.append(bprofiles_ml[i][0])
    ax.set_title(f"ML bootstrap concentration profiles - {ion1}{ion2}")
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("Concentration (mol/L)")
    fig.savefig(f"{outdir}/maxconc_scatter_{ion1}{ion2}.png", dpi=150)
    plt.close(fig)
    av_maxconcs = np.mean(maxconcs)
    # bprofiles_ml is already in mol/L (converge_c scales by cmax_ideal, which is
    # molar), so convert straight to molal. Do NOT apply 1e24/NA here -- that was a
    # spurious extra conversion that inflated the reported Cmax by ~1.66x.
    av_maxconcs_molal = to_molal(
        np.array([av_maxconcs]))[0]
    print(f"Average maximum concentration {av_maxconcs_molal:.2f} mol/kg")

    # ------------------------------------------------------------------
    # Osmotic results table
    # ------------------------------------------------------------------
    def calculate_osmotic_properties(target_values, percentiles_oc, percentiles_op, c_opt, opt_oc, opt_op):
        results_ = dict(
            osmotic_coefficient=[], osmotic_coefficient_error=[],
            osmotic_pressure=[], osmotic_pressure_error=[],
        )
        c_top = np.nanmax(c_opt)
        for tv in target_values:
            if tv > 1.02 * c_top:
                for _k in results_:
                    results_[_k].append(float("nan"))
                print(f"Target {tv:.2f} mol/kg is beyond the curve (max {c_top:.2f}) - left blank")
                continue
            ci, cv = find_closest(c_opt, tv)
            y_oc = opt_oc[ci]
            u_oc = (percentiles_oc["95"][1][ci] - percentiles_oc["95"][0][ci]) / 2
            y_op = opt_op[ci]
            u_op = (percentiles_op["95"][1][ci] - percentiles_op["95"][0][ci]) / 2
            results_["osmotic_coefficient"].append(float(y_oc))
            results_["osmotic_coefficient_error"].append(float(u_oc))
            results_["osmotic_pressure"].append(float(y_op))
            results_["osmotic_pressure_error"].append(float(u_op))
            print(f"Closest to {tv:.2f}: {cv:.2f} idx {ci}, OC: {y_oc:.3f}±{u_oc:.3f}, OP: {y_op:.3f}±{u_op:.3f}")
        return results_

    # Convert simulation profiles to molal for the results table and plots
    c_opt_sparse_molal = to_molal(c_opt_sparse_all)
    bprofiles_uw_molal = to_molal(bprofiles_unweighted[0])
    bprofiles_w_molal  = to_molal(bprofiles_weighted[0])
    bprofiles_ml_molal = to_molal(bprofiles_ml[0])

    # exp_concs is already in mol/kg; bprofiles_ml_molal is also in mol/kg
    osmotic_results = calculate_osmotic_properties(
        exp_concs,
        ml_osmotic["percentiles_oc"], ml_osmotic["percentiles_op"],
        bprofiles_ml_molal, ml_osmotic["opt_oc"], ml_osmotic["opt_op"],
    )

    # Save JSON / CSV
    json_results = json.dumps(format_dict(osmotic_results), indent=4)
    with open(f"{outdir}/{ion1}{ion2}_final_results_{mi}m.json", "w") as outfile:
        outfile.write(json_results)

    csv_filename = f"{outdir}/{ion1}{ion2}_final_results_{mi}m.csv"
    with open(csv_filename, mode="w", newline="") as csvfile:
        fieldnames = [
            "Molality", "Osmotic Coefficient", "Osmotic Coefficient Error",
            "Osmotic Pressure", "Osmotic Pressure Error",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, molality_key in enumerate(salt_dict[f"{ion1}{ion2}"].keys()):
            writer.writerow({
                "Molality": molality_key.split()[1],
                "Osmotic Coefficient": _blank_nan(osmotic_results["osmotic_coefficient"][i]),
                "Osmotic Coefficient Error": _blank_nan(osmotic_results["osmotic_coefficient_error"][i]),
                "Osmotic Pressure": _blank_nan(osmotic_results["osmotic_pressure"][i]),
                "Osmotic Pressure Error": _blank_nan(osmotic_results["osmotic_pressure_error"][i]),
            })
    print(f"Results saved to {csv_filename}")

    # ------------------------------------------------------------------
    # Final summary plots (molal x-axis)
    # ------------------------------------------------------------------
    MEDIUM_SIZE, BIGGER_SIZE = 18, 20
    # Note: c_opt_sparse_molal, bprofiles_*_molal already computed above.

    lower_bound_95_oc, upper_bound_95_oc = ml_osmotic["percentiles_oc"]["95"]
    lower_bound_95_op, upper_bound_95_op = ml_osmotic["percentiles_op"]["95"]

    N_i_last = int(round(N_s))          # measured, not a table lookup

    # --- Osmotic coefficient vs molal ---
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    ax.plot(exp_concs, yexpOC, marker="o", linestyle="-", color="black",
            label="Experimental (Hamer & Wu)", linewidth=3, markersize=8)
    ax.plot(c_opt_sparse_molal, ml_osmotic["opt_oc"], lw=2, label=f"{ff_used}", color="darkgreen")
    ax.fill_between(c_opt_sparse_molal, lower_bound_95_oc, upper_bound_95_oc,
                    color="g", alpha=0.2, label="95% Confidence Interval")
    ax.set_xlim(0, molality + 0.5)
    ax.set_title(
        f"HP using k={k_val:.2f} kJ/(mol·nm²) - {ion1}{ion2}: Cmax {av_maxconcs_molal:.1f} mol/kg ({N_i_last} ion pairs)",
        fontsize=BIGGER_SIZE,
    )
    ax.set_xlabel("Concentration (mol/kg)", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Osmotic Coefficients", fontsize=MEDIUM_SIZE)
    ax.legend(); ax.grid()
    ax.text(0.02, 0.02, _res_note, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    fig.savefig(f"{outdir}/{ion1}{ion2}_oc.png")
    plt.close(fig)

    # --- OC vs log(molal) ---
    fig, ax = plt.subplots()
    ax.plot(np.log(c_opt_sparse_molal), ml_osmotic["opt_oc"], label="Mean Profile", color="darkgreen")
    ax.fill_between(np.log(c_opt_sparse_molal), lower_bound_95_oc, upper_bound_95_oc,
                    color="g", alpha=0.2, label="95% Confidence Interval")
    ax.plot(np.log(exp_concs), yexpOC, marker="o", linestyle="-", color="black",
            label="Experimental (Hamer & Wu)", linewidth=3, markersize=8)
    ax.set_xlabel("Log(Concentration mol/kg)")
    ax.set_ylabel("Osmotic Coefficients")
    ax.grid()
    fig.savefig(f"{outdir}/{ion1}{ion2}_oc_log.png")
    plt.close(fig)

    # --- Osmotic pressure vs molal ---
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    ax.plot(exp_concs, yexpOP, marker="o", linestyle="-", color="black",
            label="Experimental (Hamer & Wu)", linewidth=3, markersize=8)
    ax.plot(c_opt_sparse_molal, ml_osmotic["opt_op"], lw=2, label=f"{ff_used}", color="darkgreen")
    ax.fill_between(c_opt_sparse_molal, lower_bound_95_op, upper_bound_95_op,
                    color="g", alpha=0.2, label="95% Confidence Interval")
    ax.set_xlim(0, molality + 0.5)
    ax.set_title(
        f"HP using k={k_val:.2f} kJ/(mol·nm²) - {ion1}{ion2}: Cmax {av_maxconcs_molal:.1f} mol/kg ({N_i_last} ion pairs)",
        fontsize=BIGGER_SIZE,
    )
    ax.set_xlabel("Concentration (mol/kg)", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Osmotic Pressure (bar)", fontsize=MEDIUM_SIZE)
    ax.legend(); ax.grid()
    ax.text(0.02, 0.02, _res_note, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    fig.savefig(f"{outdir}/{ion1}{ion2}_op.png")
    plt.close(fig)

    # --- Comparison plot (molal) ---
    fig, ax = plt.subplots()
    ax.plot(bprofiles_uw_molal, unweighted_osmotic["opt_oc"], lw=1.5, label="unweighted")
    ax.fill_between(
        bprofiles_uw_molal,
        unweighted_osmotic["opt_oc"] + 0.5 * unweighted_osmotic["oc_std"],
        unweighted_osmotic["opt_oc"] - 0.5 * unweighted_osmotic["oc_std"],
        alpha=0.4,
    )
    ax.plot(bprofiles_w_molal, weighted_osmotic["opt_oc"], lw=1.5, label="weighted")
    ax.fill_between(
        bprofiles_w_molal,
        weighted_osmotic["opt_oc"] + 0.5 * weighted_osmotic["oc_std"],
        weighted_osmotic["opt_oc"] - 0.5 * weighted_osmotic["oc_std"],
        alpha=0.4,
    )
    ax.plot(bprofiles_ml_molal, ml_osmotic["opt_oc"], lw=1.5, label="maximum_likelihood")
    ax.fill_between(
        bprofiles_ml_molal,
        ml_osmotic["opt_oc"] + 0.5 * ml_osmotic["oc_std"],
        ml_osmotic["opt_oc"] - 0.5 * ml_osmotic["oc_std"],
        alpha=0.4,
    )
    ax.set_ylim([0.9, 1.3])
    ax.legend()
    ax.set_xlabel("Concentration (mol/kg)")
    ax.set_ylabel("Osmotic Coefficient")
    ax.set_title(
        f"Osmotic coefficient [molal]:\n{n_expansion_terms + 1} expansion "
        f"terms with k={k_val:.2f}"
    )
    fig.savefig(f"{outdir}/comp_{want_tag}_{n_expansion_terms}_{eps:.2e}.pdf")
    plt.close(fig)

    # Same three curves as the PDF above, but as data, so they can be replotted
    # or compared across runs without rerunning the analysis.
    np.savez(
        f"{outdir}/curves_{ion1}{ion2}_{mi}m_{n_expansion_terms}alphas.npz",
        c_unweighted=bprofiles_uw_molal, oc_unweighted=unweighted_osmotic["opt_oc"],
        std_unweighted=unweighted_osmotic["oc_std"],
        c_weighted=bprofiles_w_molal, oc_weighted=weighted_osmotic["opt_oc"],
        std_weighted=weighted_osmotic["oc_std"],
        c_maximum_likelihood=bprofiles_ml_molal, oc_maximum_likelihood=ml_osmotic["opt_oc"],
        std_maximum_likelihood=ml_osmotic["oc_std"],
        exp_molality=np.asarray(exp_concs), exp_oc=np.asarray(yexpOC),
        n_expansion_terms=n_expansion_terms, k=k_val, molality=molality,
        salt=f"{ion1}{ion2}", water=water, ff_used=ff_used,
        n_pairs=N_i_last, n_replicates=N_replicates,
    )


    if thermo_extras:
        import traceback
        try:
            run_thermo_extras(
                outdir=outdir, ion1=ion1, ion2=ion2, mi=mi,
                us=us, ion_groups=_ion_groups, dz_nm=dz_nm,
                molar_mass_salt=_molar_mass_salt, vant_hoff=vantHoff, T=T, A_fit=A,
                methods={
                    "unweighted": dict(params=result_unweighted[0], c_molar=bprofiles_unweighted[0],
                                       op_bar=unweighted_osmotic["opt_op"], oc=unweighted_osmotic["opt_oc"]),
                    "weighted": dict(params=result_weighted[0], c_molar=bprofiles_weighted[0],
                                     op_bar=weighted_osmotic["opt_op"], oc=weighted_osmotic["opt_oc"]),
                    "maximum_likelihood": dict(params=full_opt, c_molar=c_opt_sparse_all,
                                               op_bar=ml_osmotic["opt_op"], oc=ml_osmotic["opt_oc"]),
                },
                exp_molality=exp_concs, exp_oc=yexpOC, exp_op=yexpOP,
                to_molal=to_molal, prof=_prof, kappa=kappa_eff_per_bar, rho_w0_ref=rho_w0_ref,
                reference=reference, rho_res=_rho_res, reservoir_target=reservoir_density_target,
            )
        except Exception:
            # the osmotic results above are already on disk; do not lose them to the extras
            print("[thermo_extras] FAILED - osmotic results are unaffected:")
            traceback.print_exc()

    print("Analysis complete. All figures saved to result directories.")

    return dict(
        dz_nm=dz_nm,
        c_unweighted=bprofiles_uw_molal, oc_unweighted=unweighted_osmotic["opt_oc"],
        std_unweighted=unweighted_osmotic["oc_std"],
        c_weighted=bprofiles_w_molal, oc_weighted=weighted_osmotic["opt_oc"],
        std_weighted=weighted_osmotic["oc_std"],
        c_ml=bprofiles_ml_molal, oc_ml=ml_osmotic["opt_oc"],
        std_ml=ml_osmotic["oc_std"],
        exp_molality=np.asarray(exp_concs), exp_oc=np.asarray(yexpOC),
        reservoir_density=_rho_res, A=A,
    )


# ===========================================================================
# HISTOGRAM-WIDTH SENSITIVITY SCAN
# ===========================================================================

def run_dz_width_scan(
    salt_data_path,
    results_dir,
    ion1,
    ion2,
    molality,
    N_replicates,
    base_dz_nm: float = 0.05,
    width_factors=(2.0, 1.0, 0.5, 0.25, 0.1),
    n_bootstraps: int = 200,
    out_dir=None,
    **run_analysis_kwargs,
):
    """Re-run the HP analysis at several histogram bin widths (dz_nm) and plot
    how the unweighted / weighted / maximum-likelihood osmotic-coefficient
    curves respond.

    `width_factors` multiply `base_dz_nm`: 2.0 -> bins twice as wide (coarser,
    half as many bins), 0.5 -> half as wide, 0.25 -> a quarter, 0.1 -> a tenth
    (much finer, noisier per-bin counts). Every other analysis knob (k,
    replicates, bootstraps, ...) is held fixed so the only thing that changes
    between panels is dz_nm.

    Each width gets its own subdirectory (so its normal figure set is not
    overwritten by the next width), plus one combined comparison figure is
    written to `out_dir` (default: results_dir).
    """
    frdir = Path(results_dir)
    outdir = Path(out_dir) if out_dir else frdir
    outdir.mkdir(parents=True, exist_ok=True)

    scan, failures = {}, {}
    for factor in width_factors:
        dz = base_dz_nm * factor
        sub_out = outdir / "dz_scan" / f"x{factor:g}_dz{dz:.4f}"
        print(f"\n=== dz_nm scan: factor {factor:g} -> dz_nm = {dz:.4f} nm ===")
        try:
            res = run_analysis(
                salt_data_path=salt_data_path,
                results_dir=results_dir,
                ion1=ion1,
                ion2=ion2,
                molality=molality,
                N_replicates=N_replicates,
                dz_nm=dz,
                n_bootstraps=n_bootstraps,
                out_dir=sub_out,
                **run_analysis_kwargs,
            )
        except Exception as exc:
            # A too-fine dz_nm can leave the weighted eq-12 fit numerically
            # singular, which then poisons the ML optimizer it seeds - this
            # can raise anywhere downstream (curve_fit, a NaN-filled
            # histogram, ...). Skip that width rather than losing every
            # other width's results.
            print(f"  FAILED at factor {factor:g} (dz_nm={dz:.4f}): "
                  f"{type(exc).__name__}: {exc}")
            failures[factor] = str(exc)
            continue
        scan[factor] = res

    if failures:
        print(f"\n{len(failures)} width(s) failed and were skipped: "
              f"{sorted(failures, reverse=True)}")
    if not scan:
        raise RuntimeError("Every width in the scan failed - nothing to plot.")

    factors_sorted = sorted(scan.keys(), reverse=True)
    n = len(factors_sorted)
    cmap = plt.get_cmap("viridis")

    # ------------------------------------------------------------------
    # One figure per method: every surviving width overlaid, each with its
    # own +/-1 sigma bootstrap band, so within-method width-sensitivity can
    # be judged directly against the bootstrap noise at that width.
    # ------------------------------------------------------------------
    methods = [
        ("unweighted", "c_unweighted", "oc_unweighted", "std_unweighted"),
        ("weighted", "c_weighted", "oc_weighted", "std_weighted"),
        ("maximum likelihood", "c_ml", "oc_ml", "std_ml"),
    ]
    for label, ckey, ockey, stdkey in methods:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, factor in enumerate(factors_sorted):
            res = scan[factor]
            color = cmap(i / max(1, n - 1))
            c, oc, std = res[ckey], res[ockey], res[stdkey]
            ax.plot(c, oc, lw=2, color=color,
                    label=f"dz={res['dz_nm']:.4f} nm (x{factor:g})")
            ax.fill_between(c, oc - std, oc + std, color=color, alpha=0.15)
        ax.plot(scan[factors_sorted[0]]["exp_molality"],
                scan[factors_sorted[0]]["exp_oc"],
                "ko-", ms=4, lw=1, label="Experimental")
        ax.set_xlabel("Concentration (mol/kg)")
        ax.set_ylabel("Osmotic coefficient")
        ax.set_title(f"{label.capitalize()} osmotic coefficient vs histogram "
                     f"bin width - {ion1}{ion2}\n(shaded = +/-1 bootstrap sigma)")
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        safe_label = label.replace(" ", "_")
        fig.savefig(f"{outdir}/dz_width_scan_{safe_label}_{ion1}{ion2}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Quantitative check: is the width-to-width shift bigger than the noise?
    # Interpolate each width's curve onto the narrowest width's own points,
    # combine 1-sigma bootstrap uncertainties in quadrature, and compare.
    # ------------------------------------------------------------------
    print("\nWidth-to-width shift vs combined bootstrap noise "
          "(1 = shift is within noise, higher = outside noise):")
    for label, ckey, ockey, stdkey in methods:
        print(f"  {label}:")
        for factor_a, factor_b in zip(factors_sorted[:-1], factors_sorted[1:]):
            res_a, res_b = scan[factor_a], scan[factor_b]
            c_a, oc_a, std_a = res_a[ckey], res_a[ockey], res_a[stdkey]
            c_b, oc_b, std_b = res_b[ckey], res_b[ockey], res_b[stdkey]
            lo = max(c_a.min(), c_b.min())
            hi = min(c_a.max(), c_b.max())
            grid = np.linspace(lo, hi, 200)
            oc_a_i = np.interp(grid, c_a, oc_a)
            oc_b_i = np.interp(grid, c_b, oc_b)
            std_a_i = np.interp(grid, c_a, std_a)
            std_b_i = np.interp(grid, c_b, std_b)
            shift = np.abs(oc_a_i - oc_b_i)
            combined_sigma = np.sqrt(std_a_i ** 2 + std_b_i ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = shift / combined_sigma
            ratio = ratio[np.isfinite(ratio)]
            print(f"    x{factor_a:g} vs x{factor_b:g}: "
                  f"median shift={np.median(shift):.4f}, "
                  f"median shift/sigma={np.median(ratio):.2f}, "
                  f"max shift/sigma={np.max(ratio):.2f}")

    print(f"\ndz_nm width scan complete. Combined figures in {outdir}")
    return scan