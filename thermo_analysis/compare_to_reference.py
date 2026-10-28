#!/usr/bin/env python
"""
compare_to_reference.py
=======================
Overlay your HP results (thermo analysis) on a literature reference, quantity by quantity, plus residuals.

The reference is a module with a dict REFERENCE (default: run_scripts/literature_reference_nacl_chi0.py =
Hosseini & Ashbaugh 2023, NaCl, chi = 0, SI Tables S3/S4). From its modDH parameters and eq-22 densities the
script builds the reference curves analytically, and it RECONSTRUCTS the concentration / water / density
profiles for the k, box and ion count of each of YOUR runs (the chemical-potential relation is intrinsic,
so this is valid for any k; the peak only equals the paper's ~3.5 M when your k and N match the paper's).

Figures (written to --out, default results/_comparisons/vs_reference/):
  op_vs_molarity   Pi(C): your fits, reference (+ optional digitized curve)
  op_vs_molality   Pi(m): molar fit and molal description, reference both ways, experiment
  oc               osmotic coefficient
  delta_mu_ex      mu_s^ex(C) - mu_s^ex(C_max): measured DATA POINTS (needs --traj), your fits, reference
  ln_gamma         ln gamma+-(m)
  mu_water         mu_w - mu_w0 = RT ln a_w (m)
  density          rho(m) at constant mu_w and constant P
  pmv_water        partial molar volume of water
  profiles         m(z) (linear/log); with --traj also rho(z) and C_w(z); reference reconstructed per run
  residuals        yours - reference for each quantity
  summary.csv      RMS and max |difference| per run and quantity

Usage (from thermo_analysis/):
  python compare_to_reference.py                       # default: the NaCl / TIP4P/2005 "new" runs that exist
  python compare_to_reference.py "LABEL=results/<case>/new,traj=HP_results/<case>/result_files_35m,salt=structures/salt_data_30x30x100.py" ...
  python compare_to_reference.py --fits all --digitized HP_results/HP_NaCl_TIP4P2005_paperbox/result_files_35m/nacl_chi0_digitized.csv

A run spec is LABEL=FOLDER[,traj=DIR][,salt=SALT_DATA.py]. FOLDER holds the outputs of one analysis
(curves_*.npz, thermo_curves_*.npz, thermo_params_*.json, concprof_*.npz). Without traj the measured density /
water profiles and the delta_mu_ex data points are skipped. salt_data supplies box, T, molar mass.
Baseline for "the old method": point a run at results/<case>/legacy_check.
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import HP_analysis_replicates_thermo as hp                                   # noqa: E402

R = hp.R
NA = hp.NA
MW = hp.M_WATER_G_PER_MOL
RUN_COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
FIT_STYLE = {"maximum_likelihood": "-", "weighted": "--", "unweighted": ":"}
REF_COLOR = "crimson"

DEFAULT_RUNS = [
    ("TIP4P/2005 paperbox (new)", "results/HP_NaCl_TIP4P2005_paperbox/new",
     "HP_results/HP_NaCl_TIP4P2005_paperbox/result_files_35m", "structures/salt_data_30x30x100.py"),
    ("TIP4P/2005 refpaper (new)", "results/HP_NaCl_TIP4P2005_refpaper/new",
     "HP_results/HP_NaCl_TIP4P2005_refpaper/result_files_35m", "structures/salt_data_30x30x100_refpaper.py"),
]


# ============================================================================
# reference curves (analytic, from the literature parameters)
# ============================================================================
def load_reference(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reference_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = mod.REFERENCE
    for key in ("molar", "molal", "rho_mu", "rho_P"):
        if key not in ref:
            raise SystemExit(f"reference {path} lacks '{key}' (needs molar, molal, rho_mu, rho_P)")
    return ref


def density_to_c(m, rho_coeffs, Ms):
    """Salt molarity of a solution of molality m whose density is eq-22(m): c = m rho / (1 + m Ms/1000)."""
    m = np.asarray(m, dtype=float)
    return m * hp.density_eq22(m, rho_coeffs) / (1.0 + m * Ms / 1000.0)


def reference_curves(ref, Ms, nu, T, m_grid):
    """Everything the molal / molar reference gives as a function of molality."""
    out = {}
    out["rho_mu"] = hp.density_eq22(m_grid, ref["rho_mu"])
    out["rho_P"] = hp.density_eq22(m_grid, ref["rho_P"])
    out["Vw"] = hp.partial_molar_volume_water(m_grid, ref["rho_P"], Ms)              # eq 24 from the S4 P fit
    Am, Bm, a1m, a2m = ref["molal"]
    bracket = m_grid + hp.dh_osmotic(m_grid, Am, Bm) + 0.5 * a1m * m_grid ** 2 + (2.0 / 3.0) * a2m * m_grid ** 3
    out["phi_molal"] = bracket / m_grid
    out["Pi_molal"] = 0.01 * nu * R * T * (MW / out["Vw"]) * bracket                 # constant-P description
    out["lng"] = hp.ln_gamma(m_grid, a1m, a2m, Am, Bm)
    out["rtlnaw"] = -nu * R * T * 1e-3 * (MW / 1000.0) * bracket
    # constant-mu_w (molar) description mapped onto molality through the S4 mu_w density
    c = density_to_c(m_grid, ref["rho_mu"], Ms)
    out["c_of_m"] = c
    out["Pi_molar"] = hp.pi_bar(c, *ref["molar"], nu, T)
    out["phi_molar"] = out["Pi_molar"] / (0.01 * nu * R * T * c)
    return out


def reconstruct_profile(z_nm, k, T, n_pairs, area_nm2, lz_nm, ref_molar, n_int=700):
    """Salt molarity C_s(z) [mol/L] of the reference chemical potential in YOUR harmonic restraint.

    ln(C/C0) + [ln gamma(C) - ln gamma(C0)] = -U(z)/RT,  U = k z^2 / 2  (k in kJ/mol/nm^2),
    and C0 is fixed by 2 * area * integral_0^(Lz/2) C dz * (NA 1e-24) = n_pairs.
    """
    A, B, a1, a2 = ref_molar
    lng = lambda c: hp.ln_gamma(c, a1, a2, A, B)                                    # noqa: E731

    def c_of_z(zv, c0):
        if zv <= 0:
            return c0
        target = -(0.5 * k * zv ** 2 * 1000.0 / (R * T))
        f = lambda c: np.log(c / c0) + lng(c) - lng(c0) - target                    # noqa: E731
        return brentq(f, 1e-14, c0, xtol=1e-12, rtol=1e-12)

    zz = np.linspace(0.0, lz_nm / 2.0, n_int)

    def n_calc(c0):
        cz = np.array([c_of_z(zv, c0) for zv in zz])
        return 2.0 * area_nm2 * (NA * 1e-24) * simpson(cz, x=zz)

    c0 = brentq(lambda c0: n_calc(c0) - n_pairs, 1e-4, 30.0, xtol=1e-9)
    return np.array([c_of_z(zv, c0) for zv in np.asarray(z_nm, float)]), c0


def profile_quantities(c_s, ref, Ms):
    """m(z), rho(z), C_w(z) of the reference from its C_s(z) (inverts c = m rho(m)/(1 + m Ms/1000))."""
    m_tab = np.linspace(0.0, 12.0, 6001)
    c_tab = density_to_c(m_tab, ref["rho_mu"], Ms)
    m = np.interp(c_s, c_tab, m_tab)
    rho = hp.density_eq22(m, ref["rho_mu"])
    kgw = rho / (1.0 + m * Ms / 1000.0)
    return m, rho, kgw * 1000.0 / MW


# ============================================================================
# loading one run of yours
# ============================================================================
class Run:
    pass


def _one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"{what}: nothing matches {pattern}")
    return hits[0]


def parse_spec(spec):
    label, rest = spec.split("=", 1)
    parts = [p.strip() for p in rest.split(",")]
    kw = {"folder": parts[0]}
    for p in parts[1:]:
        key, val = p.split("=", 1)
        kw[key.strip()] = val.strip()
    return label.strip(), kw


def load_run(label, folder, traj=None, salt=None, dz_nm=0.05, default_salt="structures/salt_data_30x30x100.py"):
    r = Run()
    r.label, r.folder = label, folder
    r.curves = np.load(_one(f"{folder}/curves_*.npz", f"{label} curves"), allow_pickle=True)
    r.thermo = np.load(_one(f"{folder}/thermo_curves_*.npz", f"{label} thermo_curves"), allow_pickle=True)
    r.params = json.load(open(_one(f"{folder}/thermo_params_*.json", f"{label} thermo_params")))
    cp = glob.glob(f"{folder}/concprof_*.npz")
    r.concprof = np.load(cp[0], allow_pickle=True) if cp else None
    r.salt = str(r.curves["salt"])
    r.k = float(r.curves["k"])
    r.n_pairs = int(r.curves["n_pairs"])
    r.molality = float(r.curves["molality"])
    r.exp_m, r.exp_oc = np.asarray(r.curves["exp_molality"], float), np.asarray(r.curves["exp_oc"], float)
    r.SD = hp.load_salt_data(salt or default_salt)
    r.Ms = r.SD.MOLAR_MASS[r.salt]
    r.nu = getattr(r.SD, "VANT_HOFF", {}).get(r.salt, 2)
    r.T = r.SD.BOX["temperature"]
    r.lz, r.area = r.SD.BOX["lz"], r.SD.BOX["area"]
    r.prof = None
    r.traj = traj
    r.reservoir = list(r.params.get("reservoir_density_g_cm3") or [])      # as saved by the analysis run
    if traj:
        r.prof, r.area = measure(traj, r, dz_nm)
        r.reservoir = list(hp.reservoir_density(r.prof))                    # measured now (edge-bin wrap fixed)
    return r


def measure(traj, r, dz_nm):
    import re
    import MDAnalysis as mda
    ion1, ion2 = re.findall(r"[A-Z][a-z]*", r.salt)[:2]
    mi, _ = hp._format_molality(r.molality)
    n_rep = int(r.curves["n_replicates"])
    us, groups, areas = [], [], []
    for i in range(n_rep):
        u = mda.Universe(f"{traj}/md{mi}m_r{i}.pdb", f"{traj}/md{mi}m_r{i}.xtc")
        us.append(u)
        groups.append((hp.select_ion(u, ion1.upper(), ion1), hp.select_ion(u, ion2.upper(), ion2)))
        areas.append(hp.get_average_cross_section_area(u))
    prof = hp.measured_profiles(us, groups, dz_nm, r.Ms)
    return prof, float(np.mean(areas))


# ============================================================================
# our curves, per fit
# ============================================================================
def fit_names(sel):
    allf = ["maximum_likelihood", "weighted", "unweighted"]
    return allf if sel == "all" else [sel]


def get(r, fit, key):
    return np.asarray(r.thermo[f"{fit}_{key}"], float)


def delta_mu_points(r):
    """-U/RT - ln(C/C_max) = ln gamma(C) - ln gamma(C_max): measured, no fit involved."""
    z, c = r.prof["z"], r.prof["c_s"]
    ok = np.isfinite(c) & (c > 1e-4)
    cmax = float(np.max(c[ok]))
    y = -(0.5 * r.k * z[ok] ** 2 * 1000.0 / (R * r.T))
    return c[ok], y - np.log(c[ok] / cmax), cmax


# ============================================================================
# figures
# ============================================================================
def _finish(ax, xlabel, ylabel, title=None, legend=True):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7)


def make_figures(runs, ref, fits, out, digitized=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rlabel = ref.get("label", "reference")
    summary = []
    r0 = runs[0]
    m_hi = max(float(r.params["m_max"]) for r in runs)
    m_grid = np.linspace(0.02, m_hi, 300)
    refc = reference_curves(ref, r0.Ms, r0.nu, r0.T, m_grid)
    stamp = "reservoir $\\rho_w$: " + "; ".join(
        f"{r.label.split('(')[0].strip()} {np.mean(r.reservoir):.4f}" for r in runs if r.reservoir)

    def save(fig, name):
        fn = out / f"{name}.png"
        fig.tight_layout(); fig.savefig(fn, dpi=170); plt.close(fig); print(f"  wrote {fn}")

    def stampit(ax):
        if stamp.endswith(": "):
            return
        ax.text(0.02, 0.02, stamp, transform=ax.transAxes, fontsize=6, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    def over(ax, run_i, fit, x, y, **kw):
        ax.plot(x, y, c=RUN_COLORS[run_i % len(RUN_COLORS)], ls=FIT_STYLE[fit],
                lw=2 if fit == "maximum_likelihood" else 1.2, **kw)

    def label_of(r, fit):
        return r.label if len(fits) == 1 else f"{r.label} [{fit.replace('_', ' ')}]"

    # ---- Pi vs molarity
    fig, ax = plt.subplots(figsize=(7, 5))
    cc = np.linspace(0.02, max(get(r, f, "c_molar").max() for r in runs for f in fits), 300)
    ax.plot(cc, hp.pi_bar(cc, *ref["molar"], r0.nu, r0.T), "--", c=REF_COLOR, lw=2, label=f"{rlabel} (molar fit)")
    if digitized:
        d = np.genfromtxt(digitized, delimiter=",", names=True, dtype=None, encoding=None)
        ax.plot(d["C_s_M"], d["Pi_bar"], "o", ms=3, c="k", label="digitized from the paper")
    for i, r in enumerate(runs):
        for f in fits:
            over(ax, i, f, get(r, f, "c_molar"), get(r, f, "op_molar"), label=label_of(r, f))
    _finish(ax, "concentration $C_s$ (mol/L)", "osmotic pressure (bar)", "Osmotic pressure vs molarity"); stampit(ax)
    save(fig, "op_vs_molarity")

    # ---- Pi vs molality
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r0.exp_m, [0.01 * r0.nu * R * r0.T * m * 0.99705 * p for m, p in zip(r0.exp_m, r0.exp_oc)],
            "ko-", ms=3, label="experiment")
    ax.plot(m_grid, refc["Pi_molar"], "--", c=REF_COLOR, lw=2, label=f"{rlabel} (molar fit via $\\rho_{{\\mu_w}}$)")
    ax.plot(m_grid, refc["Pi_molal"], ":", c=REF_COLOR, lw=2, label=f"{rlabel} (molal description)")
    for i, r in enumerate(runs):
        for f in fits:
            over(ax, i, f, r.thermo["m_grid"], get(r, f, "Pi"), label=label_of(r, f) + " molar fit")
            ax.plot(r.thermo["m_grid"], get(r, f, "op_molal"), c=RUN_COLORS[i % len(RUN_COLORS)], ls="-.", lw=1,
                    label=label_of(r, f) + " molal description")
    ax.set_xlim(0, m_hi * 1.05)
    _finish(ax, "molality (mol/kg)", "osmotic pressure (bar)", "Osmotic pressure vs molality"); stampit(ax)
    save(fig, "op_vs_molality")

    # ---- osmotic coefficient
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r0.exp_m, r0.exp_oc, "ko-", ms=3, label="experiment")
    ax.plot(m_grid, refc["phi_molar"], "--", c=REF_COLOR, lw=2, label=f"{rlabel} (molar fit)")
    ax.plot(m_grid, refc["phi_molal"], ":", c=REF_COLOR, lw=2, label=f"{rlabel} (molal description)")
    for i, r in enumerate(runs):
        for f in fits:
            over(ax, i, f, r.thermo["m_grid"], get(r, f, "Pi_molar_phi"), label=label_of(r, f))
    ax.set_xlim(0, m_hi * 1.05)
    _finish(ax, "molality (mol/kg)", "osmotic coefficient", "Osmotic coefficient"); stampit(ax)
    save(fig, "oc")

    # ---- delta mu_s^ex (data points + fits + reference)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, r in enumerate(runs):
        col = RUN_COLORS[i % len(RUN_COLORS)]
        if r.prof is not None:
            c, dmu, cmax = delta_mu_points(r)
            ax.plot(c, dmu, ".", ms=4, c=col, alpha=0.6, label=f"{r.label}: measured (no fit)")
        else:
            cmax = float(np.max(get(r, fits[0], "c_molar")))
        for f in fits:
            cg, mu = get(r, f, "c_molar"), get(r, f, "mu_s_ex")
            ax.plot(cg, mu - np.interp(cmax, cg, mu), c=col, ls=FIT_STYLE[f], lw=1.5, label=label_of(r, f) + " fit")
        cref = np.linspace(0.02, cmax, 200)
        ax.plot(cref, hp.ln_gamma(cref, *ref["molar"][2:], ref["molar"][0], ref["molar"][1])
                - float(hp.ln_gamma(np.array([cmax]), *ref["molar"][2:], ref["molar"][0], ref["molar"][1])[0]),
                "--", c=col, lw=1, alpha=0.9, label=f"{rlabel}, aligned at this run's $C_{{max}}$")
    ax.axhline(0, c="k", lw=.5)
    _finish(ax, "concentration $C_s$ (mol/L)", "$\\mu_s^{ex}(C)-\\mu_s^{ex}(C_{max})$ / RT",
            "Salt excess chemical potential (only differences are measurable)"); stampit(ax)
    save(fig, "delta_mu_ex")

    # ---- ln gamma, mu_w, density, V_w
    def simple(name, ykey, refkey, ylabel, title, xkey="m_grid", sqrt_x=False, exp=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = np.sqrt(m_grid) if sqrt_x else m_grid
        if exp is not None:
            ax.plot(*exp, "ko-", ms=3, label="experiment")
        ax.plot(xs, refc[refkey], "--", c=REF_COLOR, lw=2, label=rlabel)
        for i, r in enumerate(runs):
            for f in fits:
                x = np.sqrt(r.thermo["m_grid"]) if sqrt_x else r.thermo["m_grid"]
                over(ax, i, f, x, get(r, f, ykey), label=label_of(r, f))
        _finish(ax, "$\\sqrt{m}$ (mol/kg)$^{1/2}$" if sqrt_x else "molality (mol/kg)", ylabel, title); stampit(ax)
        save(fig, name)

    simple("ln_gamma", "lng", "lng", "$\\ln\\gamma_\\pm$", "Mean ionic activity coefficient (molal description)")
    em = r0.exp_m
    simple("mu_water", "rtlnaw", "rtlnaw", "$\\mu_w-\\mu_w^0$ = RT ln $a_w$ (kJ/mol)", "Water chemical potential",
           sqrt_x=True, exp=(np.sqrt(em), -r0.nu * R * r0.T * 1e-3 * (MW / 1000.0) * em * r0.exp_oc))
    simple("pmv_water", "Vw", "Vw", "$\\bar V_w$ (cm$^3$/mol)", "Partial molar volume of water")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m_grid, refc["rho_mu"], "--", c=REF_COLOR, lw=2, label=f"{rlabel}, constant $\\mu_w$")
    ax.plot(m_grid, refc["rho_P"], ":", c=REF_COLOR, lw=2, label=f"{rlabel}, constant P")
    for i, r in enumerate(runs):
        col = RUN_COLORS[i % len(RUN_COLORS)]
        ax.plot(r.thermo["rho_profile_m"], r.thermo["rho_profile"], ".", ms=3, c=col, alpha=0.35,
                label=f"{r.label}: measured")
        ax.plot(m_grid, hp.density_eq22(m_grid, r.thermo["coeffs_mu"]), c=col, lw=2, label=f"{r.label}: eq 22, $\\mu_w$")
        for f in fits:
            ax.plot(m_grid, hp.density_eq22(m_grid, get(r, f, "coeffs_P")), c=col, ls="--", lw=1,
                    label=label_of(r, f) + ": constant P")
    _finish(ax, "molality (mol/kg)", "density (g/cm$^3$)", "Solution density"); stampit(ax)
    save(fig, "density")

    # ---- profiles
    have_traj = [r for r in runs if r.prof is not None]
    npan = 4 if have_traj else 2
    fig, axes = plt.subplots(1, npan, figsize=(4.6 * npan, 4.6))
    axes = np.atleast_1d(axes)
    for i, r in enumerate(runs):
        col = RUN_COLORS[i % len(RUN_COLORS)]
        zr = np.linspace(0.0, r.lz / 2, 160)
        c_ref, c0 = reconstruct_profile(zr, r.k, r.T, r.n_pairs, r.area, r.lz, ref["molar"])
        m_ref, rho_ref, cw_ref = profile_quantities(c_ref, ref, r.Ms)
        tag = f"{rlabel}, k={r.k:.3g}, N={r.n_pairs}, C0={c0:.2f} M"
        if r.concprof is not None:
            for a in axes[:2]:
                a.plot(r.concprof["z_salt"], r.concprof["mean_salt"], c=col, lw=2, label=f"{r.label}: measured")
        for a in axes[:2]:
            a.plot(zr, m_ref, "--", c=col, lw=1.2, label=tag)
        if r.prof is not None and npan == 4:
            axes[2].plot(r.prof["z"], r.prof["rho"], c=col, lw=2, label=f"{r.label}: measured")
            axes[3].plot(r.prof["z"], r.prof["c_w"], c=col, lw=2, label=f"{r.label}: measured")
        if npan == 4:
            axes[2].plot(zr, rho_ref, "--", c=col, lw=1.2, label=tag)
            axes[3].plot(zr, cw_ref, "--", c=col, lw=1.2, label=tag)
    axes[1].set_yscale("log"); axes[1].set_ylim(1e-3, None)
    titles = ["local molality m(z)", "m(z), log", "solution density $\\rho$(z)", "water concentration $C_w$(z)"]
    ylabels = ["molality (mol/kg)", "molality (mol/kg)", "density (g/cm$^3$)", "$C_w$ (mol/L)"]
    for a, t, yl in zip(axes, titles, ylabels):
        _finish(a, "z (nm)", yl, t, legend=(a is axes[0]))
    stampit(axes[0])
    save(fig, "profiles")

    # ---- residuals + summary
    quant = [
        ("Pi(C) molar fit (bar)", "C", lambda r, f: (get(r, f, "c_molar"), get(r, f, "op_molar")),
         lambda x: hp.pi_bar(x, *ref["molar"], r0.nu, r0.T)),
        ("phi(m) molar fit", "m", lambda r, f: (r.thermo["m_grid"], get(r, f, "Pi_molar_phi")),
         lambda x: np.interp(x, m_grid, refc["phi_molar"])),
        ("ln gamma(m) molal", "m", lambda r, f: (r.thermo["m_grid"], get(r, f, "lng")),
         lambda x: np.interp(x, m_grid, refc["lng"])),
        ("RT ln a_w (kJ/mol)", "m", lambda r, f: (r.thermo["m_grid"], get(r, f, "rtlnaw")),
         lambda x: np.interp(x, m_grid, refc["rtlnaw"])),
        ("rho(m) constant mu_w", "m", lambda r, f: (r.thermo["m_grid"], hp.density_eq22(r.thermo["m_grid"], r.thermo["coeffs_mu"])),
         lambda x: np.interp(x, m_grid, refc["rho_mu"])),
        ("V_w(m) (cm3/mol)", "m", lambda r, f: (r.thermo["m_grid"], get(r, f, "Vw")),
         lambda x: np.interp(x, m_grid, refc["Vw"])),
        ("mu_s^ex(C) aligned at C_max", "C", None, None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for qi, (name, xk, ours, reff) in enumerate(quant):
        ax = axes[qi]
        for i, r in enumerate(runs):
            for f in fits:
                if ours is None:
                    cg, mu = get(r, f, "c_molar"), get(r, f, "mu_s_ex")
                    cmax = float(cg.max())
                    A_, B_, a1_, a2_ = ref["molar"]
                    x = cg
                    y_ours = mu - np.interp(cmax, cg, mu)
                    y_ref = hp.ln_gamma(cg, a1_, a2_, A_, B_) - float(hp.ln_gamma(np.array([cmax]), a1_, a2_, A_, B_)[0])
                else:
                    x, y_ours = ours(r, f)
                    y_ref = reff(x)
                ok = np.isfinite(y_ours) & np.isfinite(y_ref) & (np.asarray(x) > (0.05 if xk == "m" else 0.05))
                d = np.asarray(y_ours)[ok] - np.asarray(y_ref)[ok]
                ax.plot(np.asarray(x)[ok], d, c=RUN_COLORS[i % len(RUN_COLORS)], ls=FIT_STYLE[f],
                        lw=2 if f == "maximum_likelihood" else 1.1, label=label_of(r, f))
                if d.size:
                    summary.append([r.label, f, name, f"{np.sqrt(np.mean(d ** 2)):.6g}", f"{np.max(np.abs(d)):.6g}"])
        ax.axhline(0, c="k", lw=.7)
        _finish(ax, "$C_s$ (mol/L)" if xk == "C" else "molality (mol/kg)", f"yours $-$ reference: {name}", legend=(qi == 0))
    for ax in axes[len(quant):]:
        ax.axis("off")
    fig.suptitle(f"Residuals against {rlabel}", fontsize=12)
    save(fig, "residuals")

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "fit", "quantity", "rms_difference", "max_abs_difference"])
        w.writerows(summary)
    print(f"  wrote {out / 'summary.csv'}")
    return summary


# ============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help='"LABEL=FOLDER[,traj=DIR][,salt=SALT_DATA.py]" (default: the NaCl TIP4P/2005 new runs)')
    p.add_argument("--reference", default=str(HERE / "run_scripts" / "literature_reference_nacl_chi0.py"))
    p.add_argument("--fits", default="maximum_likelihood",
                   choices=["maximum_likelihood", "weighted", "unweighted", "all"])
    p.add_argument("--dz", type=float, default=0.05, help="bin width (nm) for the measured profiles")
    p.add_argument("--digitized", default=None, help="CSV with columns C_s_M,Pi_bar (curve digitized from the paper)")
    p.add_argument("--no-traj", action="store_true", help="skip trajectories: no measured density/water profiles")
    p.add_argument("-o", "--out", default="results/_comparisons/vs_reference")
    a = p.parse_args(argv)

    specs = []
    if a.runs:
        for s in a.runs:
            label, kw = parse_spec(s)
            specs.append((label, kw["folder"], kw.get("traj"), kw.get("salt")))
    else:
        for label, folder, traj, salt in DEFAULT_RUNS:
            if glob.glob(f"{folder}/thermo_curves_*.npz"):
                specs.append((label, folder, traj if os.path.isdir(traj) else None, salt))
        if not specs:
            raise SystemExit("no default run found (results/HP_NaCl_TIP4P2005_*/new/thermo_curves_*.npz); pass run specs")
    ref = load_reference(a.reference)
    runs = []
    for label, folder, traj, salt in specs:
        print(f"loading {label}: {folder}" + (f" + trajectories {traj}" if traj and not a.no_traj else ""))
        runs.append(load_run(label, folder, None if a.no_traj else traj, salt, a.dz))
    summary = make_figures(runs, ref, fit_names(a.fits), a.out, a.digitized)
    print("\nRMS difference vs reference (maximum likelihood unless --fits):")
    for row in summary:
        if row[1] == fit_names(a.fits)[0]:
            print(f"  {row[0]:<28s} {row[2]:<32s} rms {row[3]:>10s}  max {row[4]:>10s}")


if __name__ == "__main__":
    main()
