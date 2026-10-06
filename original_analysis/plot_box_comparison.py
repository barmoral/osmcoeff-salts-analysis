"""
plot_box_comparison.py
======================
Overlay the HP osmotic-coefficient curves from different box lengths.
Three panels - unweighted / weighted / maximum likelihood - one colour per box.

Sibling of plot_alpha_comparison.py: that one holds the box fixed and varies
the number of fitting terms, this one holds the terms fixed and varies the box.

    conda activate osmotic-analysis-hp
    cd /media/bamo6610/DATA/osmcoeff-salts-analysis/original_analysis

    # original vs clone
    python plot_box_comparison.py "14.4 nm (original)=HP_results/HP_NaCl_TIP3P" "28.8 nm (clone)=HP_results/HP_NaCl_TIP3P_dbox"
    python plot_box_comparison.py  "28.8 nm (clone)=HP_NaCl_TIP3P_dbox" "18.0 nm (clone)=HP_NaCl_TIP3P_lcbox" "14.4 nm (original)=HP_NaCl_TIP3P" 

    # add the non-clone 28.8 nm box, pin the term count, save elsewhere
    python plot_box_comparison.py \
        "14.4 nm=HP_results/HP_NaCl_TIP3P" \
        "28.8 nm clone=HP_results/HP_NaCl_TIP3P_dbox" \
        "28.8 nm designed=HP_results/HP_NaCl_TIP3P_designed" \
        --alphas 2 -o figs/box_comparison.png

Each argument is LABEL=PATH, where PATH is the case directory. Below it the
script finds result_files*/<N>alphas/curves_*.npz on its own, so it does not
care what the k tag is or whether the layout is nested or flat.

--alphas is optional and defaults to 2. A run sitting directly in
result_files* with no <N>alphas/ subdirectory is treated as a 2-alpha run,
which is the layout of the boxes that were analysed before the sweep
directories existed. So

    HP_NaCl_TIP3P/result_files_k0p661_35m/2alphas/curves_*.npz   -> 2
    HP_NaCl_TIP3P_dbox/result_files_k0p661_35m/curves_*.npz      -> 2

compare directly with no flags. The count must match across boxes: a 1-alpha
fit against a 3-alpha fit produces a gap that looks like physics and isn't.

WHAT TO LOOK FOR
----------------
The curves should lie on top of each other. They are the same salt at the same
molality; box length is a numerical convergence parameter, not physics. A gap
between 14.4 and 28.8 clone means the short box was starving its reservoirs -
which is the whole reason the clone exists.

The designed 28.8 box is a different comparison: it has a softer k and twice
the ions, so it probes whether the METHOD is invariant when the restraint is
re-derived properly. Read it separately, not as a third repeat of the clone.

The script prints Lz, k, n_pairs and n_water for each box so you can confirm
you are comparing what you think you are - the 14.4 box and the 28.8 clone
share salt, molality, k AND n_pairs, so the water count is the only thing that
tells them apart.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

FITS = ["unweighted", "weighted", "maximum_likelihood"]

COLORS = ["maroon", "dodgerblue", "orange"]

# One linestyle per fitting mode, so a combined figure can encode box (color)
# and fit (linestyle) independently instead of needing len(boxes)*len(FITS)
# distinct colors.
LINESTYLES = {"unweighted": ":", "weighted": "--", "maximum_likelihood": "-"}


_ALPHA_RE = re.compile(r"(\d+)alphas")

# A run sitting directly in result_files* with no <N>alphas/ subdirectory is a
# 2-alpha run - that was the setting before the sweep directories existed.
DEFAULT_ALPHAS = 2


def available_alphas(case_dir):
    """{alpha_count: npz_path} for everything found under a case directory.

    Covers both layouts without caring which is which:
      <case>/result_files*/<N>alphas/curves_*.npz    (an alpha sweep)
      <case>/result_files*/curves_*.npz             (a single run, no subdir)

    The count comes from n_expansion_terms INSIDE each npz, which is what
    run_analysis was actually called with. Reading it from the path instead
    would trust a directory name someone typed; _ALPHA_RE is kept only as a
    fallback for a file too old or too broken to open.
    """
    hits = (glob.glob(os.path.join(case_dir, "result_files*", "curves_*.npz"))
            + glob.glob(os.path.join(case_dir, "result_files*", "*alphas",
                                     "curves_*.npz")))
    found = {}
    for p in sorted(set(hits)):
        try:
            with np.load(p, allow_pickle=True) as d:
                n = int(d["n_expansion_terms"])
        except Exception as e:
            m = _ALPHA_RE.search(p)
            n = int(m.group(1)) if m else DEFAULT_ALPHAS
            print(f"  warn: could not read n_expansion_terms from "
                  f"{os.path.basename(p)} ({type(e).__name__}); "
                  f"assuming {n} from the path")
        found.setdefault(n, p)
    return found


def choose_alphas(avail, n_alphas):
    """Resolve which alpha count to read from every box. Default DEFAULT_ALPHAS.

    The count must be the same everywhere: comparing a 1-alpha fit in one box
    against a 3-alpha fit in another produces a gap that looks like physics and
    isn't.
    """
    for label, a in avail.items():
        if not a:
            raise SystemExit(
                f"no curves_*.npz found for '{label}'.\n"
                f"  Has this run been re-analysed since the np.savez block was"
                f" added to run_analysis()?")

    want = DEFAULT_ALPHAS if n_alphas is None else n_alphas
    missing = [l for l, a in avail.items() if want not in a]
    if missing:
        how = "default" if n_alphas is None else "requested"
        raise SystemExit(
            f"{want} alphas ({how}) not available for: {', '.join(missing)}\n"
            + "\n".join(f"  {l}: has {sorted(a)}" for l, a in avail.items())
            + "\n  Pass --alphas N to pick a count they all have.")

    if n_alphas is None:
        extra = sorted(set().union(*(set(a) for a in avail.values())) - {want})
        print(f"  --alphas not given; using {want}"
              + (f" (also available somewhere: {extra})" if extra else ""))
    return want


def _sem(d, fit):
    """Standard error of the mean for a fit's osmotic-coefficient curve.

    std_<fit> in the npz is the replicate-to-replicate spread; dividing by
    sqrt(n_replicates) turns that into the uncertainty on the mean curve
    itself, which is what a band around the mean should show. Falls back to
    the raw std (with a warning) for older npz files saved before
    n_replicates was recorded.
    """
    std = d[f"std_{fit}"]
    if "n_replicates" in d.files:
        return std / np.sqrt(float(d["n_replicates"]))
    print(f"  no n_replicates saved (old run) - showing raw std instead of "
          f"the standard error of the mean")
    return std


def plot(pairs, n_alphas=None, out_png=None, show_exp=True, bands=True,
         ylim=(0.9, 1.3)):
    avail = {label: available_alphas(case) for label, case in pairs}
    n_alphas = choose_alphas(avail, n_alphas)

    data = {}
    for label, _case in pairs:
        d = np.load(avail[label][n_alphas], allow_pickle=True)
        data[label] = d
        lz = float(d["lz"]) if "lz" in d.files else float("nan")
        nw = float(d["n_water"]) if "n_water" in d.files else float("nan")
        print(f"  {label:<24} Lz={lz:5.1f} nm  k={float(d['k']):.4f}  "
              f"N={int(d['n_pairs']):4d}  waters={nw:.0f}  "
              f"[{int(d['n_expansion_terms'])} alphas]")

    labels = [l for l, _ in pairs]
    ref = data[labels[0]]
    salt, molality = str(ref["salt"]), float(ref["molality"])

    # Comparing different salts or molalities is almost always a mistake.
    for l in labels[1:]:
        if str(data[l]["salt"]) != salt or float(data[l]["molality"]) != molality:
            print(f"  WARNING: {l} is {data[l]['salt']} at "
                  f"{float(data[l]['molality'])} m, not {salt} at {molality} m")

    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)

    for ax, fit in zip(axes, FITS):
        for label, c in zip(labels, colors):
            d = data[label]
            x, y = d[f"c_{fit}"], d[f"oc_{fit}"]
            s = _sem(d, fit)
            ax.plot(x, y, lw=1.8, color=c,
                    label=f"{label} (mean sigma={np.mean(s):.4f})")
            if bands:
                ax.fill_between(x, y - s, y + s, color=c, alpha=0.35, lw=0)

        if show_exp and "exp_molality" in ref.files:
            ax.plot(ref["exp_molality"], ref["exp_oc"], "o-", color="k",
                    ms=4, lw=1.5, zorder=5, label="experiment")

        ax.set_title(fit.replace("_", " "))
        ax.set_xlabel("Concentration (mol/kg)")
        ax.grid(alpha=0.3)
        # sigma differs per fitting mode, so every panel gets its own legend
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Osmotic coefficient")
    axes[0].set_ylim(*ylim)
    axes[0].set_xlim(0, molality + 0.2)
    nterm = int(ref["n_expansion_terms"]) + 1
    fig.suptitle(f"{salt} at {molality} mol/kg,"
                 f"effect of box length", fontsize=13)
    fig.tight_layout()

    out_png = out_png or f"box_comparison_{salt}_{molality}m.png"
    d = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_png}")

    combined_png = _combined_name(out_png)
    _plot_combined(data, labels, colors, ref, salt, molality, show_exp,
                    bands, ylim, combined_png)
    print(f"  wrote {combined_png}")

    curve_dirs = {label: os.path.dirname(avail[label][n_alphas])
                  for label in labels}
    outputs = [out_png, combined_png]

    profile_png = _profiles_name(out_png)
    counts = _load_profile_npz(curve_dirs, salt, "countsprof",
                                "profile-comparison")
    if counts:
        _plot_profile_npz(
            counts, labels, colors, profile_png,
            ylabel="Average ions per bin",
            title=f"Count profiles for {salt} - effect of box length",
            bands=bands, legend_sigma=True)
        print(f"  wrote {profile_png}")
        outputs.append(profile_png)

    conclog_png = _concentration_log_name(out_png)
    conc = _load_profile_npz(curve_dirs, salt, "concprof", "log-concentration")
    if conc:
        _plot_profile_npz(
            conc, labels, colors, conclog_png,
            ylabel="Concentration (mol/kg)",
            title=f"Concentration profiles for {salt} - effect of box "
                  f"length (log scale, shading = SEM)",
            bands=bands, log_y=True,
            hline=molality, hline_label=f"target bulk = {molality} mol/kg",
            legend_sigma=True, legend_outside=True)
        print(f"  wrote {conclog_png}")
        outputs.append(conclog_png)

    return tuple(outputs)


def _combined_name(out_png):
    root, ext = os.path.splitext(out_png)
    return f"{root}_all_fits{ext}"


def _plot_combined(data, labels, colors, ref, salt, molality, show_exp,
                    bands, ylim, out_png):
    """All boxes x all fitting modes on one axes - 6 curves for 2 boxes.

    Color encodes the box, linestyle encodes the fitting mode, so the legend
    stays at len(labels) colors x len(FITS) styles instead of needing six
    distinct hues. Each legend entry carries the curve's mean SEM.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for label, c in zip(labels, colors):
        d = data[label]
        for fit in FITS:
            x, y = d[f"c_{fit}"], d[f"oc_{fit}"]
            s = _sem(d, fit)
            leg_label = (f"{label} - {fit.replace('_', ' ')} "
                         f"(mean sigma={np.mean(s):.4f})")
            if bands:
                ax.fill_between(x, y - s, y + s, color=c, alpha=0.35, lw=0)
            ax.plot(x, y, lw=1.8, color=c, ls=LINESTYLES[fit],
                     label=leg_label)

    if show_exp and "exp_molality" in ref.files:
        ax.plot(ref["exp_molality"], ref["exp_oc"], "o-", color="k",
                ms=4, lw=1.5, zorder=5, label="experiment")

    ax.set_xlabel("Concentration (mol/kg)")
    ax.set_ylabel("Osmotic coefficient")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, molality + 0.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Comparison of box lengths {salt} at {molality} mol/kg "
                 f"all fitting modes")
    fig.tight_layout()

    d = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _profiles_name(out_png):
    root, ext = os.path.splitext(out_png)
    return f"{root}_profiles{ext}"


def _concentration_log_name(out_png):
    root, ext = os.path.splitext(out_png)
    return f"{root}_concentration_log{ext}"


def _load_profile_npz(curve_dirs, salt, file_prefix, figure_desc):
    """{label: npz} for <file_prefix>_<salt>.npz next to each box's curves_*.npz.

    Both countsprof_*.npz (ion counts) and concprof_*.npz (molal
    concentration) are written by HP_analysis_replicates.py in that same
    layout, so this loader is shared between them. Returns None (having
    already printed why) if any box's run predates that npz being saved.
    """
    profiles = {}
    for label, cdir in curve_dirs.items():
        pattern = f"{file_prefix}_{salt}.npz"
        hits = glob.glob(os.path.join(cdir, pattern))
        if not hits:
            # The profile npz is written once per result_files* case, not
            # once per <N>alphas subdirectory - fall back to the parent dir
            # when curves_*.npz came from a nested alpha-sweep layout.
            hits = glob.glob(os.path.join(os.path.dirname(cdir), pattern))
        if not hits:
            print(f"  no {pattern} for '{label}' in {cdir} (or its parent) -"
                  f" skipping the {figure_desc} figure "
                  f"(re-run the analysis to produce it)")
            return None
        profiles[label] = np.load(hits[0], allow_pickle=True)
    return profiles


def _plot_profile_npz(profiles, labels, colors, out_png, ylabel, title,
                       bands=True, log_y=False, hline=None, hline_label=None,
                       legend_sigma=False, legend_outside=False):
    """Overlay ion1 / ion2 / per-salt-mean curves from a loaded profile npz set.

    Shared rendering for the count-profile and log-concentration figures -
    they differ only in which npz was loaded, the y-label/title, and whether
    the y-axis is log. z is folded about the box center (HP_analysis_
    replicates.fold_profile), so z=0 is mid-box and z increases outward
    toward the box edge - this is exactly the axis to read for edge effects:
    a profile that has not flattened to the target bulk concentration by the
    time it reaches the edge means the reservoir has not equilibrated, which
    throws off the reference chemical potential the whole osmotic-coefficient
    curve is built from.

    hline/hline_label draw a reference line (e.g. the target bulk molality)
    so a reader can see at a glance whether the profile has actually reached
    it by the edge, rather than eyeballing the curve against the y-axis.

    legend_sigma appends each curve's mean SEM (averaged over z) to its
    legend label, so the uncertainty already drawn as a shaded band also has
    a number attached to it.
    """
    figsize = (10.5, 5.5) if legend_outside else (7, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ref = profiles[labels[0]]
    ion1, ion2 = str(ref["ion1"]), str(ref["ion2"])

    def band(x, y, s, c):
        if bands and s is not None:
            ax.fill_between(x, y - s, y + s, color=c, alpha=0.35, lw=0)

    for label, c in zip(labels, colors):
        p = profiles[label]
        has_std = "std_ion1" in p.files

        # The mean curve's own uncertainty is the standard error of the
        # mean (std / sqrt(n_replicates)), not the replicate-to-replicate
        # std itself - the latter describes spread across replicates, not
        # how well the mean is known. Falls back to raw std (with a
        # warning) for older npz files saved before n_replicates was
        # recorded.
        if has_std and "n_replicates" in p.files:
            sem_scale = 1.0 / np.sqrt(float(p["n_replicates"]))
        elif has_std:
            sem_scale = 1.0
            print(f"  no n_replicates saved for '{label}' (old run) - "
                  f"showing raw std instead of the standard error of the "
                  f"mean")
        else:
            sem_scale = None

        def sem(key):
            return sem_scale * p[key] if sem_scale is not None else None

        def leg(base, s):
            if legend_sigma and s is not None:
                return f"{base} (mean sigma={np.mean(s):.2e})"
            return base

        s_ion1 = sem("std_ion1")
        band(p["z_ion1"], p["mean_ion1"], s_ion1, c)
        ax.plot(p["z_ion1"], p["mean_ion1"], lw=1.4, ls=":", color=c,
                 label=leg(f"{label} - {ion1}", s_ion1))

        s_ion2 = sem("std_ion2")
        band(p["z_ion2"], p["mean_ion2"], s_ion2, c)
        ax.plot(p["z_ion2"], p["mean_ion2"], lw=1.4, ls="--", color=c,
                 label=leg(f"{label} - {ion2}", s_ion2))

        s_salt = sem("std_salt")
        band(p["z_salt"], p["mean_salt"], s_salt, c)
        ax.plot(p["z_salt"], p["mean_salt"], lw=2.2, ls="-", color=c,
                 label=leg(f"{label} - {ion1}{ion2} (per-salt mean)", s_salt))

        if not has_std:
            print(f"  no std saved for '{label}' (old run) - showing its "
                  f"mean without a band")

    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1.2, alpha=0.7,
                   label=hline_label, zorder=4)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Distance from box center, z (nm)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both" if log_y else "major")
    if legend_outside:
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1),
                   borderaxespad=0.)
    else:
        ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()

    d = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight" if legend_outside else None)
    plt.close(fig)


def _pair(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PATH, got {s!r}  "
            f"(e.g. '14.4 nm=HP_results/HP_NaCl_TIP3P')")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("boxes", nargs="+", type=_pair,
                   help="LABEL=PATH per box, two or more")
    p.add_argument("--alphas", type=int, default=None,
                   help=f"which <N>alphas subdirectory to read "
                        f"(default: {DEFAULT_ALPHAS}; a run with no alphas "
                        f"subdirectory counts as {DEFAULT_ALPHAS})")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--no-exp", action="store_true")
    p.add_argument("--no-bands", dest="bands", action="store_false",
                   help="hide the +/- 1 SEM uncertainty shading")
    p.add_argument("--ylim", type=float, nargs=2, default=(0.9, 1.3))
    p.set_defaults(bands=True)
    a = p.parse_args()

    if len(a.boxes) < 2:
        raise SystemExit("give at least two boxes to compare")

    plot(a.boxes, n_alphas=a.alphas, out_png=a.out, show_exp=not a.no_exp,
         bands=a.bands, ylim=tuple(a.ylim))
