"""
plot_alpha_comparison.py
========================
Two ways to lay out the same curves:

  --mode alpha (default): one figure, three panels - unweighted / weighted /
      maximum likelihood - each holding one curve per number of fitting terms.
      This is the transpose of what run_analysis writes.
  --mode fit: one panel per alpha count, each holding one curve per fit type
      (unweighted / weighted / maximum likelihood). This is what run_analysis
      writes per-alpha-count, laid out side by side for comparison.
  --mode both: write both figures.

    conda activate osmotic-analysis-hp

    # everything under one results directory
    python plot_alpha_comparison.py HP_results/HP_NaCl_TIP3P/result_files_k0p661_35m

    # the other layout - one panel per alpha count, fit types overlaid
    python plot_alpha_comparison.py <results_dir> --mode fit

    # write somewhere else, and drop the experimental reference curve
    python plot_alpha_comparison.py <results_dir> -o figs/nacl_alphas.png --no-exp

    # only some of the alpha counts
    python plot_alpha_comparison.py <results_dir> --alphas 2 3 4

    # hide the +/- 2 sigma bootstrap uncertainty bands (shown by default)
    python plot_alpha_comparison.py <results_dir> --no-bands

    # bands are thin (uncertainties are ~0.001-0.006 vs a 0.4-unit y-axis) -
    # widen the multiplier if they're still hard to see
    python plot_alpha_comparison.py <results_dir> --sigma 4

Expects the layout you already have - one subdirectory per alpha count:

    result_files_k0p661_35m/
        1alphas/curves_NaCl_35m_1alphas.npz
        2alphas/curves_NaCl_35m_2alphas.npz
        3alphas/...

Salt, molality, k and the experimental curve are read out of the .npz, so
nothing here is NaCl- or 3.5 m-specific.

NOTE: the .npz files are written by the np.savez block at the end of
run_analysis(). Runs made before that block existed have only the PDFs, so
those directories need one re-run to produce data. The mean curves come from
the fit to the un-resampled data, so N_BOOTSTRAPS=25 regenerates identical
curves in a fraction of the time - only the shaded bands need the full 500.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

FITS = ["unweighted", "weighted", "maximum_likelihood"]
FIT_COLORS = {"unweighted": "gray", "weighted": "darkorange",
              "maximum_likelihood":"indigo"}


def find_runs(results_dir, want=None):
    """Locate one .npz per <N>alphas subdirectory. Returns [(N, path), ...]."""
    runs = []
    for d in sorted(glob.glob(os.path.join(results_dir, "*alphas"))):
        m = re.search(r"(\d+)alphas/?$", d)
        if not m:
            continue
        n = int(m.group(1))
        if want and n not in want:
            continue
        hits = glob.glob(os.path.join(d, "curves_*.npz"))
        if not hits:
            print(f"  skip {os.path.basename(d)}: no curves_*.npz "
                  f"(re-run the analysis for this one)")
            continue
        if len(hits) > 1:
            print(f"  warn {os.path.basename(d)}: {len(hits)} npz files, "
                  f"using {os.path.basename(hits[0])}")
        runs.append((n, hits[0]))
    return sorted(runs)


def _draw_curve(ax, d, fit, color, label, bands, n_sigma):
    x, y = d[f"c_{fit}"], d[f"oc_{fit}"]
    ax.plot(x, y, lw=1.8, color=color, label=label)
    if bands:
        s = n_sigma * d[f"std_{fit}"]
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.3, lw=0)
        ax.plot(x, y - s, color=color, alpha=0.6, lw=0.6)
        ax.plot(x, y + s, color=color, alpha=0.6, lw=0.6)


def _draw_exp(ax, ref, show_exp):
    if show_exp and "exp_molality" in ref:
        ax.plot(ref["exp_molality"], ref["exp_oc"], "o-", color="k",
                ms=4, lw=1.5, zorder=5, label="experiment")


def _load(results_dir, want):
    runs = find_runs(results_dir, want)
    if not runs:
        raise SystemExit(
            f"no usable <N>alphas/curves_*.npz under {results_dir}\n"
            f"  found: {sorted(os.path.basename(p) for p in glob.glob(os.path.join(results_dir, '*alphas')))}")

    data = {n: np.load(p, allow_pickle=True) for n, p in runs}
    ref = data[runs[0][0]]
    salt = str(ref["salt"])
    molality = float(ref["molality"])
    k = float(ref["k"])

    print(f"  {salt} at {molality} m, k = {k:.4f}")
    print(f"  alpha counts: {[n for n, _ in runs]}")
    return runs, data, ref, salt, molality, k


def plot(results_dir, out_png=None, want=None, show_exp=True, bands=True,
         n_sigma=2.0, ylim=(0.9, 1.3)):
    """One figure, three panels (fit types); each panel overlays alpha counts."""
    runs, data, ref, salt, molality, k = _load(results_dir, want)

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(runs)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)

    for ax, fit in zip(axes, FITS):
        for (n, _), c in zip(runs, colors):
            d = data[n]
            # n stored terms -> n+1 fitted terms (B is always included)
            _draw_curve(ax, d, fit, c, f"{n + 1} terms (B + {n} α)", bands, n_sigma)
        _draw_exp(ax, ref, show_exp)
        ax.set_title(fit.replace("_", " "))
        ax.set_xlabel("Concentration (mol/kg)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Osmotic coefficient")
    axes[0].set_ylim(*ylim)
    axes[0].set_xlim(0, molality + 0.2)
    axes[-1].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"{salt} at {molality} mol/kg, k = {k:.4f} - "
                 f"effect of the number of fitting terms", fontsize=13)
    fig.tight_layout()

    out_png = out_png or os.path.join(results_dir,
                                      f"alpha_comparison_{salt}_{molality}m.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


def plot_by_fit(results_dir, out_png=None, want=None, show_exp=True, bands=True,
                 n_sigma=2.0, ylim=(0.9, 1.3)):
    """One panel per alpha count; each panel overlays the three fit types."""
    runs, data, ref, salt, molality, k = _load(results_dir, want)

    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.2),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (n, _) in zip(axes, runs):
        d = data[n]
        for fit in FITS:
            _draw_curve(ax, d, fit, FIT_COLORS[fit], fit.replace("_", " "),
                        bands, n_sigma)
        _draw_exp(ax, ref, show_exp)
        ax.set_title(f"{n + 1} terms (B + {n} α)")
        ax.set_xlabel("Concentration (mol/kg)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Osmotic coefficient")
    axes[0].set_ylim(*ylim)
    axes[0].set_xlim(0, molality + 0.2)
    axes[-1].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"{salt} at {molality} mol/kg, k = {k:.4f} - "
                 f"effect of the fit type", fontsize=13)
    fig.tight_layout()

    out_png = out_png or os.path.join(results_dir,
                                      f"fit_comparison_{salt}_{molality}m.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", help="directory holding the <N>alphas/ subdirs")
    p.add_argument("-o", "--out", default=None,
                   help="output png (only valid for a single --mode)")
    p.add_argument("--mode", choices=["alpha", "fit", "both"], default="alpha",
                   help="alpha: panels=fit type, curves=alpha count (default). "
                        "fit: panels=alpha count, curves=fit type. both: write both")
    p.add_argument("--alphas", type=int, nargs="+", default=None,
                   help="only these alpha counts (default: all found)")
    p.add_argument("--no-exp", action="store_true",
                   help="omit the experimental reference curve")
    p.add_argument("--no-bands", dest="bands", action="store_false",
                   help="omit the bootstrap uncertainty bands")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="width of the uncertainty band, in std devs (default: 2)")
    p.add_argument("--ylim", type=float, nargs=2, default=(0.9, 1.3))
    a = p.parse_args()
    if a.out and a.mode == "both":
        raise SystemExit("--out isn't valid with --mode both (two figures, one name)")

    kwargs = dict(want=set(a.alphas) if a.alphas else None, show_exp=not a.no_exp,
                  bands=a.bands, n_sigma=a.sigma, ylim=tuple(a.ylim))
    if a.mode in ("alpha", "both"):
        plot(a.results_dir, out_png=a.out, **kwargs)
    if a.mode in ("fit", "both"):
        plot_by_fit(a.results_dir, out_png=a.out, **kwargs)
