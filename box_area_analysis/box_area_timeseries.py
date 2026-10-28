"""
Box-area (Lx*Ly) time series over NPT equilibration + production, HP vs FBP (4.8x4.8x14.4 nm, NaCl/TIP3P, 3.5 m).

    python box_area_timeseries.py [--group box_4.8x4.8x14.4 ...] [--outdir results]

Area = Box Volume / Lz (per-case Lz in CASES) from each stage's *_state_data.csv (Lz is fixed by ZFixed).
Time is cumulative across stages (NPT then production), taken from the Step column x the timestep.
Writes, under --outdir/<group>/ (groups in GROUPS):
    area_timeseries_<case>.csv   long table: replicate, stage, time_ns, area_nm2
    area_summary.csv             per replicate: NPT-end area, prod mean/std/range, prod drift (last-first 10% mean)
    area_timeseries.png          per-replicate traces, one panel per case, shared axes
    area_per_replicate_<case>.png  one small panel per replicate (all replicates, shared axes)
    area_mean_overlay.png        replicate-mean +/- std, both cases on one axis
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/media/bamo6610/DATA/osmcoeff-salts"
CASES = {  # name -> (results dir, k tag, line colour, Lz in nm)
    "FBP":      (f"{SRC}/FBP_NaCl_TIP3P",      "k4184",  "#2a6fb0", 14.4),
    "HP":       (f"{SRC}/freezed_results/HP_NaCl_TIP3P",       "k0p661", "#c8641e", 14.4),
    "HP_lcbox": (f"{SRC}/freezed_results/HP_NaCl_TIP3P_lcbox", "k0p661", "#c8641e", 18.0),
    "HP_dbox":  (f"{SRC}/freezed_results/HP_NaCl_TIP3P_dbox",  "k0p661", "#c8641e", 28.8),
    "HP_144_smoke": (f"{SRC}/HP_NaCl_TIP3P_144_smoke", "k0p661", "#2a8f5a", 14.4),  # 1026 code: periodic restraint + box carry-over
    "HP_180_smoke": (f"{SRC}/HP_NaCl_TIP3P_180_smoke", "k0p423", "#2a8f5a", 18.0),
    "HP_TIP4P2005_paperbox": (f"{SRC}/freezed_results/HP_NaCl_TIP4P2005_paperbox", "k1p56", "#c8641e", 10.0),
}
GROUPS = {  # box -> cases compared on shared axes; outputs go to results/<group>/
    "box_4.8x4.8x14.4": ["FBP", "HP"],
    "box_4.8x4.8x18.0": ["HP_lcbox"],
    "box_4.8x4.8x28.8": ["HP_dbox"],
    "box_4.8x4.8x14.4_smoke_1026": ["HP_144_smoke"],  # 2 ns NPT + 3 ns production
    "box_4.8x4.8x18.0_smoke_1026": ["HP_180_smoke"],  # 2 ns NPT + 3 ns production
    "box_2.9x2.9x10.0_TIP4P2005": ["HP_TIP4P2005_paperbox"],  # 25 ns NPT + 200 ns production
}
STEP_FS = 2.0  # time_step_fs default of the dispatch script


def load_stage(path, lz):
    df = pd.read_csv(path)
    df.columns = [c.strip('#"') for c in df.columns]
    return df["Step"].to_numpy() * STEP_FS * 1e-6, df["Box Volume (nm^3)"].to_numpy() / lz  # ns, nm^2


def load_case(name, lz):
    root, ktag, _, _ = CASES[name]
    rows = []
    for d in sorted(glob.glob(f"{root}/prod_sim_35m_{ktag}_r*")):
        r = int(d.rsplit("_r", 1)[1])
        t_off = 0.0
        for stage in ("equil_sim_NPT", "prod_sim"):
            f = glob.glob(f"{root}/{stage}_35m_{ktag}_r{r}/*_state_data.csv")
            if not f:
                continue
            t, a = load_stage(f[0], lz)
            rows.append(pd.DataFrame({"replicate": r, "stage": "NPT" if stage.startswith("equil") else "prod",
                                      "time_ns": t + t_off, "area_nm2": a}))
            t_off += t[-1]
    return pd.concat(rows, ignore_index=True)


def summarize(name, df):
    out = []
    for r, g in df.groupby("replicate"):
        npt, prod = g[g.stage == "NPT"].area_nm2, g[g.stage == "prod"].area_nm2
        n = max(1, len(prod) // 10)
        out.append(dict(case=name, replicate=r, npt_last_nm2=npt.iloc[-1], prod_first_nm2=prod.iloc[0],
                        prod_mean_nm2=prod.mean(), prod_std_nm2=prod.std(), prod_min_nm2=prod.min(),
                        prod_max_nm2=prod.max(), prod_drift_nm2=prod.iloc[-n:].mean() - prod.iloc[:n].mean()))
    return pd.DataFrame(out)


def style(ax):
    ax.grid(color="0.88", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def run_group(group, names, out):
    os.makedirs(out, exist_ok=True)
    data = {n: load_case(n, CASES[n][3]) for n in names}
    for n, df in data.items():
        df.to_csv(f"{out}/area_timeseries_{n}.csv", index=False)
    summ = pd.concat([summarize(n, df) for n, df in data.items()], ignore_index=True)
    summ.to_csv(f"{out}/area_summary.csv", index=False)
    print(f"== {group}")
    print(summ.groupby("case")[["npt_last_nm2", "prod_first_nm2", "prod_mean_nm2", "prod_std_nm2", "prod_drift_nm2"]]
          .agg(["mean", "min", "max"]).round(3).T.to_string())

    t_npt = next(iter(data.values())).query("stage == 'NPT'").time_ns.max()
    ymin = min(d.area_nm2.min() for d in data.values()) - 0.2; ymax = max(d.area_nm2.max() for d in data.values()) + 0.2

    fig, axs = plt.subplots(1, len(data), figsize=(5.5 * len(data), 4), sharey=True, constrained_layout=True, squeeze=False)
    for ax, (n, df) in zip(axs[0], data.items()):
        col = CASES[n][2]
        for r, g in df.groupby("replicate"):
            ax.plot(g.time_ns, g.area_nm2, color=col, lw=0.8, alpha=0.5)
        ax.axvline(t_npt, color="0.4", lw=0.8, ls="--"); ax.text(t_npt, ymax, " NPT | prod", va="top", fontsize=8, color="0.3")
        ax.set_title(f"{n}  (n={df.replicate.nunique()} replicates)", fontsize=10, loc="left")
        ax.set_xlabel("time (ns, NPT + production)"); ax.set_ylim(ymin, ymax); style(ax)
    axs[0][0].set_ylabel("box area Lx·Ly (nm²)")
    fig.savefig(f"{out}/area_timeseries.png", dpi=200); plt.close(fig)

    for n, df in data.items():  # every replicate in its own panel
        col = CASES[n][2]; reps = sorted(df.replicate.unique())
        nc = 5 if len(reps) > 6 else 3; nr = -(-len(reps) // nc)
        fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 1.9 * nr), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
        for ax, r in zip(axs.flat, reps):
            g = df[df.replicate == r]
            ax.plot(g.time_ns, g.area_nm2, color=col, lw=0.6)
            ax.axvline(t_npt, color="0.4", lw=0.6, ls="--")
            ax.set_title(f"r{r}", fontsize=8, loc="left"); ax.set_ylim(ymin, ymax); ax.tick_params(labelsize=7); style(ax)
        for ax in axs.flat[len(reps):]:
            ax.axis("off")
        fig.supxlabel("time (ns, NPT + production)", fontsize=9); fig.supylabel("box area Lx·Ly (nm²)", fontsize=9)
        fig.suptitle(f"{n}: area per replicate", fontsize=10, x=0.01, ha="left")
        fig.savefig(f"{out}/area_per_replicate_{n}.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for n, df in data.items():
        col = CASES[n][2]; df = df.assign(tb=df.time_ns.round(3))
        g = df.groupby("tb").area_nm2; m, s = g.mean(), g.std()
        ax.fill_between(m.index, m - s, m + s, color=col, alpha=0.18, lw=0)
        ax.plot(m.index, m, color=col, lw=1.6, label=f"{n} (mean ± std over replicates)")
    ax.axvline(t_npt, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("time (ns, NPT + production)"); ax.set_ylabel("box area Lx·Ly (nm²)"); ax.legend(frameon=False, fontsize=9); style(ax)
    fig.savefig(f"{out}/area_mean_overlay.png", dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", nargs="+", default=list(GROUPS), choices=list(GROUPS)); p.add_argument("--outdir", default="results")
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    for g in a.group:
        run_group(g, GROUPS[g], os.path.join(here, a.outdir, g))


if __name__ == "__main__":
    main()
