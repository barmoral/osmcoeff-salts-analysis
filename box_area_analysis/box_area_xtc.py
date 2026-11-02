"""
Box-area (Lx*Ly) time series from result_files_*/md35m_r*.xtc unit cells (production frames only).

    python box_area_xtc.py [resdir] [outdir]

Writes area_timeseries.csv (replicate, time_ns, area_nm2), area_summary.csv, area_timeseries.png
(overlay + per-replicate panels).
"""
import glob, os, sys
import numpy as np, pandas as pd
import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/media/bamo6610/DATA/osmcoeff-salts-analysis/HP_results/HP_NaCl_TIP3P_144/result_files_k0p661_35m"
res = sys.argv[1] if len(sys.argv) > 1 else RES
here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[2] if len(sys.argv) > 2 else f"{here}/results/HP_144_result_files"
os.makedirs(out, exist_ok=True)

rows = []
for x in sorted(glob.glob(f"{res}/md35m_r*.xtc"), key=lambda p: int(p.rsplit("_r", 1)[1][:-4])):
    r = int(x.rsplit("_r", 1)[1][:-4])
    u = mda.Universe(x.replace(".xtc", ".pdb"), x)
    t = np.array([float(ts.time) for ts in u.trajectory]) / 1000.0  # ps -> ns
    d = np.array([ts.dimensions[:2].copy() for ts in u.trajectory])
    rows.append(pd.DataFrame({"replicate": r, "time_ns": t, "area_nm2": d[:, 0] * d[:, 1] / 100.0}))
df = pd.concat(rows, ignore_index=True)
df.to_csv(f"{out}/area_timeseries.csv", index=False)

summ = []
for r, g in df.groupby("replicate"):
    a = g.area_nm2; n = max(1, len(a) // 10)
    summ.append(dict(replicate=r, first=a.iloc[0], last=a.iloc[-1], mean=a.mean(), std=a.std(), min=a.min(),
                     max=a.max(), drift=a.iloc[-n:].mean() - a.iloc[:n].mean()))
summ = pd.DataFrame(summ); summ.to_csv(f"{out}/area_summary.csv", index=False)
print(summ.round(4).to_string(index=False))
print(f"all: mean={df.area_nm2.mean():.4f}  std={df.area_nm2.std():.4f}  mean of replicate means={summ['mean'].mean():.4f} "
      f"(sd {summ['mean'].std():.4f})  drift mean={summ.drift.mean():.4f}")

col = "#c8641e"
fig, axs = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
for r, g in df.groupby("replicate"):
    axs[0].plot(g.time_ns, g.area_nm2, lw=0.8, alpha=0.6, label=f"r{r}")
m = df.groupby("time_ns").area_nm2; mm, ss = m.mean(), m.std()
axs[1].fill_between(mm.index, mm - ss, mm + ss, color=col, alpha=0.2, lw=0)
axs[1].plot(mm.index, mm, color=col, lw=1.6, label="mean ± std over replicates")
for ax, tt in zip(axs, ("per replicate", "replicate mean")):
    ax.set_xlabel("time (ns, production)"); ax.set_ylabel("box area Lx·Ly (nm²)"); ax.set_title(f"HP_144 production: {tt}", loc="left", fontsize=10)
    ax.grid(color="0.88", lw=0.6); ax.legend(frameon=False, fontsize=8, ncol=3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(f"{out}/area_timeseries.png", dpi=200)
