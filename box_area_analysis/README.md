# box_area_analysis

Box-area (Lx*Ly) time series over NPT equilibration + production for NaCl/TIP3P, 3.5 m, 4.8x4.8x14.4 nm box.
FBP (k=4184, 20 replicates) vs HP (k=0.661, 6 replicates). Source: `*_state_data.csv` in
`/media/bamo6610/DATA/osmcoeff-salts/{FBP,HP}_NaCl_TIP3P/`. Area = Box Volume / Lz (Lz fixed at 14.4 nm by ZFixed).

    ~/miniconda3/envs/mdanalysis/bin/python box_area_timeseries.py     # -> results/

Outputs in `results/`: `area_timeseries_{FBP,HP}.csv`, `area_summary.csv`, `area_timeseries.png`, `area_per_replicate_{FBP,HP}.png` (one panel per replicate), `area_mean_overlay.png`.
NVT is omitted (box is constant). Time is cumulative: NPT (0-5 ns) then production.
