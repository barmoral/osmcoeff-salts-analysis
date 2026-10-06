# Run from the thermo_analysis/ folder:   python run_scripts/run_HP_analysis.py
# Reads trajectories through the HP_results link; writes only under results/<case>/{new,legacy_check}/.
"""
run_HP_analysis.py
==================
Usage script for HP_analysis_replicates.py.

    conda activate osmotic-analysis
    python run_HP_analysis.py

Only two things are genuinely per-run: WHICH box (salt_data file) and WHERE the
trajectories are. Everything physical - k, temperature, van't Hoff factor, box
dimensions, ion count, and the experimental phi/density reference - is read
from the salt_data file the simulation itself used, so the analysis cannot
disagree with the simulation.

Trajectories are read from RESULTS_DIR, which is the folder the dispatch wrote:

    <case>/result_files/<ktag>_<mi>m/md<mi>m_r<i>.pdb
                                    md<mi>m_r<i>.xtc

Figures are written next to them unless OUT_DIR is set.
"""

import os
import sys
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # workspace root

from HP_analysis_replicates_thermo import run_analysis, run_dz_width_scan


class _Tee:
    """Write to the real terminal and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# ===========================================================================
# USER INPUTS
# ===========================================================================

# --- which system -----------------------------------------------------------
SALT_DATA   = "structures/salt_data_48x48x144.py"
RESULTS_DIR = "HP_results/HP_NaCl_TIP3P/result_files_k0p661_35m"

ION1, ION2   = "Na", "Cl"        # capitalisation must match the topology
MOLALITY     = 3.5               # target molality; keys into SALT_DATA
N_REPLICATES = 6

# --- labels for figures only ------------------------------------------------
WATER   = "TIP3P"
FF_USED = "Sage 2.3.0"

# --- analysis knobs ---------------------------------------------------------
N_EXPANSION_TERMS = 2       # alpha terms in eq. 12 (B is always included)
DZ_NM             = 0.05    # z-bin width for every profile
N_BOOTSTRAPS      = 500     # 25 for a quick end-to-end test
PROFILES_ONLY     = False   # True = stop before the bootstraps (fast look)
RANDOM_SEED       = 1
EPS               = 1e-4    # mask threshold on the concentration profile

# --- overrides; leave as None to derive ------------------------------------
# --- where results go: NEVER the trajectory folder (that would overwrite the old results) --------
# HP_LEGACY=1 python run_scripts/<this script>   -> same analysis with the OLD A and box-average molality
#                                                   (parity check against the original HP_analysis_replicates.py)
LEGACY  = os.environ.get("HP_LEGACY", "0") == "1"
OUT_DIR = f"results/{os.path.basename(os.path.dirname(RESULTS_DIR))}/{'legacy_check' if LEGACY else 'new'}"

# --- new in HP_analysis_replicates_thermo (all optional; defaults shown) --------------------------
CHECKS           = "ignore"   # "warn" | "strict": box-area and reservoir-density checks
RESERVOIR_TARGET = None       # g/cm3 of pure water for YOUR model at T,P (used by warn/strict only)
RESNAME1  = None     # default: ION1.upper()   e.g. "NA"
RESNAME2  = None     # default: ION2.upper()   e.g. "CL"
CENTER1   = None     # default: ION1          element used to pick the centre
CENTER2   = None     # default: ION2
K_OVERRIDE = None    # default: k_HP from SALT_DATA. Set ONLY if the run used -k

# --- histogram bin-width sensitivity scan -----------------------------------
# Set RUN_DZ_SCAN = True to instead re-run the analysis at several histogram
# bin widths (dz_nm x each factor below) and produce a figure comparing the
# unweighted / weighted / maximum-likelihood osmotic-coefficient curves as a
# function of bin width. This re-runs the full bootstrap pipeline once per
# factor, so it is much slower than a single run - drop N_BOOTSTRAPS for a
# quick look (e.g. 100-200).
RUN_DZ_SCAN    = False
DZ_WIDTH_FACTORS = (2.0, 1.0, 0.5)   # 2x, base, 1/2, 1/4, 1/10

# ===========================================================================

def _main():
    if RUN_DZ_SCAN:
        scan = run_dz_width_scan(
            salt_data_path=SALT_DATA,
            results_dir=RESULTS_DIR,
            ion1=ION1,
            ion2=ION2,
            molality=MOLALITY,
            N_replicates=N_REPLICATES,
            base_dz_nm=DZ_NM,
            width_factors=DZ_WIDTH_FACTORS,
            n_bootstraps=N_BOOTSTRAPS,
            out_dir=OUT_DIR,
            legacy=LEGACY, checks=CHECKS, reservoir_density_target=RESERVOIR_TARGET,
            thermo_extras=False,
            water=WATER,
            ff_used=FF_USED,
            n_expansion_terms=N_EXPANSION_TERMS,
            random_seed=RANDOM_SEED,
            eps=EPS,
            resname1=RESNAME1,
            resname2=RESNAME2,
            center_atom1=CENTER1,
            center_atom2=CENTER2,
            k_override=K_OVERRIDE,
        )
    else:
        results = run_analysis(
            salt_data_path=SALT_DATA,
            results_dir=RESULTS_DIR,
            ion1=ION1,
            ion2=ION2,
            molality=MOLALITY,
            N_replicates=N_REPLICATES,
            water=WATER,
            ff_used=FF_USED,
            n_expansion_terms=N_EXPANSION_TERMS,
            dz_nm=DZ_NM,
            n_bootstraps=N_BOOTSTRAPS,
            profiles_only=PROFILES_ONLY,
            random_seed=RANDOM_SEED,
            eps=EPS,
            out_dir=OUT_DIR,
            legacy=LEGACY, checks=CHECKS, reservoir_density_target=RESERVOIR_TARGET,
            resname1=RESNAME1,
            resname2=RESNAME2,
            center_atom1=CENTER1,
            center_atom2=CENTER2,
            k_override=K_OVERRIDE,
        )
    print("Done.")


if __name__ == "__main__":
    log_dir = Path(OUT_DIR) if OUT_DIR else Path(RESULTS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_log_{timestamp}.txt"

    _stdout, _stderr = sys.stdout, sys.stderr
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(_stdout, log_file)
        sys.stderr = _Tee(_stderr, log_file)
        try:
            _main()
        finally:
            sys.stdout, sys.stderr = _stdout, _stderr
    print(f"Log written to {log_path}")
