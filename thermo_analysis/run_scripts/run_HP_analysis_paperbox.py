# Run from the thermo_analysis/ folder:   python run_scripts/run_HP_analysis_paperbox.py
# Reads trajectories through the HP_results link; writes only under results/<case>/{new,legacy_check}/.
"""
run_HP_analysis_paperbox.py
===========================
Usage script for HP_analysis_replicates.py - Hosseini's box, 3.0 x 3.0 x 10.0 nm,
TIP4P/2005 water with the JC(TIP4P-Ew) ion parameters.

    conda activate osmotic-analysis-hp
    python run_HP_analysis_paperbox.py

Only SALT_DATA and RESULTS_DIR are per-run. k, temperature, van't Hoff factor,
box, ion count and the experimental phi/density reference all come from the
salt_data file the simulation used.

This box: k = 1.5567 kJ/mol/nm^2 (tag k1p56), 60 ion pairs, 3008 waters.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # workspace root

from HP_analysis_replicates_thermo import run_analysis

# ===========================================================================
# USER INPUTS
# ===========================================================================

# --- which system -----------------------------------------------------------
SALT_DATA   = "structures/salt_data_30x30x100.py"
RESULTS_DIR = "HP_results/HP_NaCl_TIP4P2005_paperbox/result_files_35m"

ION1, ION2   = "Na", "Cl"        # capitalisation must match the topology
MOLALITY     = 3.5               # target molality; keys into SALT_DATA
N_REPLICATES = 6

# --- labels for figures only ------------------------------------------------
WATER   = "TIP4P/2005"
FF_USED = "JC(TIP4P-Ew)"

# --- analysis knobs ---------------------------------------------------------
N_EXPANSION_TERMS = 2       # alpha terms in eq. 12 (B is always included)
DZ_NM             = 0.05    # z-bin width; 10 nm box -> 100 folded bins
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

# ===========================================================================

if __name__ == "__main__":
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
