# Run from the FBP_analysis/ folder:   python run_scripts/run_FBP_analysis.py
# Reads trajectories through the FBP_results link; writes only under results/FBP_NaCl_TIP3P/<ktag>_<mi>m/.
"""
run_FBP_analysis.py
===================
Usage script for FBP_analysis_replicates.py, generic version: set K (and the rest) by hand; per-k copies are run_FBP_analysis_k<k>.py.

    conda activate osmotic-analysis          # needs pymbar (osmotic-analysis-hp does not have it)
    cd FBP_analysis && python run_scripts/run_FBP_analysis.py

Set MOLALITIES below: only the result folders of this k and those molalities are read.
K is also the wall force constant used in the calculation, so the folders and the physics cannot disagree.

Trajectories are read from RESULTS_DIR, a template; {ktag} becomes the dispatch's tag for K (4184 -> "k4184") and {mi} the
molality key (3.5 -> "35", 4.0 -> "4"):

    FBP_results/FBP_NaCl_TIP3P/result_files_k4184_<mi>m/md<mi>m_r<i>.pdb
                                                     md<mi>m_r<i>.xtc

Outputs: one folder per k and molality in OUT_DIR, e.g. results/FBP_NaCl_TIP3P/k4184_35m/ (results_35m.json,
final_results_35m.json, ocs.png, ops.png, concprof_*.png, ...). With several molalities, a combined
ocs_k4184_35m_4m.png / ops_... is also written to OUT_DIR, plus the run log.

OLD_RESULTS: the wall position (delta_z, z_center), temperature, van't Hoff factor and ion-pair counts normally come
from the salt_data file the simulation used. Results made BEFORE the current salt_data (the old FBP_membrane runs) do
not match it, so OLD_RESULTS = True overrides delta_z (2.4 nm), T (300 K) and the pair count (counted in the
trajectory). Set it to False for results from osmotic_sim_dispatch_1026[_FBP].py.

UNITS
    K : kJ/mol/nm^2     DELTA_Z : nm     T : K     MOLALITIES : mol/kg
"""

import os
import sys
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # workspace root

import FBP_analysis_replicates as fbp


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
RESULTS_DIR = "FBP_results/FBP_NaCl_TIP3P/result_files_{ktag}_{mi}m"
OUT_DIR     = "results/FBP_NaCl_TIP3P"          # each run writes into OUT_DIR/<ktag>_<mi>m/

ION1, ION2       = "Na", "Cl"        # capitalisation must match the topology / salt_data ("NaCl")
CENTER1, CENTER2 = "Na", "Cl"        # central atom (element) of each ion (NH4 -> "N", NO3 -> "N")

# --- which results ----------------------------------------------------------
K            = 4184                  # wall force constant, kJ/mol/nm^2
MOLALITIES   = [3.5]                 # must exist in SALT_DATA and have a result_files_k4184_<mi>m folder
N_REPLICATES = 20

# --- which generation of results is in those folders ------------------------------------------------------
# True : OLD FBP_membrane results, not synced to SALT_DATA: delta_z = 2.4 nm, T = 300 K, and the number of ion
#        pairs is counted in the trajectory (233 at 3.5 m, not SALT_DATA's 236).
# False: results from osmotic_sim_dispatch_1026[_FBP].py: everything is read from SALT_DATA.
OLD_RESULTS = True

# Overrides (None = take it from SALT_DATA). The wall position matters most: the ions stop at the real wall, so a
# wrong DELTA_Z shifts the pressure by ~15 bar per 0.001 nm. The run warns if the ions' extent disagrees with it
# or if the trajectory's ion count disagrees with the pair count in use.
DELTA_Z = 2.4 if OLD_RESULTS else None            # nm
T       = 300.0 if OLD_RESULTS else None          # K
N_PAIRS = "trajectory" if OLD_RESULTS else None   # ion pairs in the slab: None | "trajectory" | {3.5: 233}

# --- figure label -----------------------------------------------------------
WATER = "TIP3P"

# --- analysis knobs ---------------------------------------------------------
DZ_NM    = 0.05      # z-bin width for the profiles
NSKIP    = 100       # pymbar equilibration-detection stride
PROFILES = True      # False = only the osmotic values (much faster; the profile plots re-read each xtc 6x)

# ===========================================================================


def _main():
    fbp.run(fbp.FBPConfig(
        salt_data_path=SALT_DATA, results_dir=RESULTS_DIR,
        ion1=ION1, ion2=ION2, center_atom1=CENTER1, center_atom2=CENTER2,
        k=K, delta_z=DELTA_Z, T=T, n_pairs=N_PAIRS,
        molalities=MOLALITIES, N_replicates=N_REPLICATES, water=WATER,
        dz_nm=DZ_NM, nskip=NSKIP, profiles=PROFILES, out_dir=OUT_DIR,
    ))
    print("Done.")


if __name__ == "__main__":
    log_dir = Path(OUT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = "_".join([fbp.k_tag(K)] + [f"{fbp.mol_keys(m)[0]}m" for m in MOLALITIES])     # e.g. k4184_35m
    log_path = log_dir / f"run_log_{run_tag}_{timestamp}.txt"

    _stdout, _stderr = sys.stdout, sys.stderr
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(_stdout, log_file)
        sys.stderr = _Tee(_stderr, log_file)
        try:
            _main()
        finally:
            sys.stdout, sys.stderr = _stdout, _stderr
    print(f"Log written to {log_path}")
