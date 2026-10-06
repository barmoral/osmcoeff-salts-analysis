#!/bin/bash
# Recreate the symlinks from each workspace to the shared (heavy, never-versioned) data.
# Run from the project root:  bash make_links.sh
cd "$(dirname "$0")"
for ws in original_analysis thermo_analysis; do
  for d in HP_results structures; do ln -sfn ../$d $ws/$d; done
done
ln -sfn ../FBP_results original_analysis/FBP_results
ls -l original_analysis thermo_analysis | grep -- '->'
