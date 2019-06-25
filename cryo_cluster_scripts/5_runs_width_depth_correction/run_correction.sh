#!/bin/bash
for script in 5_runs_width_depth_correction/*; do sbatch ./run_rgi_generic_singularity_all.slurm "$script"; done
