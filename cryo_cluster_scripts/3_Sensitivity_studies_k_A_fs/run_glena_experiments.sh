#!/bin/bash
for script in 3_2_glen_A_exp/*; do sbatch ./run_rgi_generic_singularity.slurm "$script"; done
