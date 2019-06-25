#!/bin/bash
for script in 4_1_No_calving_exp/*; do sbatch ./run_rgi_generic_singularity.slurm "$script"; done
