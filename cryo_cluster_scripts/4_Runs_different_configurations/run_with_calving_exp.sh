#!/bin/bash
for script in 4_2_With_calving_exp_onlyMT/*; do sbatch ./run_rgi_generic_singularity.slurm "$script"; done
