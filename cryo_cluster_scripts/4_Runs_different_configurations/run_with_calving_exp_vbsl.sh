#!/bin/bash
for script in 4_3_With_calving_exp_onlyMT_vbsl/*; do sbatch ./run_rgi_generic_all.slurm "$script"; done
