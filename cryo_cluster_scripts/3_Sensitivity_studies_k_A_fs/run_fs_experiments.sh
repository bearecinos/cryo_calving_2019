#!/bin/bash
for script in 3_3_fs_exp/*; do sbatch ./run_rgi_generic.slurm "$script"; done
