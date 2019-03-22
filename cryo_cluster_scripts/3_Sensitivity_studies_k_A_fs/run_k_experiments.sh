#!/bin/bash
for script in 3_1_k_exp/*; do sbatch ./run_rgi_generic.slurm "$script"; done
