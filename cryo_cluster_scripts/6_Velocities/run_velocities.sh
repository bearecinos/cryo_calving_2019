#!/bin/bash
for script in 6_Velocities/*; do sbatch ./run_rgi_generic_all.slurm "$script"; done
