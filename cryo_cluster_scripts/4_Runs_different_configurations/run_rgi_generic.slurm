#!/bin/bash
#
#SBATCH --job-name=config_runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --partition=high
#SBATCH --time=24:00:00
#SBATCH --mail-user=recinos@uni-bremen.de
#SBATCH --mail-type=ALL

# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e

# Load the required environment modules for OGGM
# module purge
module load python/3.6 oggm-binary-deps/3

# Activate our local OGGM virtualenv
source ~/oggm_new/venv/bin/activate

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/"
mkdir -p "$WORKDIR"
echo "Workdir for this run: $WORKDIR"

# Export the WORKDIR as environment variable so our script can use it to find its working directory.
export WORKDIR

# Use the local data download cache
export OGGM_DOWNLOAD_CACHE=/home/data/download
export OGGM_DOWNLOAD_CACHE_RO=1
export OGGM_EXTRACT_DIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/oggm_tmp"

# Run the actual job.
python3 "$@"

echo "Start copying..."

# Once a slurm job is done, slurm will clean up the /work directory on that node from any leftovers from that user.
# So copy any result data you need from there back to your home dir!
# $SLURM_SUBMIT_DIR points to the directory from where the job was initially commited.
OUTDIR="$HOME/cryo_calving_2019/output_data/4_Runs_different_configurations/${1/.py/}/"
mkdir -p "$OUTDIR"

# Copy any neccesary result data.
cp -R "${WORKDIR}/"*.csv "${OUTDIR}/"

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"

