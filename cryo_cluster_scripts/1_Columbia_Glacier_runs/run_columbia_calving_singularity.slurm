#!/bin/bash
#
#SBATCH --job-name=prepro_columbia_singu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mail-user=recinos@uni-bremen.de
#SBATCH --mail-type=ALL

# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/"
mkdir -p "$WORKDIR"
echo "Workdir for this run: $WORKDIR"

# Export the WORKDIR as environment variable so our script can use it to find its working directory.
export WORKDIR

# Run the actual job. The srun invocation starts it as individual step for slurm.
srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec docker://oggm/oggm:20190612 bash -s <<EOF
  set -e

  python3 -m venv --system-site-packages "$WORKDIR/oggm_env"
  source "$WORKDIR/oggm_env/bin/activate"

  pip install --upgrade pip setuptools wheel
  pip install --upgrade "git+https://github.com/OGGM/oggm.git@da8910f8edf0607266760eefb4befd799504a9cd"

  export OGGM_DOWNLOAD_CACHE=/home/data/download
  export OGGM_DOWNLOAD_CACHE_RO=1

  python3 ./run_calving_columbia.py
EOF

echo "Start copying..."

# Once a slurm job is done, slurm will clean up the /work directory on that node from any leftovers from that user.
# So copy any result data you need from there back to your home dir!
# $SLURM_SUBMIT_DIR points to the directory from where the job was initially commited.
OUTDIR="$HOME/cryo_calving_2019/output_data/1_Columbia_Glacier_runs/2_calving/"
mkdir -p "$OUTDIR"

# Copy any neccesary result data.
cp -R "${WORKDIR}/per_glacier"* "${OUTDIR}/"
cp -R "${WORKDIR}/"*.csv "${OUTDIR}/"

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"

