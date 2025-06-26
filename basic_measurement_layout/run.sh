#!/bin/bash
#SBATCH --job-name=aai
#SBATCH --output=../../logs/aai/aai_%A_%a.out
#SBATCH --error=../../logs/aai/aai_%A_%a.err
#SBATCH --array=1-3:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.voudouris@helmholtz-munich.de

#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --nice=10000

# Exit immediately if a command exits with a non-zero status.
set -e

# Define arrays for the parameter sweep
# pixels=(4 8 12 16 20 24 28 32 36 40)
# noise=(0.0 0.1 0.2 0.3 0.4 0.5)
pixels=(8 20 24)
noise=(0.4 0.0 0.5)

# Calculate total combinations to select the correct parameters for this task
# total_pixels=${#pixels[@]}
# total_noise=${#noise[@]}

# Calculate indices for this specific array task
combo_index=$((SLURM_ARRAY_TASK_ID - 1))
# pixel_index=$((combo_index / total_noise))
# noise_index=$((combo_index % total_noise))

# Get the parameters for this run
# current_pixel=${pixels[$pixel_index]}
# current_noise=${noise[$noise_index]}
current_pixel=${pixels[$combo_index]}
current_noise=${noise[$combo_index]}

echo "Running task ${SLURM_ARRAY_TASK_ID} with pixels: $current_pixel, noise: $current_noise"

export TMPDIR="/lustre/groups/hcai/workspace/k.voudouris/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

function cleanup {
  echo "Cleaning up temporary directory: $TMPDIR"
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

# Set other environment variables that might be needed by applications
chmod u+rwx "$TMPDIR"
export XDG_RUNTIME_DIR="$TMPDIR"
export APPTAINER_TMPDIR="$TMPDIR" # For newer Apptainer/Singularity versions
export SINGULARITY_TMPDIR="$TMPDIR" # For older Singularity versions
export SSL_CERT_DIR=/etc/ssl/certs/

# Change to the project's working directory
cd /lustre/groups/hcai/workspace/k.voudouris/measurement-layouts/basic_measurement_layout

# Execute the python script within the singularity container
# xvfb-run creates a virtual framebuffer, which is necessary for GUI-less environments
singularity exec \
  --bind /lustre/groups/hcai/workspace/k.voudouris/measurement-layouts/basic_measurement_layout:/mnt \
  --bind "$TMPDIR" \
  animalai.sif \
  xvfb-run -a \
  python3.10 /mnt/heuristicAgentSimulation.py \
    "$current_pixel" \
    "$current_noise" \
    "/mnt/configs/" \
    "/mnt/data/results_pixels_${current_pixel}_noise_${current_noise}.csv" \
    "/mnt/env/animalAI.x86_64"

echo "Task ${SLURM_ARRAY_TASK_ID} completed successfully."

# The 'trap cleanup EXIT' will handle the removal of the temp directory automatically.
