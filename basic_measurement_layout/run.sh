#!/bin/bash
#SBATCH --job-name=aai
#SBATCH --output=../../logs/aai_%A_%a.out
#SBATCH --error=../../logs/aai_%A_%a.err
#SBATCH --array=1-60:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.voudouris@helmholtz-munich.de

#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --nice=10000

# Define arrays
pixels=(4 8 12 16 20 24 28 32 36 40)
noise=(0.0 0.1 0.2 0.3 0.4 0.5)

# Calculate total combinations
total_pixels=${#pixels[@]}
total_noise=${#noise[@]}

# Calculate indices for this job
combo_index=$((SLURM_ARRAY_TASK_ID - 1))
pixel_index=$((combo_index / total_noise))
noise_index=$((combo_index % total_noise))

current_pixel=${pixels[$pixel_index]}
current_noise=${noise[$noise_index]}

echo "Running task with pixels: $current_pixel, noise: $current_noise"

cd /lustre/groups/hcai/workspace/k.voudouris/measurement-layouts/basic_measurement_layout

singularity exec --bind /lustre/groups/hcai/workspace/k.voudouris/measurement-layouts/basic_measurement_layout:/mnt --env SSL_CERT_DIR=/etc/ssl/certs/ animalai.sif xvfb-run -a python3.10 /mnt/heuristicAgentSimulation.py $current_pixel $current_noise "/mnt/configs/" "/mnt/data/results_pixels_${current_pixel}_noise_${current_noise}.csv" "/mnt/env/animalAI.x86_64" 555
echo "Task with model: $current_model completed"
