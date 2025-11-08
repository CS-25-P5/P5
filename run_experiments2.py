#!/bin/bash
#SBATCH --job-name=P5_MF_MMR          # Job name
#SBATCH --output=mf_mmr_output.log    # Log file
#SBATCH --time=04:00:00               # Max runtime (HH:MM:SS) — adjust as needed
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=8             # Number of CPU cores
#SBATCH --mem=24G                     # Memory (adjust if needed)

python MF_test.py