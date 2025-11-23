#!/bin/bash
#SBATCH --job-name=mmr_genre_training
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# Run your training script
singularity exec --nv /ceph/container/pytorch/pytorch_25.10.sif \
    python3 src/MMR/MMR_train.py

echo "=== Job finished at $(date) ==="
