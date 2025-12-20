#!/bin/bash
#SBATCH --job-name=candidate_nn  # Job name
#SBATCH --output=candidate_nn_%j.out
#SBATCH --error=candidate_nn_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=VO55ZP@student.aau.dk

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Activate your Python environment
echo "Activating virtual environment..."
source ~/projects/P5/venv/bin/activate

# Debug info
echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"

cd ~/projects/P5

echo "Running on $(hostname)"
echo "Start time: $(date)"
echo "Starting pipeline..."
python3 src/MMR/pipeline_nn.py

echo "End time: $(date)
