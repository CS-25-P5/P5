#!/bin/bash
#SBATCH --job-name=books_movies  # Job name
#SBATCH --output=books_movies_%j.out
#SBATCH --error=books_movies_%j.err
#SBATCH --time=6:00:00
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
python3 src/MMR/pipeline.py

echo "End time: $(date)
