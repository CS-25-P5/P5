#!/bin/bash
#SBATCH --job-name=dpp_experiment   # Job name
#SBATCH --output=Slurmscripts/DPP/OUTPUT/dpp_%j.out
#SBATCH --error=Slurmscripts/DPP/OUTPUT/dpp_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=LA15YK@student.aau.dk

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Activate your Python environment
echo "Activating virtual environment..."
source /ceph/home/student.aau.dk/la15yk/P5/venv/bin/activate


# Debug info
echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"

cd ~/P5

echo "Running on $(hostname)"
echo "Start time: $(date)"
echo "Starting pipeline..."
python3 src/DPP/DPP_Pipeline_NN.py

echo "End time: $(date)"
