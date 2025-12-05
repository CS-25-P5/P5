#!/bin/bash

#SBATCH --job-name=movies_neural_network_with_BPR		#Name of the job
#SBATCH  --partition=l4						#GPU partition
#SBATCH --gres=gpu:1						#Ask for one GPU

#SBATCH --time=03:00:00						#Max amount for training is 1 hour
#SBATCH --mem=32G						#Amount of RAM allocated
#SBATCH --cpus-per-task=8					#CPu core for loading data and process
#SBATCH --output=logs/%x-%j.out 				#%x is the jobe name, and %j is the ID of the job
#SBATCH --error=logs/%x-%j.err

set -e #stop srcipt if any command fails

cd /ceph/home/student.aau.dk/nr38sp/P5
mkdir -p logs
source venv/bin/activate

python -u src/backend/scripts/MLPwitBPR_1M_movies.py
python -u src/backend/scripts/MLPwitBPR_100K_movies.py
python -u src/backend/scripts/MLPwitBPR_100K_books.py
