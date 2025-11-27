#!/bin/bash

#SBATCH --job-name=movies_neural_network_with_BPR		#Name of the job
#SBATCH --gres=gpu:1						#Requesting a GPU otherwise NN runs on CPU
#SBATCH --time=01:00:00						#Max amount for training is 1 hour
#SBATCH --mem=32G						#Amount of RAM allocated
#SBATCH --cpus-per-task=8					#CPu core for loading data and process
#SBATCH --output=logs/%x-%j.out 				#%x is the jobe name, and %j is the ID of the job

cd ~/ceph/home/student.aau.dk/nr38sp/P5

source venv/bin/activate

python -u src/backend/MLPwithBPR.py
