#!/bin/bash

#SBATCH --job-name=recommend_movies_with_NN_genres		            #Name of the job
#SBATCH  --partition=l4						                        #GPU partition
#SBATCH --gres=gpu:4							                    #Ask for one GPU

#SBATCH --time=12:00:00						                        #Max amount for training is 1 hour
#SBATCH --mem=32G						                            #Amount of RAM allocated
#SBATCH --cpus-per-task=8					                        #CPu core for loading data and process
#SBATCH --output=logs/%x-%j.out 				                    #%x is the jobe name, and %j is the ID of the job
#SBATCH --error=logs/%x-%j.err

set -e #stop srcipt if any command fails

cd /ceph/home/student.aau.dk/nr38sp/P5
mkdir -p logs
source venv/bin/activate

python -u src/backend/scripts/recommend_Genres_100Kmovies.py
