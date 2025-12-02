# P5 - Data analysis
AAU Computer Science : 5th semester project. 

This project aims to analyse and compare different ML algorithms, that serves the basis for a movie recommendation system. The authors wish to compare the relevance and diversity of the recommended items by said algoritms, and evaluate the results.

# Quick start
The packages/modules needed in the enviroment are:
- pytorch
- numpy 
- pandas
- math
- sklearn

Install everything with the following command:
pip install -r requirements.txt

# Datasets

1) Kaggle: The movies dataset 100K from : https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

2) Grouplens: The movies dataset 1M from : https://grouplens.org/datasets/movielens/1m/

3) Kaggle: The goodbooks dataset 5M from : https://www.kaggle.com/datasets/mustafayazici/goodbooks-10k-rating-and-description?select=ratings.csv

Multiple csv files contain the following attributes and more: "userId", "movieId", "crew", "keywords", "release dates", "production companies" etc.. Important: the authors have decided to create a new dataset called "input_dataset_for_NN" which combines a subset of the attributes listed across multiple datasets into one unified csv file. This will serve as input for the MLP alorithm which takes the "genres" attribute as extra input.

The authors of this project would like to ###emphasize### that the collection of datasets is not their own work, and give credit to Grouplens.org, as well as the authors and data scientists on Kaggle.com for using their datasets as input for the algorithms. 


# Algorithms

1) Maximal Marginal Relevance (MMR) with rating and genres as attributes
2) Detrimental Process Point (DPP) with rating and genres as attributes
3) Neural network with movie rating as attribute (loss function: MSE, optimizer: Adams)
4) Neural network with Bayesian Personalized Ranking (loss function: BPR, optimizer: Adams)
5) Neural network with extra attribute "genres"  (loss function: MSE, optimizer: Adams)


# Validation and testing
The following methods will test the validity of the results.


# Important commands
