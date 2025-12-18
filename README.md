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

# Data
1) The MovieLens dataset is used for algoritm #1,#2, #3, #4, #5  which can be found here: https://grouplens.org/datasets/movielens/latest/

2) Further datasets are retrieved from Kaggle "The Movies Dataset": https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset used for algorithm #2

Multiple csv files contain the following attributes and more: "userId", "movieId", "crew", "keywords", "release dates", "production companies" etc.. Important: the authors have decided to create a new dataset called "input_dataset_for_NN" which combines a subset of the attributes listed across multiple datasets into one unified csv file, which serves as input for algorithm #2. 

3) The goodbooks_10_k_rating_and_description dataset is used for algorithm #1 and #2, and can be downloaded here: https://www.kaggle.com/datasets/mustafayazici/goodbooks-10k-rating-and-description?select=ratings.csv 


The authors of this project would like to ###emphasize### that the collection of datasets is not their own work, and give credit to Grouplens.org, as well as the authors/data scientists on Kaggle.com for using their material as input for the algorithms. 


# Algorithms

1) Maximal Marginal Relevance (MMR) with rating and genres as attributes
2) Detrimental Process Point (DPP) with rating and genres as attributes

3) Neural network with movie rating as attribute (loss function: MSE, optimizer: Adams)

4) Neural network with attributes "rating", "budget", "genres", "original_language", "popularity", "production_companies", "runtime", "title", "vote_average", "vote_count"  (loss function: MSE, optimizer: Adams)

5) Neural network with Bayesian Personalized Ranking (loss function: BPR, optimizer: Adams)


# Validation and testing
The following methods will test the validity of the results.

- Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), Precision@k, Recall@k, NDCG@k, MAP and AUC for matrix factorization algorithms to measure relevance

-Intra list diversity, coverage and inverse gini to measure diversity

# Important commands
