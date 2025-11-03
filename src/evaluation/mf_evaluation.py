import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


#RMSE / MAE, measures how accuratly the ratings are predicted
#single value but can plot per epoch to show convergence
def evaluate_rmse_mae(model, test, movie_to_idx): #the model, the test data, and a dictionary of movie to idx
    y_true = []
    y_pred = []

    for row in test.itertuples(index=False):
        user = int(row.userId)
        movieId = int(row.movieId)
        if movieId not in movie_to_idx: #skip loop iteration if movie not seen before
            continue
        i = movie_to_idx[movieId]
        pred = model.mu + model.b_u[user] + model.b_i[i] + np.dot(model.P[user, :], model.Q[i, :])
        y_true.append(float(row.rating))
        y_pred.append(pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # calculate rmse
    mae = mean_absolute_error(y_true, y_pred) #calculate mae
    return rmse, mae


#Precision@k and recall@k, measures how good the model is at ranking and recommending relevant items.
#Precissíon@k is out of the top items recommended, how many were relevant
#Recall@k is out of the relevant items in the test set, how many were succesfully included in recommendations?

#can plot vs k to illustrate how performance changes with list length
def precision_recall_at_k(all_recommendations, test, movie_to_idx, k=10): #needs a dictionary where all user IDs are the key and the recommendation list what is keyed to the key, the test data, a dictionary of movie  to idx, and the number of recommendations (10)
    precisions, recalls = [], []

    for user, rec_titles in all_recommendations.items():
        # Relevant movies in test set (the movies the user rated)
        user_test_movies = test[test.userId == user]['movieId'].tolist()
        user_test_idx = {movie_to_idx[movieId] for movieId in user_test_movies if movieId in movie_to_idx} #converts relevant moveIDs to model indices (represents the total number of movies the model could have correctly recommended)

        if not user_test_idx:
            continue  # skip users with no test data

        recommended_idx = [movie_to_idx[movieId] for movieId in movie_to_idx if movieId in movie_to_idx]
        recommended_titles = rec_titles[:k] #takes top k recommendations from the models output
        recommended_movie_ids = {
            movieId for movieId, title in zip(movie_to_idx.keys(), movie_to_idx.keys())
            if title in recommended_titles
        }

        hits = len(user_test_idx.intersection(recommended_movie_ids))
        precision = hits / k
        recall = hits / len(user_test_idx)
        precisions.append(precision)
        recalls.append(recall)

    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    return mean_precision, mean_recall


#ndcgv@K (Normalized Discounted Cumulative Gain), evaluates the quality of ranking (takes into concideration where a recommended item is in the list), calculates an average score across all uses
def ndcg_at_k(all_recommendations, test, movie_to_idx, k=10): #needs a dictionary where all user IDs are the key and the recommendation list what is keyed to the key, the test data, a dictionary of movie  to idx, and the number of recommendations (10)
    def dcg(scores): #helper function for calculating dcg, the scores are a lst of relevance scores of items at position 1,2,3...
        return np.sum((2 ** np.array(scores) - 1) / np.log2(np.arange(2, len(scores) + 2)))

    ndcgs = []
    for user, rec_titles in all_recommendations.items():
        user_test_movies = test[test.userId == user]['movieId'].tolist()
        user_test_idx = {movie_to_idx[mid] for movieId in user_test_movies if movieId in movie_to_idx} #identifies relevant movies rated by user
        if not user_test_idx:
            continue

        # Binary relevance: 1 if in test set
        recs = rec_titles[:k]
        rel = [1 if title in recs else 0 for title in recs]
        ideal_rel = sorted(rel, reverse=True)

        dcg_val = dcg(rel)
        idcg_val = dcg(ideal_rel)
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)

    return np.mean(ndcgs) if ndcgs else 0.0

#MAP@K
