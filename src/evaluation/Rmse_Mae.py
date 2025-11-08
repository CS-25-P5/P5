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
