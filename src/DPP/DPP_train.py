import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MMR.MF import (
    MatrixFactorization,
    load_and_prepare_matrix,
    filter_empty_users_data,
    get_top_n_recommendations_MF,
tune_mf, train_mf_with_best_params, log_mf_experiment, align_train_val_matrices
)

from DPP import (
    DPP, build_dpp_models, get_recommendations_for_dpp
)

import os
import pandas as pd
import numpy as np




def run_dpp_pipeline(
        ratings_train_path, ratings_val_path, item_path, output_dir=None, dataset=None, datasize=None,
        top_n=10, top_k=20, chunksize = 10000 , n_epochs=50, similarity_types = ["cosine", "jaccard"]
):

    os.makedirs(output_dir, exist_ok=True)

    # Load train/validation matrices
    item_user_rating_train, genre_map, all_genres = load_and_prepare_matrix(ratings_train_path, item_path, nrows_items=chunksize)
    item_user_rating_val, _, _ = load_and_prepare_matrix(ratings_val_path, item_path, nrows_items=chunksize)

    # Align train/val matrices
    train_aligned, val_aligned = align_train_val_matrices(item_user_rating_train, item_user_rating_val)

    R_train = train_aligned.values
    R_val = val_aligned.values

    # Filter out empty movies/users
    R_filtered_train, filtered_item_titles = filter_empty_users_data(R_train, train_aligned.columns)
    filtered_indices = [train_aligned.columns.get_loc(m) for m in filtered_item_titles]
    R_filtered_val = R_val[:, filtered_indices]

    # Tune MF hyperparameters
    best_params = tune_mf(R_filtered_train, R_filtered_val, n_epochs=n_epochs)

    # Train MF with best hyperparameters
    mf, predicted_ratings, train_rmse = train_mf_with_best_params(R_filtered_train, best_params, n_epochs=n_epochs)
    val_rmse = mf.compute_rmse(R_filtered_val, predicted_ratings)

    # Log experiment
    log_mf_experiment(
        output_dir,
        {
            "K": best_params["k"],
            "ALPHA": best_params["alpha"],
            "LAMDA_": best_params["lambda_"],
            "N_EPOCHS": n_epochs,
            "DATASET_NAME": dataset,
            "Data_Size": datasize,
        },
        train_rmse=train_rmse,
        val_rmse=val_rmse
    )

    # Top-N MF recommendations
    get_top_n_recommendations_MF(
        genre_map, predicted_ratings, R_filtered_train,
        item_user_rating_train.index, filtered_item_titles,
        top_n=top_n,
        save_path=os.path.join(output_dir, "mf_train_predictions.csv")
    )

    movie_titles = filtered_item_titles
    filtered_movie_indices = range(predicted_ratings.shape[1])
    item_user_rating_filtered = item_user_rating_train.iloc[:, filtered_movie_indices]


# Build DPP models

    dpp_cosine, dpp_jaccard = build_dpp_models(movie_titles, genre_map, all_genres, predicted_ratings)



    # Run DPP recommendations
    get_recommendations_for_dpp(
        dpp_cosine, item_user_rating_filtered, movie_titles,
        genre_map, predicted_ratings, top_k, top_n,
        output_dir, "cosine"
    )


    get_recommendations_for_dpp(
        dpp_jaccard, item_user_rating_filtered, movie_titles,
        genre_map, predicted_ratings, top_k, top_n,
        output_dir, "jaccard"
    )



    return best_params


# PARAMETER
TOP_N = 10
CHUNK_SIZE = 100000
K = 20
ALPHA = 0.01
LAMDA_ = 0.1
N_EPOCHS = 50
TOP_K = 20
LAMBDA_PARAM = 0.7
DATASET_NAME = "movies"

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_train.csv")
ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_val.csv")
books_file_path = os.path.join(base_dir, "../datasets/MovieLens", "movies.csv")

output_dir = os.path.join(base_dir, f"../datasets/dpp_data/{DATASET_NAME}")

best_params = run_dpp_pipeline(
    ratings_train_path = ratings_train_file,
    ratings_val_path= ratings_val_file,
    item_path = books_file_path,
    output_dir = output_dir,
    top_n = TOP_N,
    top_k = TOP_K,
    chunksize= CHUNK_SIZE,
    n_epochs= N_EPOCHS,
    dataset=DATASET_NAME,
    datasize=CHUNK_SIZE)






