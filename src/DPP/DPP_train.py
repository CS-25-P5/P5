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
        top_n=10, top_k=20, chunksize=10000, n_epochs=50, similarity_types=["cosine", "jaccard"]
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
ratings_val_file = os.path.join(base_dir, "../datasets/mmr_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_val.csv")
movies_file_path = os.path.join(base_dir, "../datasets/MovieLens", "movies.csv")

output_dir = os.path.join(base_dir, f"../datasets/dpp_data/{DATASET_NAME}")

best_params = run_dpp_pipeline(
    ratings_train_path = ratings_train_file,
    ratings_val_path= ratings_val_file,
    item_path = movies_file_path,
    output_dir = output_dir,
    top_n = TOP_N,
    top_k = TOP_K,
    chunksize= CHUNK_SIZE,
    n_epochs= N_EPOCHS,
    dataset=DATASET_NAME,
    datasize=CHUNK_SIZE)































'''
version 2
# Add this to make sure Python can find the 'MMR' module

import os
import pandas as pd
import numpy as np


# define train pipline
def run_dpp_training_pipeline(
    ratings_path, movies_path,
    output_dir="results_dpp_train",
    top_n=10,           # MF candidate pool
    top_k=20,           # DPP reranked size
    chunksize=10000,
    k=20,
    alpha=0.01,
    lambda_=0.1,
    n_epochs=50,
    similarity_type="cosine"):

    # make output directonary
    os.makedirs(output_dir, exist_ok=True)


    #load train set
    movie_user_rating, genre_map, all_genres = load_and_prepare_matrix(
        ratings_path,
        movies_path,
        chunksize
    )

    R = movie_user_rating.values

    #filter empty rows
    R_filtered, filtered_movie_titles = filter_empty_users_data(
        R, movie_user_rating.columns
    )

    #Train MF
    mf = MatrixFactorization(R_filtered, k, alpha, lambda_, n_epochs)
    mf.train()

    # Full predicted rating matrix
    predicted_ratings = mf.full_prediction()


    # Get top-N candidates for MF
    get_top_n_recommendations_MF(
        genre_map, predicted_ratings, R_filtered,
        movie_user_rating.index, filtered_movie_titles,
        top_n=top_n,
        save_path = os.path.join(output_dir,"mf_train_predictions.csv"))

    # get top 10 movies of DPP
    movie_titles = movie_user_rating.columns.tolist()

    dpp_results = []

    #build DPP model
    dpp_cosine, dpp_jaccard = build_dpp_models(movie_titles,
                                               genre_map,
                                               all_genres,
                                               predicted_ratings)

    #run DPP
    get_recommendations_for_dpp(
        dpp_cosine, movie_user_rating, movie_titles,genre_map, predicted_ratings, top_k, top_n,output_dir, "cosine")

    get_recommendations_for_dpp(
        dpp_jaccard, movie_user_rating, movie_titles,genre_map, predicted_ratings, top_k, top_n, output_dir, "jaccard")

    print("Pipeline for DPP train finished successfully!")










# PARAMETER
TOP_N = 10
CHUNK_SIZE = 10000
K = 20
ALPHA = 0.01
LAMDA_ = 0.1
N_EPOCHS = 50
TOP_K = 20
DATASET_NAME = "movies"

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets/dpp_data", f"ratings_{CHUNK_SIZE}_train.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")
output_dir = os.path.join(base_dir, f"../datasets/dpp_data/{DATASET_NAME}")

run_dpp_training_pipeline(
    ratings_path = ratings_file_path,
    movies_path = movies_file_path,
    output_dir = output_dir,
    top_n = TOP_N,
    top_k = TOP_K,
    chunksize= CHUNK_SIZE,
    k = K,
    alpha = ALPHA,
    lambda_= LAMDA_,
    n_epochs= N_EPOCHS,
    similarity_type = "cosine")
'''

'''
#version 1
movie_user_rating = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=chunksizeMovies)
R = movie_user_rating.values

R_filtered, filtered_user_ids, filtered_movie_titles = filter_empty_users_data(
    R, movie_user_rating.index, movie_user_rating.columns
)

# Train Matrix Factorization model
mf = MatrixFactorization(R_filtered, k=20, alpha=0.01, lamda_=0.1, n_epochs=50)
mf.train()

# Full predicted rating matrix
predicted_ratings = mf.full_prediction()

# Ensure output folder exists before saving CSV
# Save top-N recommendations from MF
all_recommendations = get_top_n_recommendations_MF(
    predicted_ratings, R_filtered, filtered_user_ids, filtered_movie_titles, top_n=top_n
)

output_dir = os.path.join(base_dir, "../datasets")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "DPP_train_predictions.csv")
save_mf_predictions(all_recommendations, output_path= output_path)



# Run DPP Re-ranking
movie_embeddings = mf.Q
movie_titles = movie_user_rating.columns.tolist()

dpp_recommendations_list = []

movies_df = pd.read_csv(movies_file_path)

# Build a genre map like in your MMR code
genre_map = {}
for _, row in movies_df.iterrows():
    title = row['title']
    genres = set(row['genres'].split('|')) if pd.notna(row['genres']) else set()
    genre_map[title] = genres

# Create list of all unique genres
all_genres = sorted({g for genres in genre_map.values() for g in genres})

# Run DPP Re-ranking (genre-based)
movie_titles = movie_user_rating.columns.tolist()
dpp_recommendations_list = []

for user_idx, user_id in enumerate(movie_user_rating.index):
    user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

    # call the new DPP function using genres, not embeddings
    dpp_indices = dpp_recommendations(
        user_id=user_idx,
        predicted_ratings=predicted_ratings,
        movie_titles=movie_titles,
        genre_map=genre_map,
        user_history=user_history,
        all_genres=all_genres,
        top_k=top_n,
        similarity_type = "cosine",
    )

    # Store recommendations
    for rank, idx in enumerate(dpp_indices, start=1):
        title = movie_titles[idx]
        genres = ",".join(genre_map.get(title, []))
        dpp_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'movieTitle': title,
            'predictedRating': predicted_ratings[user_idx, idx],
            'genres': genres  #include genres for clarity
        })

# Save results

save_DPP(dpp_recommendations_list, base_dir,similarity_type= "cosine")

'''
