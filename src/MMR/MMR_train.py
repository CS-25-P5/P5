from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF
from MMR import MMR,build_mmr_models, get_recommendations_for_mmr
import os
import pandas as pd


# PARAMETER
TOP_N = 10
CHUNK_SIZE_MOVIES = 10000
K = 20
ALPHA = 0.01
LAMDA_ = 0.1
N_EPOCHS = 50
TOP_K = 20
LAMBDA_PARAM = 0.7
DATASET_NAME = "movies"

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets", "ratings_train.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")

ouput_dir = os.path.join(base_dir,"../datasets/mmr_data", DATASET_NAME)


def run_mmr_pipeline(
    ratings_path, movies_path, output_dir="results", 
    top_n=10, top_k=20, chunksize_movies=10000,
    k=20, alpha=0.01, lambda_=0.1, n_epochs=50, lambda_param=0.7):

    os.makedirs(output_dir, exist_ok=True)

    #load data
    movie_user_rating, genre_map, all_genres = load_and_prepare_matrix(
        ratings_path, movies_path, chunksize_movies)
    
    R = movie_user_rating.values
    R_filtered, filtered_movie_titles = filter_empty_users_data(R, movie_user_rating.columns )

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

    # get top 10 movies of MMR
    movie_titles = movie_user_rating.columns.tolist()

    # Build MMR models
    mmr_cosine, mmr_jaccard = build_mmr_models(
    movie_titles,
    genre_map,
    all_genres,
    predicted_ratings,
    lambda_param
    )

    # Run MMR
    get_recommendations_for_mmr(
        mmr_cosine, movie_user_rating, movie_titles, 
        genre_map, predicted_ratings, top_k, top_n, 
        output_dir, "cosine_test"
    )


    get_recommendations_for_mmr(
        mmr_jaccard, movie_user_rating, movie_titles, 
        genre_map, predicted_ratings, top_k, top_n,
        output_dir, "jaccard_test"
    )

    print("Pipeline for MMR train finished successfully!")


run_mmr_pipeline(
    ratings_path = ratings_file_path,
    movies_path = movies_file_path,
    output_dir = ouput_dir,
    top_n = TOP_N,
    top_k = TOP_K,
    chunksize_movies= CHUNK_SIZE_MOVIES,
    k = K,
    alpha = ALPHA,
    lambda_= LAMDA_,
    n_epochs= N_EPOCHS,
    lambda_param= LAMBDA_PARAM)




# movie_user_rating, genre_map, all_genres = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=CHUNK_SIZE_MOVIES)

# R = movie_user_rating.values

# R_filtered, filtered_movie_titles = filter_empty_users_data(R, movie_user_rating.columns )


# # Train the model
# mf = MatrixFactorization(R_filtered, K, ALPHA, LAMDA_ , N_EPOCHS)
# mf.train()

# # Full predicted rating matrix
# predicted_ratings = mf.full_prediction()

# # Get top-N candidates for MMR
# get_top_n_recommendations_MF(
#     genre_map, predicted_ratings, R_filtered, 
#     movie_user_rating.index, filtered_movie_titles, 
#     top_n=TOP_N, save_path ="src/datasets/mmr_data/mf_train_predictions.csv" )


# # print top 10 movies of MMR
# movie_titles = movie_user_rating.columns.tolist()


# # Build MMR models
# mmr_cosine, mmr_jaccard = build_mmr_models(
#     movie_titles,
#     genre_map,
#     all_genres,
#     predicted_ratings,
#     LAMBDA_PARAM
# )

# # run mmr
# get_recommendations_for_mmr(
#     mmr_cosine, movie_user_rating, movie_titles, 
#     genre_map, predicted_ratings, TOP_K, TOP_N, 
#     base_dir, "cosine_test"
# )


# get_recommendations_for_mmr(
#     mmr_jaccard, movie_user_rating, movie_titles, 
#     genre_map, predicted_ratings, TOP_K, TOP_N,
#     base_dir, "jaccard_test"
# )
