import numpy as np
from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF, log_mf_experiment,  tune_mf, train_mf_with_best_params, align_train_val_matrices
from MMR import MMR, build_mmr_models, get_recommendations_for_mmr
import os
import pandas as pd




def run_mmr_pipeline(
    ratings_train_path, ratings_val_path , item_path, output_dir=None, dataset=None, datasize=None,
    top_n=10, top_k=20, chunksize=10000, n_epochs=50, lambda_param=0.7):

    os.makedirs(output_dir, exist_ok=True)


    #load data
    item_user_rating_train, genre_map, all_genres = load_and_prepare_matrix( 
        ratings_train_path, item_path,nrows_items=chunksize)
    

    item_user_rating_val, genre_map, all_genres = load_and_prepare_matrix( 
    ratings_val_path, item_path, nrows_items=chunksize)

    train_aligned, val_aligned= align_train_val_matrices(item_user_rating_train, item_user_rating_val )

    #Convert to numpy arrays
    R_train = train_aligned.values
    R_val = val_aligned.values

    R_filtered_train, filtered_item_titles = filter_empty_users_data(
    R_train,
    train_aligned.columns
    )


    #Make sure valdaiton uses the same columns
    filtered_indices = [
        train_aligned.columns.get_loc(m) 
        for m in filtered_item_titles]

    R_filtered_val = R_val[:, filtered_indices]
    

    # Tune MF parameters 
    best_params = tune_mf(
        R_train=R_filtered_train, 
        R_val = R_filtered_val, 
        n_epochs=n_epochs)

    # Train MF with best hyperparameters
    mf, predicted_ratings, train_rmse = train_mf_with_best_params(R_filtered_train, best_params, n_epochs=n_epochs)
    val_rmse = mf.compute_rmse(R_filtered_val, predicted_ratings)

    # #Train MF
    # mf = MatrixFactorization(R_filtered, k, alpha, lambda_, n_epochs)
    # train_rmse = mf.train()

    # # Full predicted rating matrix
    # predicted_ratings = mf.full_prediction()


    # Log hyperparameters and results
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


    # Get top-N candidates for MF
    get_top_n_recommendations_MF(
        genre_map, predicted_ratings, R_filtered_train, 
        item_user_rating_train.index, filtered_item_titles, 
        top_n=top_n, 
        save_path = os.path.join(output_dir,"mf_train_predictions.csv"))

    # get top 10 movies of MMR
    movie_titles = filtered_item_titles

 

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
        mmr_cosine, R_filtered_train, item_user_rating_train, movie_titles, 
        genre_map, predicted_ratings, top_k, top_n, 
        output_dir, "cosine_test"
    )


    get_recommendations_for_mmr(
        mmr_jaccard, R_filtered_train,  item_user_rating_train, movie_titles, 
        genre_map, predicted_ratings, top_k, top_n,
        output_dir, "jaccard_test"
    )

    print("Pipeline for MMR train finished successfully!")

    return best_params

def run_mf_pipeline(
    ratings_path, item_path, output_dir=None, top_n=10, chunksize=10000,
    k=20, alpha=0.01, lambda_=0.1, n_epochs=50
):

    os.makedirs(output_dir, exist_ok=True)


    # Load and prepare data
    item_user_rating, genre_map, all_genres = load_and_prepare_matrix(
        ratings_path, item_path, nrows_items=chunksize)

    R = item_user_rating.values

    R_filtered, filtered_movie_titles = filter_empty_users_data(R,item_user_rating.columns )

    # Train the model
    mf = MatrixFactorization(R_filtered, k, alpha, lambda_ , n_epochs)
    mf.train()


    # Full predicted rating matrix
    predicted_ratings = mf.full_prediction()


    # Get top-N candidates for MMR
    save_path = os.path.join(output_dir, f"mf_test_predictions.csv")
    get_top_n_recommendations_MF(
        genre_map, predicted_ratings, R_filtered, 
        item_user_rating.index, filtered_movie_titles, 
        top_n=top_n, save_path=save_path)
    

    print("Pipeline for MF train finished successfully!")




if __name__ == "__main__":
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
    ratings_train_file= os.path.join(base_dir, "../datasets/mmr_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/mmr_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/mmr_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_test.csv")
    movies_file_path = os.path.join(base_dir, "../datasets/MovieLens", "movies.csv")

    output_dir = os.path.join(base_dir,"../datasets/mmr_data/movie")

    best_params = run_mmr_pipeline(
        ratings_train_path = ratings_train_file,
        ratings_val_path= ratings_val_file,
        item_path = movies_file_path,
        output_dir = output_dir,
        top_n = TOP_N,
        top_k = TOP_K,
        chunksize= CHUNK_SIZE,
        n_epochs= N_EPOCHS,
        lambda_param= LAMBDA_PARAM,
        dataset=DATASET_NAME,
        datasize=CHUNK_SIZE)
    

    # Run MF pipeline for test dataset
    run_mf_pipeline(
        ratings_path=ratings_test_path,
        item_path=movies_file_path,
        output_dir=output_dir,
        top_n=TOP_N,
        chunksize=CHUNK_SIZE,
        k=best_params["k"],
        alpha=best_params["alpha"],
        lambda_=best_params["lambda_"],
        n_epochs=N_EPOCHS
    )



