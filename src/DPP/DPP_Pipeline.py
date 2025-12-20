import os
import sys
import time



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MMR.MF import (
    MatrixFactorization,
    load_and_prepare_matrix,
    get_top_n_recommendations_MF,
tune_mf, train_mf_with_best_params,
    save_mf_predictions
)

from MMR.helperFunctions import ( generate_run_id, align_matrix_to_user_items, align_matrix_to_user,
                                  prepare_train_val_matrices, get_filtered_predictions,
                                   log_experiment, build_mmr_input
                                  )


from DPP import (
    DPP, build_dpp_models, get_recommendations_for_dpp, save_DPP
)

import pandas as pd
import numpy as np
import csv

def build_dpp_input_per_user(
        candidate_list_csv,
        R_filtered,
        filtered_user_ids,
        filtered_item_ids,
):
    # Load CSV
    df = pd.read_csv(candidate_list_csv)
    df = df[df["userId"].isin(filtered_user_ids)]

    # Ensure type consistency
    df["itemId"] = df["itemId"].astype(str)
    filtered_item_ids_str = [str(i) for i in filtered_item_ids]

    # Filter items in filtered_item_ids
    df = df[df["itemId"].isin(filtered_item_ids_str)]

    #Build candidate items per user
    candidate_items_per_user = []
    predicted_ratings_per_user = []
    user_history_per_user = []

    user_to_row = {u: i for i, u in enumerate(filtered_user_ids)}

    for user_id in filtered_user_ids:
        user_df = df[df["userId"] == user_id]

        candidate_items = user_df["itemId"].tolist()
        num_items = len(candidate_items)

        # Build predicted ratings vector for this user
        predicted_ratings_vec = np.zeros(num_items)
        for j, item_id in enumerate(candidate_items):
            rating = user_df[user_df["itemId"] == item_id]["predictedRating"].values[0]
            predicted_ratings_vec[j] = rating

        # Build history mask for this user
        rated_indices = np.where(R_filtered[user_to_row[user_id]] > 0)[0]
        rated_item_ids = {str(filtered_item_ids[i]) for i in rated_indices if i < len(filtered_item_ids)}
        mask = np.array([item in rated_item_ids for item in candidate_items])

        candidate_items_per_user.append(candidate_items)
        predicted_ratings_per_user.append(predicted_ratings_vec)
        user_history_per_user.append(mask)

    return predicted_ratings_per_user, user_history_per_user, candidate_items_per_user



#pipeline start
def run_dpp_pipeline(
        run_id, ratings_train_path, ratings_val_path, item_path, output_dir, dataset=None, datasize=None,
        top_n=10, top_k=20, chunksize = 10000 , n_epochs=50, random_state = 42
):
    print(f"Starting pipeline for {dataset} train")

    os.makedirs(output_dir, exist_ok=True)

    # Load train/validation matrices (dataframes)
    item_user_rating_train, genre_map, all_genres = load_and_prepare_matrix(
        ratings_train_path, item_path)

    item_user_rating_val, _, _ = load_and_prepare_matrix(
        ratings_val_path, item_path,)


    # Use MMR helper to prepare aligned and filtered matrices
    (
        R_filtered_train,
        R_filtered_val,
        val_data_filtered,
        filtered_user_ids,
        filtered_item_ids,
    )= prepare_train_val_matrices(
        item_user_rating_train,
        item_user_rating_val,
    )


    # Tune MF hyperparameters
    best_params = tune_mf(
        R_train = R_filtered_train,
        R_val = R_filtered_val,
        n_epochs = n_epochs)
    print("→ Training Matrix Factorization...")

    # Train MF with best hyperparameters

    (
        mf, predicted_ratings,
        train_mse_history,
        best_train_rmse,
        val_mse_history,
        best_val_rmse,
        best_epoch
    ) = train_mf_with_best_params(
        R_filtered = R_filtered_train,
        R_val = R_filtered_val,
        best_params = best_params,
        n_epochs=n_epochs,
        random_state = random_state)


    # Attach filtered item ids to MF model (strings)
    mf.item_ids = filtered_item_ids


    # Build DPP models
    print("→ Building DPP models...")
    t0 = time.time()
    # Create a builder for cosine similarity
    build_start = time.time()
    build_dpp_cosine = build_dpp_models(filtered_item_ids, genre_map, all_genres, predicted_ratings, 'cosine')

    build_dpp_jaccard = build_dpp_models(filtered_item_ids, genre_map, all_genres, predicted_ratings, 'jaccard')
    build_end = time.time()
    print(f"DPP model build time: {build_end - build_start:.2f} sec")


    align_start = time.time()
    # Prepare item_user_rating_filtered aligned to filtered_item_ids and filtered_user_ids
    # align_matrix_to_items returns (aligned_matrix_numpy, aligned_df)
    _, item_user_rating_filtered_df = align_matrix_to_user_items(
        matrix_df = item_user_rating_train,
        filtered_item_ids = filtered_item_ids,
        filtered_user_ids = filtered_user_ids
    )
    align_end = time.time()
    print(f"Matrix alignment time: {align_end - align_start:.2f} sec")



    # Run DPP recommendations
    cos_start = time.time()
    get_recommendations_for_dpp(
        build_dpp_cosine, item_user_rating_filtered_df, filtered_item_ids,
        genre_map, predicted_ratings,  top_k, top_n,  "cosine")

    cos_end = time.time()
    print(f"DPP cosine runtime: {cos_end - cos_start:.2f} sec")


    jac_start = time.time()
    get_recommendations_for_dpp(
        build_dpp_jaccard, item_user_rating_filtered_df, filtered_item_ids,
        genre_map, predicted_ratings, top_k, top_n, "jaccard"
    )
    jac_end = time.time()

    print(f"DPP jaccard runtime: {jac_end - jac_start:.2f} sec")
    # Total time
    t1 = time.time()
    print(f"TOTAL DPP pipeline time: {t1 - t0:.2f} sec")

    print("DPP TRAIN pipeline completed successfully!")

    return (
        best_params,
        predicted_ratings,
        filtered_item_ids,
        filtered_user_ids,
        mf
    )

# Test pipeline:
def run_dpp_pipeline_test(
        run_id,
        ratings_path,
        ground_truth_path,
        item_path,
        output_dir=None,
        dataset=None,
        chunksize = 10000,
        top_n=10, top_k=20, trained_mf_model=None,
        train_filtered_user_ids=None,
        train_filtered_item_ids=None
):
    print(f"Start {dataset} test pipeline")

    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    item_user_rating, genre_map, all_genres = load_and_prepare_matrix(ratings_path, item_path)

    # Align test matrix to training users (to map to MF)
    R_filtered, filtered_df = align_matrix_to_user(
        matrix_df=item_user_rating,
        filtered_user_ids=train_filtered_user_ids
    )

    # Get predicted ratings for test users and items
    filtered_user_ids, filtered_item_ids, predicted_ratings = get_filtered_predictions(
        trained_mf_model, filtered_df, train_filtered_user_ids, train_filtered_item_ids
    )

    # Get top-N MF recommendations
    mf_top_n_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv")

    get_top_n_recommendations_MF(
        predicted_ratings=predicted_ratings,
        R_filtered=R_filtered,
        filtered_user_ids=filtered_user_ids,
        filtered_item_ids=filtered_item_ids,
        top_n=top_n,
        save_path=mf_top_n_path
    )

    # Build MMR input
    predicted_ratings_top_n, user_history_top_n, candidate_items = build_dpp_input_per_user(
        candidate_list_csv=mf_top_n_path,
        R_filtered=R_filtered,
        filtered_user_ids=filtered_user_ids,
        filtered_item_ids=filtered_item_ids
    )
    _, _, candidate_items_list = build_mmr_input(
        candidate_list_csv=mf_top_n_path,
        R_filtered=R_filtered,
        filtered_user_ids=filtered_user_ids,
        filtered_item_ids=filtered_item_ids
    )

    print(f"Candidate items: {len(candidate_items)}")

    # Save MF predictions for reference
    dataset_root = os.path.dirname(output_dir)
    mf_predictions_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_predictions.csv")

    save_mf_predictions(
        trained_mf_model=trained_mf_model,
        train_user_ids=train_filtered_user_ids,
        train_item_ids=train_filtered_item_ids,
        ground_truth_path=ground_truth_path,
        output_path=mf_predictions_path
    )

    # Build DPP models
    genre_map_test = {item: genre_map[item] for item in candidate_items_list if item in genre_map}
    all_genres_test = sorted({g for genres in genre_map_test.values() for g in genres})


    # SANITY CHECK
    gt_items_test = item_user_rating.columns[item_user_rating.sum(axis=0) > 0]  # all items with any ratings in test
    num_gt_in_candidates = sum(item in candidate_items_list for item in gt_items_test)
    print(f"Candidate items for DPP: {len(candidate_items_list)}")
    print(f"Number of GT items included in candidates: {num_gt_in_candidates}")
    if num_gt_in_candidates == 0:
        print("Warning: No GT items in DPP candidate pool! GT metrics will be zero.")

    # Use full predicted ratings (not top-N)
    predicted_ratings_dpp = predicted_ratings

    dpp_cosine = build_dpp_models(filtered_item_ids, genre_map_test, all_genres_test, predicted_ratings_dpp, 'cosine')
    dpp_jaccard = build_dpp_models(filtered_item_ids, genre_map_test, all_genres_test, predicted_ratings_dpp, 'jaccard')

    # Run DPP recommendations
    cosine_reco = get_recommendations_for_dpp(
        dpp_cosine, filtered_df, filtered_item_ids, genre_map_test, predicted_ratings_dpp,
        top_k, top_n, "cosine", candidate_items_per_user=candidate_items,   # from build_dpp_input()
        user_history_per_user=user_history_top_n
    )
    jaccard_reco = get_recommendations_for_dpp(
        dpp_jaccard, filtered_df, filtered_item_ids, genre_map_test, predicted_ratings_dpp,
        top_k, top_n, "jaccard", candidate_items_per_user=candidate_items,
        user_history_per_user=user_history_top_n
    )
    # Save DPP results
    cosine_path = os.path.join(output_dir, f"{run_id}/dpp_test_{chunksize}_cosine_top_{top_n}.csv")
    save_DPP(cosine_reco, cosine_path)

    jaccard_path = os.path.join(output_dir, f"{run_id}/dpp_test_{chunksize}_jaccard_top_{top_n}.csv")
    save_DPP(jaccard_reco, jaccard_path)

    print("DPP TEST pipeline completed successfully!")




if __name__ == "__main__":


    # PARAMETER
    TOP_N = 10
    CHUNK_SIZE = 100000
    K = 20
    ALPHA = 0.01
    LAMDA_ = 0.1
    N_EPOCHS = 50
    TOP_K = 50
    LAMBDA_PARAM = 0.7
    #DATASET_NAME = "books"
    RANDOM_STATE = 42



    #load data
    dataset_books = "books"
    folder_books = "GoodBooks"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", "ratingsbooks_100K.csv")
    book_ground_truth = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_test.csv")
    item_file_path = os.path.join(base_dir, f"../datasets/{folder_books}", f"{dataset_books}.csv")

    output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{dataset_books}")


    run_book_id = generate_run_id()


    best_params, predicted_ratings, filtered_item_ids, filtered_user_ids, mf = run_dpp_pipeline(
        run_id = run_book_id,
        ratings_train_path = ratings_train_file,
        ratings_val_path= ratings_val_file,
        item_path = item_file_path,
        output_dir = output_dir,
        top_n = TOP_N,
        top_k = TOP_K,
        chunksize= CHUNK_SIZE,
        n_epochs= N_EPOCHS,
        dataset=dataset_books,
        random_state=RANDOM_STATE)


    #Run MF pipeline for test dataset
    run_dpp_pipeline_test(
        run_id = run_book_id,
        ratings_path=ratings_test_path,
        ground_truth_path = book_ground_truth,
        item_path=item_file_path,
        output_dir= output_dir,
        dataset= dataset_books,
        chunksize=CHUNK_SIZE,
        top_n=TOP_N,
        top_k=TOP_K,
        trained_mf_model = mf,
        train_filtered_user_ids=filtered_user_ids,
        train_filtered_item_ids=filtered_item_ids
    )



    # Now start movies block at correct indentation
    dataset_movie = "movies"
    folder_movie = "MovieLens"

    #base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", "ratings_100K_movies.csv")
    movies_ground_truth = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_test.csv")

    item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"{dataset_movie}.csv")

    output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{dataset_movie}")

    run_movie_id = generate_run_id()

    best_params, predicted_ratings, filtered_item_ids, filtered_user_ids, mf = run_dpp_pipeline(
        run_id = run_movie_id,
        ratings_train_path = ratings_train_file,
        ratings_val_path= ratings_val_file,
        item_path = item_file_path,
        output_dir = output_dir,
        top_n = TOP_N,
        top_k = TOP_K,
        chunksize= CHUNK_SIZE,
        n_epochs= N_EPOCHS,
        dataset=dataset_movie,
        random_state=RANDOM_STATE)


    #Run MF pipeline for test dataset
    run_dpp_pipeline_test(
        run_id = run_movie_id,
        ratings_path=ratings_test_path,
        ground_truth_path = movies_ground_truth,
        item_path=item_file_path,
        output_dir= output_dir,
        dataset= dataset_movie,
        chunksize=CHUNK_SIZE,
        top_n=TOP_N,
        top_k=TOP_K,
        trained_mf_model = mf,
        train_filtered_user_ids=filtered_user_ids,
        train_filtered_item_ids=filtered_item_ids
    )