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
    DPP, build_dpp_models, get_recommendations_for_dpp, save_DPP, get_recommendations_for_dpp_test
)


import pandas as pd
import numpy as np
import csv


def build_dpp_input_from(
        candidate_list_csv,
        interactions_df = None,
):
    # Load MF candidate list
    df = pd.read_csv(candidate_list_csv)
    df["userId"] = df["userId"].astype(str)
    df["itemId"] = df["itemId"].astype(str)

    # Extract users & items
    user_ids = df["userId"].unique().tolist()
    candidate_items = df["itemId"].unique().tolist()

    # Map users/items to row/column indices in the predicted rating matrix
    user_to_row = {u: i for i, u in enumerate(user_ids)}
    item_to_col = {i: j for j, i in enumerate(candidate_items)}

    num_users = len(user_ids)
    num_items = len(candidate_items)

    # Initialize predicted ratings matrix (users x candidate items)
    predicted_ratings = np.zeros((num_users, num_items))

    for _, row in df.iterrows():
        predicted_ratings[
            user_to_row[row["userId"]],
            item_to_col[row["itemId"]],
        ] = row["predictedRating"]

    # Build user history mask
    user_history = None

    if interactions_df is not None:
        interactions_df["userId"] = interactions_df["userId"].astype(str)
        interactions_df["itemId"] = interactions_df["itemId"].astype(str)

        user_history = []

        for u in user_ids:
            seen_items = set(
                interactions_df.loc[
                    interactions_df["userId"] == u, "itemId"
                ]
            )

            mask = np.array(
                [item in seen_items for item in candidate_items],
                dtype=bool,
            )
            user_history.append(mask)
    else:
        # Cold-start safe fallback
        user_history = [
            np.zeros(num_items, dtype=bool)
            for _ in range(num_users)
        ]

    # Explicit per-user candidate list (DPP API expects this)
    candidate_items_per_user = [
        candidate_items for _ in range(num_users)
    ]

    return (
        predicted_ratings,
        user_history,
        user_ids,
        candidate_items,
        candidate_items_per_user,
    )



#pipeline start
def run_dpp_pipeline(
        run_id, ratings_train_path, ratings_val_path, item_path, output_dir, dataset=None, datasize=None,
        top_n=10, top_k=20, chunksize = 10000 , n_epochs=50, relevance_weight = None, diversity_weight=None, random_state = 42
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
        ratings_train_path,
        ratings_test_path,
        ground_truth_path,
        item_path,
        output_dir=None,
        dataset=None,
        chunksize = 10000,
        top_n=10, top_k=50, trained_mf_model=None,
        train_filtered_user_ids=None,
        train_filtered_item_ids=None
):

    print(f"Start {dataset} test pipeline ")

    # Ensure the output directory exists for saving results
    os.makedirs(output_dir, exist_ok=True)

    # Load the user-item rating matrix and item genre metadata
    item_user_rating, genre_map, all_genres= load_and_prepare_matrix(
        ratings_train_path, item_path)

    # Ensure train_filtered_user_ids are all ints
    train_filtered_user_ids = [int(uid) for uid in train_filtered_user_ids]

    # Load unseen test data
    test_df = pd.read_csv(ratings_test_path)
    #test_df["userId"] = test_df["userId"].astype(str)
    #test_df["itemId"] = test_df["itemId"].astype(str)


    # Keep only users that exist in the trained MF model
    existing_test_df = test_df[test_df['userId'].isin(train_filtered_user_ids)].copy()
    # Make userId the DataFrame index.
    existing_test_df.set_index('userId', inplace=True)
    test_user_ids = existing_test_df.index.unique()
    test_item_ids = existing_test_df['itemId'].unique()


    # Extract predicted ratings for filtered users and items from the trained MF model
    predicted_ratings = get_filtered_predictions(
        trained_mf_model,
        test_user_ids,
        train_filtered_user_ids,
    )

    # Generate top-N recommendations for each user from MF predictions
    get_top_n_recommendations_MF(
        predicted_ratings=predicted_ratings,
        R_filtered=item_user_rating.values,
        filtered_user_ids=test_user_ids,
        filtered_item_ids=test_item_ids,
        top_n=top_n,
        save_path=os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv"))

    # Use the top-N MF recommendations as the candidate list for MMR
    candidate_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv")

    ratings_df = pd.read_csv(ratings_train_path)[["userId", "itemId"]]
    ratings_df["userId"] = ratings_df["userId"].astype(str)
    ratings_df["itemId"] = ratings_df["itemId"].astype(str)

    (
        predicted_ratings_top_n,
        user_history_top_n,
        user_ids,
        candidate_items,
        candidate_items_per_user,
    ) = build_dpp_input_from(
        candidate_list_csv=candidate_path,
        interactions_df=ratings_df,
    )

    assert user_ids == list(map(str, test_user_ids))
    assert predicted_ratings_top_n.shape == (
        len(user_ids),
        len(candidate_items),
    )
    assert len(user_history_top_n) == len(user_ids)





    # Save the full MF predictions to CSV
    save_mf_predictions(
        trained_mf_model=trained_mf_model,
        train_user_ids=train_filtered_user_ids,
        train_item_ids=train_filtered_item_ids,
        ground_truth_path=ground_truth_path,
        output_path=os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_predictions.csv")
    )



    movie_user_rating = pd.DataFrame(
        0.0,
        index=user_ids,
        columns=candidate_items,
    )

    for _, row in test_df.iterrows():
        u, i = row["userId"], row["itemId"]
        if u in movie_user_rating.index and i in movie_user_rating.columns:
            movie_user_rating.at[u, i] = 1.0


    # Build DPP models
    genre_map_test = {item: genre_map[item] for item in candidate_items if item in genre_map}
    all_genres_test = sorted({g for genres in genre_map_test.values() for g in genres})

    print(f"Candidate items: {len(candidate_items)}")
    print(f"Test users: {len(user_ids)}")
    print(f"GT interactions: {movie_user_rating.values.sum()}")


    # SANITY CHECK
    print(f"Candidate items for DPP: {len(candidate_items)}")
    print(f"Number of test users in ground truth matrix: {movie_user_rating.shape[0]}")
    print(f"Number of GT interactions: {movie_user_rating.values.sum()}")
    gt_items_test = movie_user_rating.columns[movie_user_rating.sum(axis=0) > 0]
    num_gt_in_candidates = sum(item in candidate_items for item in gt_items_test)
    print(f"GT items included in candidate pool: {num_gt_in_candidates}")
    if num_gt_in_candidates == 0:
        print("Warning: No GT items in DPP candidate pool! Metrics will be zero.")


    # Use full predicted ratings (not top-N)
    predicted_ratings_dpp = predicted_ratings_top_n

    dpp_cosine = build_dpp_models(candidate_items, genre_map_test, all_genres_test, predicted_ratings_dpp, 'cosine')
    dpp_jaccard = build_dpp_models(candidate_items, genre_map_test, all_genres_test, predicted_ratings_dpp, 'jaccard')

    # Run DPP recommendations
    cosine_reco = get_recommendations_for_dpp_test(
        dpp_cosine, movie_user_rating, candidate_items, genre_map_test, predicted_ratings_dpp,
        top_k, top_n, "cosine", candidate_items_per_user=candidate_items_per_user,   # from build_dpp_input()
        user_history_per_user=user_history_top_n
    )
    jaccard_reco = get_recommendations_for_dpp_test(
        dpp_jaccard, movie_user_rating, candidate_items, genre_map_test, predicted_ratings_dpp,
        top_k, top_n, "jaccard", candidate_items_per_user=candidate_items_per_user,
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
    TOP_N = 100
    CHUNK_SIZE = 100000
    CHUNK_SIZE_NAME = "100K"
    K = 20
    ALPHA = 0.01
    LAMDA_ = 0.1
    N_EPOCHS = 50
    TOP_K = 20
    LAMBDA_PARAM = 0.7
    #DATASET_NAME = "books"
    RANDOM_STATE = 42


    #load data
    dataset_books = "books"
    folder_books = "GoodBooks"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    books_ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_train.csv")
    books_ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_val.csv")
    books_ground_truth = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_test.csv")
    books_ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", "ratingsbooks_100K.csv")
    books_item_file_path = os.path.join(base_dir, f"../datasets/{folder_books}", f"{dataset_books}.csv")

    books_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{dataset_books}")




    run_book_id = generate_run_id()


    best_params, predicted_ratings, filtered_item_ids, filtered_user_ids, mf = run_dpp_pipeline(
        run_id = run_book_id,
        ratings_train_path = books_ratings_train_file,
        ratings_val_path= books_ratings_val_file,
        item_path = books_item_file_path,
        output_dir = books_output_dir,
        dataset=dataset_books,
        top_n = TOP_N,
        top_k = TOP_K,
        chunksize= CHUNK_SIZE_NAME,
        n_epochs= N_EPOCHS,
        random_state=RANDOM_STATE)


    #Run MF pipeline for test dataset
    #run_dpp_pipeline_test(
    #    run_id = run_book_id,
    #    ratings_train_path=books_ratings_train_file,
    #    ratings_test_path=books_ratings_test_path,
    #    ground_truth_path = books_ground_truth,
    #    item_path=books_item_file_path,
    #    output_dir=books_output_dir,
    #    dataset= dataset_books,
    #    chunksize=CHUNK_SIZE,
    #    top_n=TOP_N,
    #    top_k=TOP_K,
    #    trained_mf_model = mf,
    #    train_filtered_user_ids=filtered_user_ids,
    #    train_filtered_item_ids=filtered_item_ids
    #)




    #load MovieLens data
    dataset_movie = "movies"
    folder_movie = "MovieLens"
    movies_ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_train.csv")
    movies_ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_val.csv")
    movies_ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", "ratings_100K_movies.csv")
    movies_ground_truth = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_test.csv")
    movies_item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"{dataset_movie}.csv")
    movies_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{dataset_movie}")

    #run_movie_id = generate_run_id()

    #best_params, predicted_ratings, filtered_item_ids, filtered_user_ids, mf = run_dpp_pipeline(
    #    run_id = run_movie_id,
    #    ratings_train_path = movies_ratings_train_file,
    #    ratings_val_path= movies_ratings_val_file,
    #    item_path = movies_item_file_path,
    #    output_dir = movies_output_dir,
    #    dataset=dataset_movie,
    #    top_n = TOP_N,
    #    top_k = TOP_K,
    #    chunksize= CHUNK_SIZE_NAME,
    #    n_epochs= N_EPOCHS,
    #    random_state=RANDOM_STATE)


    #Run MF pipeline for test dataset
    #run_dpp_pipeline_test(
    #    run_id = run_movie_id,
    #    ratings_train_path=movies_ratings_train_file,
    #    ratings_test_path=movies_ratings_test_path,
    #    ground_truth_path = movies_ground_truth,
    #    item_path=movies_item_file_path,
    #    output_dir=movies_output_dir,
    #    dataset= dataset_movie,
    #    chunksize=CHUNK_SIZE,
    #    top_n=TOP_N,
    #    top_k=TOP_K,
    #    trained_mf_model = mf,
    #    train_filtered_user_ids=filtered_user_ids,
    #    train_filtered_item_ids=filtered_item_ids
    #)