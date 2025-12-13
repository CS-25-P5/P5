import csv
import os
import sys
import tracemalloc
import time
import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MMR.MF import (
    MatrixFactorization,
    load_and_prepare_matrix,
    get_top_n_recommendations_MF,
tune_mf, train_mf_with_best_params,
    save_mf_predictions
)




from DPP import (
    DPP, build_dpp_models, get_recommendations_for_dpp, save_DPP
)

import os
import pandas as pd
import numpy as np

def generate_run_id():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}"
    return run_id


def align_matrix_to_items(matrix_df, filtered_item_ids, filtered_user_ids):
    # Filter users that exist in matrix_df
    user_indices = [matrix_df.index.get_loc(u) for u in filtered_user_ids if u in matrix_df.index]
    # Filter items that exist in matrix_df
    item_indices = [matrix_df.columns.get_loc(i) for i in filtered_item_ids if i in matrix_df.columns]

    aligned_matrix = matrix_df.values[np.ix_(user_indices, item_indices)]
    aligned_df = matrix_df.iloc[user_indices, item_indices]

    return aligned_matrix, aligned_df


def prepare_train_val_matrices(train_df, val_df, id_to_title=None):

    # Align train and val to common users/items
    common_users = train_df.index.intersection(val_df.index)
    common_items = train_df.columns.intersection(val_df.columns)

    train_aligned = train_df.loc[common_users, common_items]
    val_aligned = val_df.loc[common_users, common_items]

    # Convert to numpy and remove users with no interactions
    R_train = train_aligned.values
    user_filter = R_train.sum(axis=1) > 0
    R_filtered_train = R_train[user_filter, :]
    filtered_user_ids = train_aligned.index[user_filter].tolist()
    filtered_item_ids = train_aligned.columns.tolist()
    filtered_item_titles = [id_to_title[i] for i in filtered_item_ids]

    R_filtered_val, val_data_filtered = align_matrix_to_items(
        val_aligned,
        filtered_item_ids,
        filtered_user_ids
    )

    # Log shapes for debugging
    print(f"Train matrix: {R_filtered_train.shape}, Val matrix: {R_filtered_val.shape}")

    return R_filtered_train, R_filtered_val,  val_data_filtered, filtered_user_ids, filtered_item_ids, filtered_item_titles

def get_filtered_predictions(trained_mf_model, filtered_df, train_filtered_user_ids, filtered_item_ids=None):
    # Get the filtered user and item IDs from the aligned DataFrame
    filtered_user_ids = filtered_df.index.tolist()
    filtered_item_ids = filtered_df.columns.tolist()
    print(f"Filtered users: {len(filtered_user_ids)}, Filtered items: {len(filtered_item_ids)}")

    # Align filtered items to MF model
    trained_items = np.array([str(i) for i in trained_mf_model.item_ids])
    filtered_item_ids_str = np.array([str(i) for i in filtered_item_ids])

    item_mask = np.isin(trained_items, filtered_item_ids_str)
    item_indices_in_mf = np.where(item_mask)[0]

    # item_indices_in_mf = []
    # for item_id in filtered_item_ids_str:
    #     # Check if the item exists in trained MF
    #     if item_id in trained_items:
    #         index = np.where(trained_items == item_id)[0][0]
    #         item_indices_in_mf.append(index)


    # Get the predicted ratings for the filtered items
    predicted_ratings_all = trained_mf_model.full_prediction()[:, item_indices_in_mf]


    # Map training user IDs to MF model indices
    mf_user_to_idx = {}
    for idx, user_id in enumerate(train_filtered_user_ids):
        user_str = str(user_id)   # convert user ID to string
        mf_user_to_idx[user_str] = idx

    # Get indices of test users in MF predictions
    test_user_indices = []
    for user_id in filtered_user_ids:
        user_str = str(user_id)
        if user_str in mf_user_to_idx:
            test_user_indices.append(mf_user_to_idx[user_str])
        else:
            #test_user_indices.append(0)  #
            raise ValueError(f"User {user_id} not in MF model")

    # Extract only predictions for test users
    predicted_ratings = predicted_ratings_all[test_user_indices, :]

    return filtered_user_ids, filtered_item_ids, predicted_ratings


def prepare_top_n_data(all_recommendations, filtered_item_ids, filtered_user_ids, predicted_ratings, R_filtered):
    #Keep order, remove duplicates
    top_n_items = []
    seen_items = set()
    for user_id, indices in all_recommendations.items():
        for idx in indices:
            item_id = filtered_item_ids[idx]
            if item_id not in seen_items:
                top_n_items.append(item_id)
                seen_items.add(item_id)

    # Map top_n_items to columns in predicted_ratings
    item_idx_map = [filtered_item_ids.index(i) for i in top_n_items]
    predicted_ratings_top_n = predicted_ratings[:, item_idx_map]

    # sanity check
    #assert predicted_ratings_top_n.shape[1] == len(top_n_items), "Predicted ratings and items mismatch!"

    # Create user histories aligned to top-N items
    user_history_top_n = []
    num_top_items = len(top_n_items)  # should be 21 in your case

    for user_idx, user_id in enumerate(filtered_user_ids):
        items_user_has_rated = np.where(R_filtered[user_idx, :] > 0)[0]
        item_ids_user_has_rated = [filtered_item_ids[i] for i in items_user_has_rated]

        # Boolean mask aligned to top_n_items
        mask = np.zeros(num_top_items, dtype=bool)
        for i, item_id in enumerate(top_n_items):
            if item_id in item_ids_user_has_rated:
                mask[i] = True

        user_history_top_n.append(mask)

    return predicted_ratings_top_n, user_history_top_n



def log_experiment(output_dir, file_name, params):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir,file_name)

    #Cheeck if file exists
    file_exists = os.path.isfile(log_file)

    #Write to Csv
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=params.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(params)

    print(f"Logged experiment to {log_file}")



def run_dpp_pipeline(
        run_id, ratings_train_path, ratings_val_path, item_path, output_dir, dataset=None, datasize=None,
        top_n=10, top_k=20, chunksize = 10000 , n_epochs=50, random_state = 42
):
    print(f"Starting pipeline for {dataset} train")

    os.makedirs(output_dir, exist_ok=True)

    # Load train/validation matrices (dataframes)
    item_user_rating_train, genre_map, all_genres, id_to_title = load_and_prepare_matrix(ratings_train_path, item_path)
    item_user_rating_val, _, _, _ = load_and_prepare_matrix(ratings_val_path, item_path)

    # Use MMR helper to prepare aligned and filtered matrices
    R_filtered_train, R_filtered_val, val_data_filtered, filtered_user_ids, filtered_item_ids, filtered_item_titles = prepare_train_val_matrices(
        train_df = item_user_rating_train,
        val_df = item_user_rating_val,
        id_to_title = id_to_title
    )

    # Tune MF hyperparameters
    best_params = tune_mf(R_filtered_train, R_filtered_val, n_epochs=n_epochs)

    print("→ Training Matrix Factorization...")

    # Train MF with best hyperparameters
    mf, predicted_ratings, train_rmse, random_state = train_mf_with_best_params(
        R_filtered_train,
        best_params,
        n_epochs=n_epochs,
        random_state = random_state)

    # Validation RMSE (if R_filtered_val available)
    val_rmse = mf.compute_rmse(R_filtered_val, predicted_ratings)

    # Attach filtered item ids to MF model (strings)
    mf.item_ids = filtered_item_ids



    # Top-N MF recommendations
    #get_top_n_recommendations_MF(
     #   genre_map, predicted_ratings, R_filtered_train,
      #  filtered_user_ids, filtered_item_ids,
        #top_n=top_n,
        #save_path=os.path.join(output_dir, run_id, f"{run_id}_mf_train_top_{top_n}.csv")
    #)

    # Build DPP models
    print("→ Building DPP models...")
    t0 = time.time()
    # Create a builder for cosine similarity
    build_start = time.time()
    build_dpp_cosine = build_dpp_models(filtered_item_titles, genre_map, all_genres, predicted_ratings, 'cosine')

    build_dpp_jaccard = build_dpp_models(filtered_item_titles, genre_map, all_genres, predicted_ratings, 'jaccard')
    build_end = time.time()
    print(f"DPP model build time: {build_end - build_start:.2f} sec")


    align_start = time.time()
    # Prepare item_user_rating_filtered aligned to filtered_item_ids and filtered_user_ids
    # align_matrix_to_items returns (aligned_matrix_numpy, aligned_df)
    _, item_user_rating_filtered_df = align_matrix_to_items(
        matrix_df = item_user_rating_train,
        filtered_item_ids = filtered_item_ids,
        filtered_user_ids = filtered_user_ids
    )
    align_end = time.time()
    print(f"Matrix alignment time: {align_end - align_start:.2f} sec")



    # Run DPP recommendations
    cos_start = time.time()
    get_recommendations_for_dpp(
        build_dpp_cosine, item_user_rating_filtered_df, filtered_item_titles,
        genre_map, predicted_ratings, id_to_title, top_k, top_n,
        output_dir, "cosine"
    )
    cos_end = time.time()
    print(f"DPP cosine runtime: {cos_end - cos_start:.2f} sec")


    jac_start = time.time()
    get_recommendations_for_dpp(
        build_dpp_jaccard, item_user_rating_filtered_df, filtered_item_titles,
        genre_map, predicted_ratings, id_to_title, top_k, top_n,
        output_dir, "jaccard"
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
        item_path,
        output_dir=None,
        dataset=None,
        chunksize = 10000,
        top_n=10, top_k=20, trained_mf_model=None,
        train_filtered_user_ids=None,
        train_filtered_item_ids=None
):
    print(f"Start {dataset} test pipeline ")

    os.makedirs(output_dir, exist_ok=True)

    # Load test data (dataframe)
    item_user_rating, genre_map, all_genres, id_to_title = load_and_prepare_matrix(
        ratings_path, item_path)

    R_filtered, filtered_df = align_matrix_to_items(
        matrix_df=item_user_rating,
        filtered_item_ids=train_filtered_item_ids,
        filtered_user_ids=train_filtered_user_ids
    )

    filtered_user_ids, filtered_item_ids, predicted_ratings = get_filtered_predictions(
        trained_mf_model, filtered_df, train_filtered_user_ids, train_filtered_item_ids)



    # Get top-N candidates for MMR
    mf_top_n_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv")


    all_recommendations = get_top_n_recommendations_MF(
                                genre_map=genre_map,
                                predicted_ratings=predicted_ratings,
                                R_filtered=R_filtered,
                                filtered_user_ids=filtered_user_ids,
                                filtered_item_ids=filtered_item_ids,
                                id_to_title=id_to_title,
                                top_n=top_n,
                                save_path=mf_top_n_path)

    predicted_ratings_top_n, user_history_top_n = prepare_top_n_data(all_recommendations, filtered_item_ids, filtered_user_ids, predicted_ratings, R_filtered)

    # Define output path for MF predictions
    dataset_root = os.path.dirname(output_dir)
    mf_predictions_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_predictions.csv")
    ground_truth_path = os.path.join(dataset_root, f"{dataset}_ratings_{chunksize}_test.csv")

    # Save MF predictions
    save_mf_predictions(
        trained_mf_model=trained_mf_model,
        train_user_ids=train_filtered_user_ids,
        train_item_ids=train_filtered_item_ids,
        ground_truth_path=ground_truth_path,
        output_path=mf_predictions_path
    )



    # Build DPP models using test predictions
    genre_map_test = {item: genre_map[item] for item in filtered_item_ids if item in genre_map}

    all_genres_test = sorted({g for genres in genre_map_test.values() for g in genres})

    dpp_cosine = build_dpp_models(filtered_item_ids, genre_map, all_genres_test, predicted_ratings_top_n, 'cosine')
    dpp_jaccard = build_dpp_models(filtered_item_ids, genre_map, all_genres_test, predicted_ratings_top_n, 'jaccard')

    dpp_path_cosine = os.path.join(output_dir,f"{run_id}/dpp_test_{chunksize}_cosine_top_{top_n}.csv")
    dpp_path_jaccard = os.path.join(output_dir,f"{run_id}/dpp_test_{chunksize}_jaccard_top_{top_n}.csv")


    # Build candidate item set (union of all MF top-N results)
    top_n_items = sorted({
        filtered_item_ids[idx]
        for user_items in all_recommendations.values()
        for idx in user_items
    })

    filtered_df_top_n = filtered_df[top_n_items]



# Run DPP recommendations on test
    cosine_reco =  get_recommendations_for_dpp(
        dpp_cosine, filtered_df_top_n, filtered_item_ids, genre_map, predicted_ratings_top_n, id_to_title,
        top_k, top_n, dpp_path_cosine, "cosine"
    )

    jaccard_rec = get_recommendations_for_dpp(
        dpp_jaccard, filtered_df_top_n, filtered_item_ids, genre_map, predicted_ratings_top_n, id_to_title,
        top_k, top_n, dpp_path_jaccard, "jaccard"
    )



    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    test_path = os.path.join(output_dir,f"{run_id}/dpp_test_{chunksize}_cosine_top_{top_n}.csv")
    save_DPP(cosine_reco, test_path)

    test_path = os.path.join(output_dir,f"{run_id}/dpp_test_{chunksize}_jaccard_top_{top_n}.csv")
    save_DPP(jaccard_rec, test_path)



    print("DPP TEST pipeline completed successfully!")




if __name__ == "__main__":

    # PARAMETER
    TOP_N = 10
    CHUNK_SIZE = 1000
    K = 20
    ALPHA = 0.01
    LAMDA_ = 0.1
    N_EPOCHS = 50
    TOP_K = 50
    LAMBDA_PARAM = 0.7
    #DATASET_NAME = "books"
    RANDOM_STATE = 42

    base_dir = os.path.dirname(os.path.abspath(__file__))

    #load data
    dataset_books = "books"
    folder_books = "GoodBooks"
    ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_test.csv")
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

    #load data
    dataset_movie = "movies"
    folder_movie = "MovieLens"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_test.csv")
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