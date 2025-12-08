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
tune_mf, train_mf_with_best_params
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
        top_n=10, top_k=20, chunksize = 10000 , n_epochs=50, similarity_types = ["cosine", "jaccard"]
):
    os.makedirs(output_dir, exist_ok=True)

    # Load train/validation matrices (dataframes)
    item_user_rating_train, genre_map, all_genres, id_to_title = load_and_prepare_matrix(ratings_train_path, item_path, nrows_items=chunksize)
    item_user_rating_val, _, _, _ = load_and_prepare_matrix(ratings_val_path, item_path, nrows_items=chunksize)

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
    mf, predicted_ratings, train_rmse, random_state = train_mf_with_best_params(R_filtered_train, best_params, n_epochs=n_epochs)

    # Validation RMSE (if R_filtered_val available)
    val_rmse = mf.compute_rmse(R_filtered_val, predicted_ratings) if (R_filtered_val is not None and R_filtered_val.size>0) else float('nan')

    # Attach filtered item ids to MF model (strings)
    mf.item_ids = filtered_item_ids

    # Save full MF prediction matrix (train) — the full matrix
    os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)
    np.save(os.path.join(output_dir, run_id, f"mf_full_pred_matrix_{run_id}.npy"), predicted_ratings)

    # export as CSV (careful with size)
    # pd.DataFrame(predicted_ratings, index=filtered_user_ids, columns=filtered_item_ids).to_csv(
    #     os.path.join(output_dir, run_id, f"mf_full_pred_matrix_{run_id}.csv"), index=True
    # )

    # Save P/Q and metadata
    np.save(os.path.join(output_dir, run_id, f"P_{run_id}.npy"), mf.P)
    np.save(os.path.join(output_dir, run_id, f"Q_{run_id}.npy"), mf.Q)
    pd.Series(filtered_item_titles).to_pickle(os.path.join(output_dir, run_id, f"item_titles_{run_id}.pkl"))
    pd.Series(genre_map).to_pickle(os.path.join(output_dir, run_id, f"genre_map_{run_id}.pkl"))
    import pickle
    with open(os.path.join(output_dir, run_id, f"id_to_title_{run_id}.pkl"), "wb") as f:
        pickle.dump(id_to_title, f)

    # Top-N MF recommendations (keeps original behaviour)
    get_top_n_recommendations_MF(
        genre_map, predicted_ratings, R_filtered_train,
        filtered_user_ids, filtered_item_ids,
        top_n=top_n,
        save_path=os.path.join(output_dir, run_id, f"{run_id}_mf_train_top_{top_n}.csv")
    )

    # Build DPP models
    print("→ Building DPP models...")
    dpp_cosine, dpp_jaccard = build_dpp_models(filtered_item_titles, genre_map, all_genres, predicted_ratings)

    # Prepare item_user_rating_filtered aligned to filtered_item_ids and filtered_user_ids
    # align_matrix_to_items returns (aligned_matrix_numpy, aligned_df)
    _, item_user_rating_filtered_df = align_matrix_to_items(
        matrix_df = item_user_rating_train,
        filtered_item_ids = filtered_item_ids,
        filtered_user_ids = filtered_user_ids
    )

    # Run DPP recommendations
    get_recommendations_for_dpp(
        dpp_cosine, item_user_rating_filtered_df, filtered_item_titles,
        genre_map, predicted_ratings, top_k, top_n,
        output_dir, "cosine"
    )

    get_recommendations_for_dpp(
        dpp_jaccard, item_user_rating_filtered_df, filtered_item_titles,
        genre_map, predicted_ratings, top_k, top_n,
        output_dir, "jaccard"
    )

    print("DPP TRAIN pipeline completed successfully!")

    return (
        best_params,
        predicted_ratings,
        dpp_cosine,
        dpp_jaccard,
        filtered_item_ids,
        filtered_user_ids,
        mf
    )

# Test pipeline:



def run_dpp_pipeline_test(
        run_id, ratings_test_path, item_path, train_artifact_dir, output_dir,
        top_n=10, top_k=20
):
    os.makedirs(output_dir, exist_ok=True)

    # Load test data (dataframe)
    item_user_rating_test, genre_map_test, all_genres_test, id_to_title_test = load_and_prepare_matrix(ratings_test_path, item_path)

    # Load trained artifacts
    import pickle
    P = np.load(os.path.join(train_artifact_dir, run_id, f"P_{run_id}.npy"))
    Q = np.load(os.path.join(train_artifact_dir, run_id, f"Q_{run_id}.npy"))
    predicted_ratings_train = np.load(os.path.join(train_artifact_dir, run_id, f"mf_full_pred_matrix_{run_id}.npy"))
    item_titles = pd.read_pickle(os.path.join(train_artifact_dir, run_id, f"item_titles_{run_id}.pkl"))
    genre_map = pd.read_pickle(os.path.join(train_artifact_dir, run_id, f"genre_map_{run_id}.pkl"))
    with open(os.path.join(train_artifact_dir, run_id, f"id_to_title_{run_id}.pkl"), "rb") as f:
        id_to_title = pickle.load(f)

    # Align test to filtered items using your helper
    # align_matrix_to_items returns (aligned_matrix_numpy, aligned_df)
    R_filtered_test, filtered_df_test = align_matrix_to_items(
        matrix_df = item_user_rating_test,
        filtered_item_ids = item_titles.tolist(),
        filtered_user_ids = item_user_rating_test.index.tolist()  # we'll filter below
    )

    # Use get_filtered_predictions to extract predictions only for test users/items (requires trained MF and train user ids)
    # NOTE: get_filtered_predictions expects (trained_mf_model, filtered_df, train_filtered_user_ids, filtered_item_ids=None)
    # We need the train user ids saved at training time; if you saved them, pass here. If not, we will map by index (best to save train user ids).
    # For now, assume train user ids were saved as 'train_user_ids_{run_id}.pkl' (recommended). If you don't have them, skip get_filtered_predictions and infer as before.
    train_user_ids_path = os.path.join(train_artifact_dir, run_id, f"train_user_ids_{run_id}.pkl")
    if os.path.exists(train_user_ids_path):
        train_filtered_user_ids = pd.read_pickle(train_user_ids_path)
        filtered_user_ids_test, filtered_item_ids_test, predicted_ratings_test = get_filtered_predictions(
            trained_mf_model = type("MFwrap", (), {"full_prediction": lambda self: predicted_ratings_train, "item_ids": np.array(item_titles.tolist())})(),
            filtered_df = filtered_df_test,
            train_filtered_user_ids = train_filtered_user_ids,
            filtered_item_ids = item_titles.tolist()
        )
    else:
        # Fallback: build dummy MF to produce predictions for test users (cold-start user latents = 0)
        num_test_users = R_filtered_test.shape[0]
        mf_dummy = type("MFModel", (), {})()
        mf_dummy.P = np.zeros((num_test_users, Q.shape[1]))
        mf_dummy.Q = Q
        mf_dummy.b_u = np.zeros(num_test_users)
        mf_dummy.b_i = np.zeros(Q.shape[0])
        mf_dummy.mu = np.mean(predicted_ratings_train)
        predicted_ratings_test = mf_dummy.P.dot(mf_dummy.Q.T) + mf_dummy.mu
        filtered_user_ids_test = filtered_df_test.index.tolist()
        filtered_item_ids_test = filtered_df_test.columns.tolist()

    # Save full MF test prediction matrix
    os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)
    np.save(os.path.join(output_dir, run_id, f"mf_full_pred_matrix_test_{run_id}.npy"), predicted_ratings_test)

    # Save top-n MF test recommendations
    get_top_n_recommendations_MF(
        genre_map=genre_map,
        predicted_ratings=predicted_ratings_test,
        R_filtered=R_filtered_test,
        filtered_user_ids=filtered_user_ids_test,
        filtered_item_ids=filtered_item_ids_test,
        id_to_title=id_to_title,
        top_n=top_n,
        save_path=os.path.join(output_dir, run_id, f"{run_id}_mf_test_top_{top_n}.csv")
    )

    # Build DPP models using test predictions
    all_genres = sorted({g for genres in genre_map.values() for g in genres})
    dpp_cosine, dpp_jaccard = build_dpp_models(item_titles.tolist(), genre_map, all_genres, predicted_ratings_test)

    # Run DPP recommendations on test
    get_recommendations_for_dpp(
        dpp_cosine, item_user_rating_test[item_titles], item_titles.tolist(), genre_map, predicted_ratings_test,
        top_k, top_n, output_dir, "cosine"
    )

    get_recommendations_for_dpp(
        dpp_jaccard, item_user_rating_test[item_titles], item_titles.tolist(), genre_map, predicted_ratings_test,
        top_k, top_n, output_dir, "jaccard"
    )

    print("DPP TEST pipeline completed successfully!")





# PARAMETER
TOP_N = 10
CHUNK_SIZE = 100000
K = 20
ALPHA = 0.01
LAMDA_ = 0.1
N_EPOCHS = 50
TOP_K = 20
LAMBDA_PARAM = 0.7
DATASET_NAME = "books"

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_train.csv")
ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_val.csv")
ratings_test_file = os.path.join(base_dir, "../datasets/dpp_data", f"{DATASET_NAME}_ratings_{CHUNK_SIZE}_test.csv")

books_file_path = os.path.join(base_dir, "../datasets/GoodBooks", "books.csv")

output_dir = os.path.join(base_dir, f"../datasets/dpp_data/{DATASET_NAME}")
output_dir_test = os.path.join(base_dir, f"../datasets/dpp_data/{DATASET_NAME}/test")

run_book_id = generate_run_id()

train_results = run_dpp_pipeline(
    run_id=run_book_id,
    ratings_train_path=ratings_train_file,
    ratings_val_path=ratings_val_file,
    item_path=books_file_path,
    output_dir=output_dir,
    dataset=DATASET_NAME,
    datasize=CHUNK_SIZE,
    top_n=TOP_N,
    top_k=TOP_K,
    chunksize=CHUNK_SIZE,
    n_epochs=N_EPOCHS,
    similarity_types=["cosine", "jaccard"]
)

run_dpp_pipeline_test(
    run_id=run_book_id,
    ratings_test_path=ratings_test_file,
    item_path=books_file_path,
    train_artifact_dir=output_dir,   # folder where train saved artifacts
    output_dir=output_dir_test,      # folder to save test results
    top_n=TOP_N,
    top_k=TOP_K,
    similarity_types=["cosine", "jaccard"]
)

