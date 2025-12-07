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
    filter_empty_users_data,
    get_top_n_recommendations_MF,
tune_mf, train_mf_with_best_params,  align_train_val_matrices, align_matrix_to_filtered_items
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

    # Load train/validation matrices
    item_user_rating_train, genre_map, all_genres, title_to_id = load_and_prepare_matrix(ratings_train_path, item_path, nrows_items=chunksize)
    item_user_rating_val, _, _, _ = load_and_prepare_matrix(ratings_val_path, item_path, nrows_items=chunksize)

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

    print("→ Training Matrix Factorization...")

# Train MF with best hyperparameters
    mf, predicted_ratings, train_rmse = train_mf_with_best_params(R_filtered_train, best_params, n_epochs=n_epochs)
    val_rmse = mf.compute_rmse(R_filtered_val, predicted_ratings)

    # Log experiment
    log_experiment(
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
        save_path=os.path.join(output_dir, f"{run_id}mf_train_predictions.csv")
    )

    movie_titles = filtered_item_titles
    filtered_movie_indices = range(predicted_ratings.shape[1])
    item_user_rating_filtered = item_user_rating_train.iloc[:, filtered_movie_indices]


# Build DPP models
    print("→ Building DPP models...")
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

    os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)
    np.save(os.path.join(output_dir, run_id, f"P_{run_id}.npy"), mf.P)
    np.save(os.path.join(output_dir, run_id, f"Q_{run_id}.npy"), mf.Q)
    np.save(os.path.join(output_dir, run_id, f"predicted_ratings_{run_id}.npy"), predicted_ratings)
    pd.Series(filtered_item_titles).to_pickle(os.path.join(output_dir, run_id, f"item_titles_{run_id}.pkl"))
    pd.Series(genre_map).to_pickle(os.path.join(output_dir, run_id, f"genre_map_{run_id}.pkl"))
    import pickle
    with open(os.path.join(output_dir, run_id, f"title_to_id_{run_id}.pkl"), "wb") as f:
        pickle.dump(title_to_id, f)

    print("DPP TRAIN pipeline completed successfully!")



    return (
        best_params,
        predicted_ratings,
        dpp_cosine,
        dpp_jaccard,
        filtered_item_titles
    )


# Test pipeline:


def run_dpp_pipeline_test(
        run_id, ratings_test_path, item_path, train_artifact_dir, output_dir,
        top_n=10, top_k=20
):
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    item_user_rating_test, _, _, _ = load_and_prepare_matrix(ratings_test_path, item_path)

    # Load train artifacts
    import pickle
    import numpy as np
    P = np.load(os.path.join(train_artifact_dir, run_id, f"P_{run_id}.npy"))
    Q = np.load(os.path.join(train_artifact_dir, run_id, f"Q_{run_id}.npy"))
    predicted_ratings_train = np.load(os.path.join(train_artifact_dir, run_id, f"predicted_ratings_{run_id}.npy"))
    item_titles = pd.read_pickle(os.path.join(train_artifact_dir, run_id, f"item_titles_{run_id}.pkl"))
    genre_map = pd.read_pickle(os.path.join(train_artifact_dir, run_id, f"genre_map_{run_id}.pkl"))
    with open(os.path.join(train_artifact_dir, run_id, f"title_to_id_{run_id}.pkl"), "rb") as f:
        title_to_id = pickle.load(f)

    # Align test to filtered items from train
    R_filtered_test, filtered_user_ids_test = align_matrix_to_filtered_items(item_user_rating_test, item_titles)
    num_test_users = R_filtered_test.shape[0]
    num_items = Q.shape[0]

    print(f"→ Test matrix aligned: {R_filtered_test.shape[0]} users × {R_filtered_test.shape[1]} items")

    # ---------- INFER USER LATENT FACTORS ----------
    print("→ Inferring user latent factors P_test...")

    # Build an MF container for inference
    print("→ Inferring test user latent factors...")
    mf_dummy = type("MFModel", (), {})()  # dummy object to hold P/Q
    mf_dummy.P = np.zeros((num_test_users, Q.shape[1]))   # zero latent factors for new users
    mf_dummy.Q = Q
    mf_dummy.b_u = np.zeros(num_test_users)
    mf_dummy.b_i = np.zeros(num_items)
    mf_dummy.mu = np.mean(predicted_ratings_train)

    predicted_ratings_test = mf_dummy.P.dot(mf_dummy.Q.T) + mf_dummy.mu


    # Top-N MF recommendations
    get_top_n_recommendations_MF(
        genre_map,
        predicted_ratings_test,
        R_filtered_test,
        filtered_user_ids_test,
        item_titles,
        title_to_id=title_to_id,
        top_n=top_n,
        save_path=os.path.join(output_dir, f"{run_id}/mf_test_predictions.csv")
    )



    print("→ Building DPP models...")

    all_genres = sorted({g for genres in genre_map.values() for g in genres})
    print("→ Building DPP models...")
    dpp_cosine, dpp_jaccard = build_dpp_models(
        item_titles, genre_map, all_genres, predicted_ratings_test
    )

    print("→ Running DPP recommendations...")

    get_recommendations_for_dpp(
        dpp_cosine, item_user_rating_test[item_titles], item_titles,
        genre_map, predicted_ratings_test, top_k, top_n,
        output_dir, "cosine"
    )

    get_recommendations_for_dpp(
        dpp_jaccard, item_user_rating_test[item_titles], item_titles,
        genre_map, predicted_ratings_test, top_k, top_n,
        output_dir, "jaccard"
    )

    print("✔ TEST DONE.")





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

run_dpp_pipeline(
    run_id = run_book_id,
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


run_dpp_pipeline_test(
    run_id = run_book_id,
    ratings_test_path=ratings_test_file,
    item_path=books_file_path,
    train_artifact_dir=output_dir,        # folder where train saved P.npy, Q.npy, etc.
    output_dir=output_dir_test,           # folder to save test results
    top_n=TOP_N,
    top_k=TOP_K
)

