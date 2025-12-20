import numpy as np
from MMR.MF import load_and_prepare_matrix, tune_mf, train_mf_with_best_params
from DPP import (
    DPP, build_dpp_models, get_recommendations_for_dpp, save_DPP
)
from MMR.helperFunctions import ( generate_run_id, align_matrix_to_user_items, align_matrix_to_user,
                                  prepare_train_val_matrices, get_filtered_predictions,
                                  log_experiment, build_mmr_input, build_mmr_input_from_nn
                                  )
import os
import pandas as pd

def build_dpp_input_from_nn(
        candidate_list_csv,
        R_filtered,
):
    df = pd.read_csv(candidate_list_csv)
    df["userId"] = df["userId"].astype(str)
    df["itemId"] = df["itemId"].astype(str)

    R_filtered["userId"] = R_filtered["userId"].astype(str)
    R_filtered["itemId"] = R_filtered["itemId"].astype(str)

    user_ids = df["userId"].unique().tolist()
    candidate_items = df["itemId"].unique().tolist()

    user_to_row = {u: i for i, u in enumerate(user_ids)}
    item_to_col = {i: j for j, i in enumerate(candidate_items)}

    predicted_ratings = np.zeros((len(user_ids), len(candidate_items)))

    for _, row in df.iterrows():
        predicted_ratings[
            user_to_row[row["userId"]],
            item_to_col[row["itemId"]],
        ] = row["predictedRating"]

    user_history = []

    if R_filtered is not None:
        R_filtered["userId"] = R_filtered["userId"].astype(str)
        R_filtered["itemId"] = R_filtered["itemId"].astype(str)

        user_history = []

        # For each user, mark items they've already interacted with
        for u in user_ids:
            # Select all items interacted with by the current user u
            seen_items = set(
                R_filtered.loc[
                    R_filtered["userId"] == u, "itemId"
                ]
            )

            # Create a boolean array marking which candidate items the user has already seen
            mask = np.array(
                [item in seen_items for item in candidate_items],
                dtype=bool,
            )
            user_history.append(mask)

        candidate_items_per_user = [candidate_items for _ in user_ids]

        return predicted_ratings, user_history, user_ids, candidate_items, candidate_items_per_user

def run_test_pipeline(
        run_id,
        ratings_path,
        item_path,
        output_dir=None,
        nn_candidates_csv = None,
        dataset=None,
        top_n=10,
        chunksize=10000,
        top_k=20,
        best_lambda_cosine = 0.7,
        best_lambda_jaccard = 0.7,
):

    print(f"Start {dataset} test pipeline ")

    # Create output directory for this run
    os.makedirs(output_dir, exist_ok=True)

    # Load user-item rating matrix and genre metadata for items
    item_user_rating, genre_map, all_genres = load_and_prepare_matrix(
        ratings_path, item_path)

    # Load basic user-item interaction history from CSV
    ratings_df = pd.read_csv(ratings_path)[["userId", "itemId"]]


    predicted_ratings_top_n, user_history_top_n, user_ids, candidate_items, candidate_items_per_user = build_dpp_input_from_nn(
        candidate_list_csv = nn_candidates_csv,
        R_filtered = ratings_df
    )

    print(f"Candidate items for DPP: {len(candidate_items)}")


    # Build DPP models
    genre_map_test = {item: genre_map[item] for item in candidate_items if item in genre_map}
    all_genres_test = sorted({g for genres in genre_map_test.values() for g in genres})


    # SANITY CHECK
    gt_items_test = item_user_rating.columns[item_user_rating.sum(axis=0) > 0]  # all items with any ratings in test
    num_gt_in_candidates = sum(item in candidate_items for item in gt_items_test)
    print(f"Candidate items for DPP: {len(candidate_items)}")
    print(f"Number of GT items included in candidates: {num_gt_in_candidates}")
    if num_gt_in_candidates == 0:
        print("Warning: No GT items in DPP candidate pool! GT metrics will be zero.")

    # Use full predicted ratings (not top-N)
    predicted_ratings_dpp = predicted_ratings_top_n

    dpp_cosine = build_dpp_models(candidate_items, genre_map_test, all_genres_test, predicted_ratings_dpp, 'cosine')
    dpp_jaccard = build_dpp_models(candidate_items, genre_map_test, all_genres_test, predicted_ratings_dpp, 'jaccard')

    # Run DPP recommendations
    cosine_reco = get_recommendations_for_dpp(
        dpp_cosine, ratings_df, candidate_items, genre_map_test, predicted_ratings_dpp,
        top_k, top_n, "cosine", candidate_items_per_user=candidate_items_per_user,   # from build_dpp_input()
        user_history_per_user=user_history_top_n
    )
    jaccard_reco = get_recommendations_for_dpp(
        dpp_jaccard, ratings_df, candidate_items, genre_map_test, predicted_ratings_dpp,
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
    TOP_N = 50
    CHUNK_SIZE = 10000
    K = 20
    ALPHA = 0.01
    LAMDA_ = 0.1
    N_EPOCHS = 50
    TOP_K = 20
    COS_LAMBDA_PARAM = 0.35
    JAC_LAMBDA_PARAM = 0.35
    RELEVANCE_WEIGHT = 1.0
    DIVERSITY_WEIGHT = 0.0
    RANDOM_STATE = 42

    base_dir = os.path.dirname(os.path.abspath(__file__))


    #load MovieLens data
    dataset_movie = "movies"
    folder_movie = "MovieLens"
    output_folder = "NN"
    movies_ratings_train_file= os.path.join(base_dir, "../datasets/spp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_train.csv")
    movies_ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_val.csv")
    movies_ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_test.csv")
    movies_item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"{dataset_movie}.csv")
    movies_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{output_folder}")

    #load NN candidate list
    nn_candidate_list_path =  os.path.join(base_dir, f"../datasets/dpp_data/{output_folder}", "mf_test_10000_top_50.csv")

    #load GOODBooks data
    dataset_books = "books"
    folder_books = "GoodBooks"
    books_ratings_train_file= os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_train.csv")
    books_ratings_val_file = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_val.csv")
    books_ratings_test_path = os.path.join(base_dir, "../datasets/dpp_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_test.csv")
    books_item_file_path = os.path.join(base_dir, f"../datasets/{folder_books}", f"{dataset_books}.csv")
    books_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{dataset_books}")


    weight_pairs = [
        #(1.0, 0.0),
        # (0.8, 0.2),
        (0.6, 0.4),
        # (0.5, 0.5),
        # (0.4, 0.6),
        # (0.2, 0.8),
        # (0.0, 1.0),
    ]

    for REL_WEIGHT, DIV_WEIGHT in weight_pairs:
        print(f"\n=== Running pipeline with weights: "f"relevance={REL_WEIGHT}, diversity={DIV_WEIGHT} ===")

        # run pipeline for movies
        run_movie_id = generate_run_id()
        run_test_pipeline(
            run_id = run_movie_id,
            nn_candidates_csv = nn_candidate_list_path,
            ratings_path=movies_ratings_test_path,
            item_path=movies_item_file_path,
            output_dir=movies_output_dir,
            dataset=dataset_movie,
            top_n=TOP_N,
            top_k=TOP_K,
            chunksize=CHUNK_SIZE,
            best_lambda_cosine = COS_LAMBDA_PARAM,
            best_lambda_jaccard = JAC_LAMBDA_PARAM,
        )


        # #RUN pipeline for books
        # run_test_pipeline(
        #     run_id = run_book_id,
        #     ratings_path=books_ratings_test_path,
        #     item_path=books_item_file_path,
        #     output_dir=books_output_dir,
        #     dataset=dataset_books,
        #     top_n=TOP_N,
        #     top_k=TOP_K,
        #     chunksize=CHUNK_SIZE,
        #     best_lambda_cosine = COS_LAMBDA_PARAM,
        #     best_lambda_jaccard = JAC_LAMBDA_PARAM,
        # )