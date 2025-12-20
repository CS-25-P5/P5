import numpy as np
from MMR.MF import (load_and_prepare_matrix)
from DPP import (
     build_dpp_models, get_recommendations_for_dpp, save_DPP
)
from MMR.helperFunctions import ( generate_run_id)
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
        top_k=20
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

    CHUNK_SIZE_100K = "100K"
    CHUNK_SIZE_1M = "1M"

    #load MovieLens data
    movies_100k_cos_lambda = 0.55
    movies_100k_jac_lambda = 0.35

    books_100k_cos_lambda = 0.05
    books_100k_jac_lambda = 0.05
    dataset_movie = "movies"
    folder_movie = "MovieLens"



    # test setup
    output_folder = "NN"
    movies_ratings_train_path = os.path.join(base_dir, "../datasets/dpp_data", "movies_ratings_10000_train.csv")
    movies_item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"{dataset_movie}.csv")
    movies_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/{output_folder}")
    candidate_nn_list =  os.path.join(base_dir, "../datasets/dpp_data/NN", "mf_test_10K_top_50.csv")


    movies_100k_ratings_train_path = os.path.join(base_dir, "../datasets/dpp_data", "movies_ratings_100000_train.csv")
    movies_100k_item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"movies.csv")

    movies_1M_ratings_train_path = os.path.join(base_dir, "../datasets/dpp_data", "ratings_1M_movies_train.csv")
    movies_1M_item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", "movies1M.csv")


    #load GOODBooks data
    folder_books = "GoodBooks"
    books_ratings_train_path = os.path.join(base_dir, "../datasets/dpp_data", "books_ratings_100000_train.csv")
    books_item_file_path = os.path.join(base_dir, f"../datasets/{folder_books}", "books.csv")




    # test run
    # run_movie_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_movie_id,
    #     nn_candidates_csv = candidate_nn_list ,
    #     train_ratings_path = movies_ratings_train_path,
    #     item_path=movies_100k_item_file_path,
    #     output_dir=movies_output_dir,
    #     dataset=dataset_movie,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_10K,
    #     best_lambda_cosine = movies_100k_cos_lambda,
    #     best_lambda_jaccard = movies_100k_jac_lambda,
    # )



    #MOVIES - MLP
    MLP_ml100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLP/ml100k", "MLP_1layers_embed64_lr0.001_batch64.csv")
    MLP_ml100k_1layer_nn = "ml100k_1layers_embed64_lr0.001_batch64"
    MLP_movies_output_dir = os.path.join(base_dir,"../datasets/dpp_data/movies_NN_MLP")

    run_movie_id = generate_run_id()
    run_test_pipeline(
         run_id = run_movie_id,
         nn_candidates_csv = MLP_ml100k_1layer_nn_candidate_list_path ,
         train_ratings_path=movies_100k_ratings_train_path,
         item_path=movies_100k_item_file_path,
         output_dir=MLP_movies_output_dir,
         dataset=MLP_ml100k_1layer_nn,
         top_n=TOP_N,
         top_k=TOP_K,
         chunksize=CHUNK_SIZE_100K,
         best_lambda_cosine = movies_100k_cos_lambda,
         best_lambda_jaccard = movies_100k_jac_lambda,
    )


    MLP_ml1M_1layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLP/ml1m", "MLP_1layers_embed64_lr0.001_batch64.csv")
    MLP_ml1M_1layer = "ml1M_1layers_embed64_lr0.001_batch64"


    #run_movie_id = generate_run_id()
    #run_test_pipeline(
    #    run_id = run_movie_id,
    #    nn_candidates_csv = MLP_ml1M_1layer_nn_candidate_list_path ,
    #    train_ratings_path=movies_1M_ratings_train_path,
    #    item_path=movies_1M_item_file_path,
    #    output_dir=MLP_movies_output_dir,
    #    dataset=MLP_ml1M_1layer,
    #    top_n=TOP_N,
    #    top_k=TOP_K,
    #    chunksize=CHUNK_SIZE_1M,
    #    best_lambda_cosine = movies_100k_cos_lambda,
    #    best_lambda_jaccard = movies_100k_jac_lambda,
    #)



    #MOVIES - MLP with BPR
    MLPwithBPR_ml100k_3layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/mmr_data/MLPwithBPR", "Movies100K_NNgenres_ThreeLayers_embed64_lr0001_batch64_ranked_final.csv")
    MLPwithBPR_ml100k_3layer = "ml100k_NNgenres_ThreeLayers_embed64_lr0001_batch64"
    MLPwithBPR_movies_output_dir = os.path.join(base_dir,"../datasets/mmr_data/movies_NN_MLPwithBPR")

    # run_movie_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_movie_id,
    #     nn_candidates_csv = MLPwithBPR_ml100k_3layer_nn_candidate_list_path ,
    #     train_ratings_path=movies_100k_ratings_train_path,
    #     item_path=movies_100k_item_file_path,
    #     output_dir=MLPwithBPR_movies_output_dir,
    #     dataset=MLPwithBPR_ml100k_3layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_100K,
    #     best_lambda_cosine = movies_100k_cos_lambda,
    #     best_lambda_jaccard = movies_100k_jac_lambda,
    # )


    MLPwithBPR_ml100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLPwithBPR", "Movies100K_RecommendBPRnn_OneLayer_embed64_lr00003_batch128_ranked_final.csv")
    MLPwithBPR_ml100k_1layer = "ml100k_OneLayer_embed64_lr00003_batch128"


    # run_movie_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_movie_id,
    #     nn_candidates_csv = MLPwithBPR_ml100k_1layer_nn_candidate_list_path ,
    #     train_ratings_path=movies_100k_ratings_train_path,
    #     item_path=movies_100k_item_file_path,
    #     output_dir=MLPwithBPR_movies_output_dir,
    #     dataset=MLPwithBPR_ml100k_1layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_100K,
    #     best_lambda_cosine = movies_100k_cos_lambda,
    #     best_lambda_jaccard = movies_100k_jac_lambda,
    # )


    MLPwithBPR_ml1M_1layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLPwithBPR", "Movies100K_RecommendBPRnn_OneLayer_embed64_lr00003_batch128_ranked_final.csv")
    MLPwithBPR_ml1M_1layer = "ml1M_OneLayer_embed64_lr00003_batch128"

    # run_movie_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_movie_id,
    #     nn_candidates_csv = MLPwithBPR_ml1M_1layer_nn_candidate_list_path ,
    #     train_ratings_path=movies_1M_ratings_train_path,
    #     item_path=movies_1M_item_file_path,
    #     output_dir=MLPwithBPR_movies_output_dir,
    #     dataset=MLPwithBPR_ml1M_1layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_1M,
    #     best_lambda_cosine = movies_100k_cos_lambda,
    #     best_lambda_jaccard = movies_100k_jac_lambda,
    # )



    MLPwithBPR_ml1M_3layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLPwithBPR", "Movies1M_NNgenres_ThreeLayers_embed64_lr0001_batch64_ranked_final.csv")
    MLPwithBPR_ml1M_3layer = "ml1M_NNgenres_ThreeLayers_embed64_lr0001_batch64"

    # run_movie_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_movie_id,
    #     nn_candidates_csv = MLPwithBPR_ml1M_3layer_nn_candidate_list_path ,
    #     train_ratings_path=movies_1M_ratings_train_path,
    #     item_path=movies_1M_item_file_path,
    #     output_dir=MLPwithBPR_movies_output_dir,
    #     dataset=MLPwithBPR_ml1M_3layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_1M,
    #     best_lambda_cosine = movies_100k_cos_lambda,
    #     best_lambda_jaccard = movies_100k_jac_lambda,
    # )


    # BOOKS - MLP
    MLP_gb100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLP/gb100k", "MLP_1layers_embed64_lr0.001_batch64.csv")
    MLP_gb100k_1layer = "gb100k_1layers_embed64_lr0.001_batch64"
    MLP_books_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/books_NN_MLP")

    # run_book_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_book_id,
    #     nn_candidates_csv = MLP_gb100k_1layer_nn_candidate_list_path ,
    #     train_ratings_path=books_ratings_train_path,
    #     item_path=books_item_file_path,
    #     output_dir=MLP_books_output_dir,
    #     dataset=MLP_gb100k_1layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_100K,
    #     best_lambda_cosine = books_100k_cos_lambda,
    #     best_lambda_jaccard = books_100k_jac_lambda,
    # )




    # BOOKS - MLP with BPR
    MLPwithBPR_gb100k_3layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLPwithBPR", "Books100K_NNgenres_ThreeLayers_embed64_lr0001_batch64_ranked_final.csv")
    MLPwithBPR_gb100k_3layer = "gb100k_NNgenres_ThreeLayers_embed64_lr0001_batch64"
    MLPwithBPR_books_output_dir = os.path.join(base_dir,f"../datasets/dpp_data/books_NN_MLPwithBPR")

    # run_book_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_book_id,
    #     nn_candidates_csv = MLPwithBPR_gb100k_3layer_nn_candidate_list_path ,
    #     train_ratings_path=books_ratings_train_path,
    #     item_path=books_item_file_path,
    #     output_dir=MLPwithBPR_books_output_dir,
    #     dataset=MLPwithBPR_gb100k_3layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_100K,
    #     best_lambda_cosine = books_100k_cos_lambda,
    #     best_lambda_jaccard = books_100k_jac_lambda,
    # )




    MLPwithBPR_gb100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "../datasets/dpp_data/MLPwithBPR", "Books100K_RecommendBPRnn_OneLayer_embed64_lr00003_batch128_ranked_final.csv")
    MLPwithBPR_gb100k_1layer = "gb100k_OneLayer_embed64_lr00003_batch128"

    # run_book_id = generate_run_id()
    # run_test_pipeline(
    #     run_id = run_book_id,
    #     nn_candidates_csv = MLPwithBPR_gb100k_1layer_nn_candidate_list_path ,
    #     train_ratings_path=books_ratings_train_path,
    #     item_path=books_item_file_path,
    #     output_dir=MLPwithBPR_books_output_dir,
    #     dataset=MLPwithBPR_gb100k_1layer,
    #     top_n=TOP_N,
    #     top_k=TOP_K,
    #     chunksize=CHUNK_SIZE_100K,
    #     best_lambda_cosine = books_100k_cos_lambda,
    #     best_lambda_jaccard = books_100k_jac_lambda,
    #)