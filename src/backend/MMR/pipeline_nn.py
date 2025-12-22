import numpy as np
from src.backend.MMR.MF import load_and_prepare_matrix
from src.backend.MMR.MMR import mmr_builder_factory, run_mmr, process_save_mmr
from src.backend.MMR.helperFunctions import generate_run_id,build_mmr_input_from_nn
import os
import pandas as pd


def run_test_pipeline(
    run_id,
    train_ratings_path,
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
        train_ratings_path, item_path)
    
    # Load basic user-item interaction history from CSV
    ratings_df = pd.read_csv(train_ratings_path)[["userId", "itemId"]]

    # Build MMR input from neural network candidate recommendations
    predicted_ratings_top_n, user_history_top_n, user_ids, candidate_items = build_mmr_input_from_nn(
    candidate_list_csv = nn_candidates_csv,
    interactions_df=ratings_df)         

    # Build MMR model using cosine similarity
    builder_cosine = mmr_builder_factory(
        item_ids=candidate_items,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings_top_n,
        similarity_type="cosine"
    )
    mmr_cosine = builder_cosine(best_lambda_cosine)

    # Build MMR model using Jaccard similarity
    builder_jaccard = mmr_builder_factory(
        item_ids=candidate_items,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings_top_n,
        similarity_type="jaccard"
    )
    mmr_jaccard = builder_jaccard(best_lambda_jaccard)

    # Run MMR re-ranking for each user (cosine similarity)
    all_recs_cosine = run_mmr(
        mmr_model = mmr_cosine,
        user_ids = user_ids,
        R_filtered = item_user_rating,
        user_history = user_history_top_n,
        top_k = top_k)
    
    # Run MMR re-ranking for each user (Jaccard similarity)
    all_recs_jaccard = run_mmr(
        mmr_model = mmr_jaccard,
        user_ids = user_ids,
        R_filtered = item_user_rating ,
        user_history = user_history_top_n,
        top_k = top_k)
    
    # Process and save the MMR results for cosine similarity
    process_save_mmr(all_recs = all_recs_cosine,
                    user_ids=user_ids,
                    item_ids=candidate_items,
                    predicted_ratings=predicted_ratings_top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}_{dataset}/mmr_test_{chunksize}_cosine_top_{top_n}.csv"))

    # Process and save the MMR results for Jaccard similarity
    process_save_mmr(all_recs = all_recs_jaccard,
                    user_ids=user_ids,
                    item_ids=candidate_items,
                    predicted_ratings=predicted_ratings_top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}_{dataset}/mmr_test_{chunksize}_jaccard_top_{top_n}.csv"))
    
    print(f"Pipeline for {dataset} test finished successfully!")

if __name__ == "__main__":
    # PARAMETER
    TOP_N = 50
    K = 20
    ALPHA = 0.01
    LAMDA_ = 0.1
    N_EPOCHS = 50
    TOP_K = 20
    RELEVANCE_WEIGHT = 0.6
    DIVERSITY_WEIGHT = 0.4
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

    movies_100k_ratings_train_path = os.path.join(base_dir, "data", "INPUT_TRAIN","ratings_100K_movies_train.csv")
    movies_100k_item_file_path =  os.path.join(base_dir,"data", "INPUT_datasets", "Input_movies_dataset_100k", "movies_100K.csv")

    movies_1M_ratings_train_path = os.path.join(base_dir, "data", "INPUT_TRAIN", "ratings_1M_movies_train.csv")
    movies_1M_item_file_path = os.path.join(base_dir,"data", "INPUT_datasets", "Input_movies_dataset_1M", "movies1M.csv")

    #load GOODBooks data
    folder_books = "GoodBooks"
    books_ratings_train_path =  os.path.join(base_dir, "data", "INPUT_TRAIN","ratings_100K_goodbooks_train.csv")
    books_item_file_path = os.path.join(base_dir, "data", "INPUT_datasets","Input_goodbooks_dataset_100k", "books_100K.csv")


    #MOVIES - MLP 
    MLP_ml100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "data", "OUTPUT_datasets", "NN_Best_Models","MLP","ml100k", "MLP_1layers_embed64_lr0.001_batch64.csv")
    MLP_ml100k_1layer_nn = "ml100k_1layers_embed64_lr0.001_batch64"
    MLP_movies_output_dir = os.path.join(base_dir, "data", "OUTPUT_datasets", "MMR", "movies_NN_MLP")

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


    MLP_ml1M_1layer_nn_candidate_list_path =  os.path.join(base_dir, "data", "OUTPUT_datasets", "NN_Best_Models", "MLP", "ml1m", "MLP_1layers_embed64_lr0.001_batch64.csv")
    MLP_ml1M_1layer = "ml1M_1layers_embed64_lr0.001_batch64"

    run_movie_id = generate_run_id()
    run_test_pipeline(
        run_id = run_movie_id,
        nn_candidates_csv = MLP_ml1M_1layer_nn_candidate_list_path ,
        train_ratings_path=movies_1M_ratings_train_path,
        item_path=movies_1M_item_file_path,
        output_dir=MLP_movies_output_dir,
        dataset=MLP_ml1M_1layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_1M,
        best_lambda_cosine = movies_100k_cos_lambda,
        best_lambda_jaccard = movies_100k_jac_lambda,
    )



    #MOVIES - MLP with BPR
    MLPwithBPR_ml100k_3layer_nn_candidate_list_path =  os.path.join(base_dir,"data", "OUTPUT_datasets", "NN_Best_Models","MLPwithBPR", "Movies100K_NNgenres_ThreeLayers_embed64_lr0001_batch64_ranked_final.csv")
    MLPwithBPR_ml100k_3layer = "ml100k_NNgenres_ThreeLayers_embed64_lr0001_batch64"
    MLPwithBPR_movies_output_dir = os.path.join(base_dir,"OUTPUT_datasets", "MMR","movies_NN_MLPwithBPR")

    run_movie_id = generate_run_id()
    run_test_pipeline(
        run_id = run_movie_id,
        nn_candidates_csv = MLPwithBPR_ml100k_3layer_nn_candidate_list_path ,
        train_ratings_path=movies_100k_ratings_train_path,
        item_path=movies_100k_item_file_path,
        output_dir=MLPwithBPR_movies_output_dir,
        dataset=MLPwithBPR_ml100k_3layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_100K,
        best_lambda_cosine = movies_100k_cos_lambda,
        best_lambda_jaccard = movies_100k_jac_lambda,
    )


    MLPwithBPR_ml100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "OUTPUT_datasets", "NN_Best_Models", "MLPwithBPR", "Movies100K_RecommendBPRnn_OneLayer_embed64_lr00003_batch128_ranked_final.csv")
    MLPwithBPR_ml100k_1layer = "ml100k_OneLayer_embed64_lr00003_batch128"


    run_movie_id = generate_run_id()
    run_test_pipeline(
        run_id = run_movie_id,
        nn_candidates_csv = MLPwithBPR_ml100k_1layer_nn_candidate_list_path ,
        train_ratings_path=movies_100k_ratings_train_path,
        item_path=movies_100k_item_file_path,
        output_dir=MLPwithBPR_movies_output_dir,
        dataset=MLPwithBPR_ml100k_1layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_100K,
        best_lambda_cosine = movies_100k_cos_lambda,
        best_lambda_jaccard = movies_100k_jac_lambda,
    )


    MLPwithBPR_ml1M_1layer_nn_candidate_list_path =  os.path.join(base_dir, "OUTPUT_datasets", "NN_Best_Models", "MLPwithBPR", "Movies100K_RecommendBPRnn_OneLayer_embed64_lr00003_batch128_ranked_final.csv")
    MLPwithBPR_ml1M_1layer = "ml1M_OneLayer_embed64_lr00003_batch128"

    run_movie_id = generate_run_id()
    run_test_pipeline(
        run_id = run_movie_id,
        nn_candidates_csv = MLPwithBPR_ml1M_1layer_nn_candidate_list_path ,
        train_ratings_path=movies_1M_ratings_train_path,
        item_path=movies_1M_item_file_path,
        output_dir=MLPwithBPR_movies_output_dir,
        dataset=MLPwithBPR_ml1M_1layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_1M,
        best_lambda_cosine = movies_100k_cos_lambda,
        best_lambda_jaccard = movies_100k_jac_lambda,
    )


    
    MLPwithBPR_ml1M_3layer_nn_candidate_list_path =  os.path.join(base_dir, "OUTPUT_datasets", "NN_Best_Models", "MLPwithBPR", "Movies1M_NNgenres_ThreeLayers_embed64_lr0001_batch64_ranked_final.csv")
    MLPwithBPR_ml1M_3layer = "ml1M_NNgenres_ThreeLayers_embed64_lr0001_batch64"

    run_movie_id = generate_run_id()
    run_test_pipeline(
        run_id = run_movie_id,
        nn_candidates_csv = MLPwithBPR_ml1M_3layer_nn_candidate_list_path ,
        train_ratings_path=movies_1M_ratings_train_path,
        item_path=movies_1M_item_file_path,
        output_dir=MLPwithBPR_movies_output_dir,
        dataset=MLPwithBPR_ml1M_3layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_1M,
        best_lambda_cosine = movies_100k_cos_lambda,
        best_lambda_jaccard = movies_100k_jac_lambda,
    )


    # BOOKS - MLP
    MLP_gb100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "OUTPUT_datasets", "NN_Best_Models","MLP", "gb100k", "MLP_1layers_embed64_lr0.001_batch64.csv")
    MLP_gb100k_1layer = "gb100k_1layers_embed64_lr0.001_batch64"
    MLP_books_output_dir = os.path.join(base_dir,"OUTPUT_datasets", "MMR","books_NN_MLP")

    run_book_id = generate_run_id()
    run_test_pipeline(
        run_id = run_book_id,
        nn_candidates_csv = MLP_gb100k_1layer_nn_candidate_list_path ,
        train_ratings_path=books_ratings_train_path,
        item_path=books_item_file_path,
        output_dir=MLP_books_output_dir,
        dataset=MLP_gb100k_1layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_100K,
        best_lambda_cosine = books_100k_cos_lambda,
        best_lambda_jaccard = books_100k_jac_lambda,
    )



    
    # BOOKS - MLP with BPR
    MLPwithBPR_gb100k_3layer_nn_candidate_list_path =  os.path.join(base_dir,"OUTPUT_datasets", "NN_Best_Models", "MLPwithBPR", "Books100K_NNgenres_ThreeLayers_embed64_lr0001_batch64_ranked_final.csv")
    MLPwithBPR_gb100k_3layer = "gb100k_NNgenres_ThreeLayers_embed64_lr0001_batch64"
    MLPwithBPR_books_output_dir = os.path.join(base_dir,"OUTPUT_datasets", "MMR", "books_NN_MLPwithBPR")

    run_book_id = generate_run_id()
    run_test_pipeline(
        run_id = run_book_id,
        nn_candidates_csv = MLPwithBPR_gb100k_3layer_nn_candidate_list_path ,
        train_ratings_path=books_ratings_train_path,
        item_path=books_item_file_path,
        output_dir=MLPwithBPR_books_output_dir,
        dataset=MLPwithBPR_gb100k_3layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_100K,
        best_lambda_cosine = books_100k_cos_lambda,
        best_lambda_jaccard = books_100k_jac_lambda,
    )




    MLPwithBPR_gb100k_1layer_nn_candidate_list_path =  os.path.join(base_dir, "OUTPUT_datasets", "NN_Best_Models", "MLPwithBPR", "Books100K_RecommendBPRnn_OneLayer_embed64_lr00003_batch128_ranked_final.csv")
    MLPwithBPR_gb100k_1layer = "gb100k_OneLayer_embed64_lr00003_batch128"

    run_book_id = generate_run_id()
    run_test_pipeline(
        run_id = run_book_id,
        nn_candidates_csv = MLPwithBPR_gb100k_1layer_nn_candidate_list_path ,
        train_ratings_path=books_ratings_train_path,
        item_path=books_item_file_path,
        output_dir=MLPwithBPR_books_output_dir,
        dataset=MLPwithBPR_gb100k_1layer,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE_100K,
        best_lambda_cosine = books_100k_cos_lambda,
        best_lambda_jaccard = books_100k_jac_lambda,
    )