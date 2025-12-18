import numpy as np
from src.backend.MMR.MF import load_and_prepare_matrix
from src.backend.MMR.MMR import mmr_builder_factory, run_mmr, process_save_mmr
from src.backend.MMR.helperFunctions import generate_run_id,build_mmr_input_from_nn
import os
import pandas as pd


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
        R_filtered = item_user_rating ,
        user_history = user_history_top_n,
        top_k = top_k)
    
    # Run MMR re-ranking for each user (Jaccard similarity)
    all_recs_jaccard = run_mmr(
        mmr_model = mmr_jaccard,
        R_filtered = item_user_rating ,
        user_history = user_history_top_n,
        top_k = top_k)
    
    # Process and save the MMR results for cosine similarity
    process_save_mmr(all_recs = all_recs_cosine,
                    user_ids=user_ids,
                    item_ids=candidate_items,
                    predicted_ratings=predicted_ratings_top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_cosine_top_{top_n}.csv"))

    # Process and save the MMR results for Jaccard similarity
    process_save_mmr(all_recs = all_recs_jaccard,
                    user_ids=user_ids,
                    item_ids=candidate_items,
                    predicted_ratings=predicted_ratings_top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_jaccard_top_{top_n}.csv"))
    
    print(f"Pipeline for {dataset} test finished successfully!")


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
    movies_ratings_train_file= os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_train.csv")
    movies_ratings_val_file = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_val.csv")
    movies_ratings_test_path = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_test.csv")
    movies_item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"{dataset_movie}.csv")
    movies_output_dir = os.path.join(base_dir,f"../datasets/mmr_data/{output_folder}")

    #load NN candidate list
    nn_candidate_list_path =  os.path.join(base_dir, f"../datasets/mmr_data/{output_folder}", "mf_test_10000_top_50.csv")

    #load GOODBooks data
    dataset_books = "books"
    folder_books = "GoodBooks"
    books_ratings_train_file= os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_train.csv")
    books_ratings_val_file = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_val.csv")
    books_ratings_test_path = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_test.csv")
    books_item_file_path = os.path.join(base_dir, f"../datasets/{folder_books}", f"{dataset_books}.csv")
    books_output_dir = os.path.join(base_dir,f"../datasets/mmr_data/{dataset_books}")


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
