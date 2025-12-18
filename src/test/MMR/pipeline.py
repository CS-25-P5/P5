import numpy as np
from src.backend.MMR.MF import  (
    load_and_prepare_matrix, 
    get_top_n_recommendations_MF, 
    tune_mf, train_mf_with_best_params, 
    save_mf_predictions)
from src.backend.MMR.MMR import mmr_builder_factory, tune_mmr_lambda, run_mmr, process_save_mmr
from src.backend.MMR.helperFunctions import (
    generate_run_id, 
    align_matrix_to_user, 
    prepare_train_val_matrices, 
    get_filtered_predictions, 
    log_experiment, 
    log_loss_history, 
    build_mmr_input)
import os
import pandas as pd
import time
import tracemalloc



def run_train_pipeline(
    run_id,
    ratings_train_path,
    ratings_val_path,
    ground_truth_path,
    item_path,
    output_dir=None,
    dataset=None,
    top_k=20,
    chunksize=10000,
    n_epochs=50,
    relevance_weight=0.6,
    diversity_weight=0.4,
    random_state = 42
    ):

    print(f"Starting pipeline for {dataset} train")

    # Ensure output directory exist
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare training and validation user-item matrices
    # Also extract item genre metadata for later MMR use
    item_user_rating_train, genre_map, all_genres = load_and_prepare_matrix(
        ratings_train_path, item_path)

    item_user_rating_val, _, _ = load_and_prepare_matrix(
    ratings_val_path, item_path,)

    # Align train and validation matrices, filter out users/items with no interactions
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

    # Tune MF hyperparameters using validation set
    best_params = tune_mf(
        R_train = R_filtered_train,
        R_val = R_filtered_val,
        n_epochs = n_epochs)


    # Train MF with best hyperparameters
    tracemalloc.start()
    start_time_mf = time.time()

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
    
    end_time_mf = time.time()
    time_mf = end_time_mf - start_time_mf
    mem_mf = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()


    # Build MMR model for cosine using MF predictions
    builder_cosine = mmr_builder_factory(
        item_ids=filtered_item_ids,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="cosine"
    )

    # Tune lambda parameter for cosine similarity MMR
    best_lambda_cosine, best_score_cosine = tune_mmr_lambda(
        mmr_builder = builder_cosine,
        predicted_ratings = predicted_ratings,
        R_filtered=R_filtered_train,
        val_data=val_data_filtered,
        item_ids = filtered_item_ids,
        top_k = top_k,
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight
    )

    # Build MMR model for jaccard similaity using MF predictions
    builder_jaccard = mmr_builder_factory(
        item_ids=filtered_item_ids,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="jaccard"
    )
    
    # Tune lambda parameter for jaccard similarity MMR
    best_lambda_jaccard, best_score_jaccard = tune_mmr_lambda(
        mmr_builder=builder_jaccard,
        predicted_ratings=predicted_ratings,
        R_filtered=R_filtered_train,
        val_data=val_data_filtered,
        item_ids = filtered_item_ids,
        top_k=top_k,
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight
    )


    # Build Final MMR models with best lambda and run MMR
    tracemalloc.start()
    start_time_cos = time.time()

    mmr_cosine = builder_cosine(best_lambda_cosine)
    run_mmr(mmr_model = mmr_cosine,
            R_filtered = R_filtered_train ,
            top_k = top_k)
    
    end_time_cos = time.time()
    time_cos = end_time_cos - start_time_cos
    mem_cos = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()


    tracemalloc.start()
    start_time_jac = time.time()
    mmr_jaccard = builder_jaccard(best_lambda_jaccard)

    run_mmr(mmr_model = mmr_jaccard,
        R_filtered = R_filtered_train,
        top_k = top_k)
    
    end_time_jac = time.time()
    time_jac = end_time_jac - start_time_jac
    mem_jac = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()

    # Log MF experiment results
    log_experiment(
        output_dir = output_dir,
        file_name="mf_train_experiment_log.csv",
        params = {
            "Run_id": run_id,
            "Dataset_name": dataset,
            "Datasize": chunksize,
            "K": best_params["k"],
            "Alpha": best_params["alpha"],
            "Lambda": best_params["lambda_"],
            "N_epochs": n_epochs,
            "Random_state": random_state,
            "Best_epoch": best_epoch,
            "Train_rmse": best_train_rmse,
            "Val_rmse": best_val_rmse,
            "Benchmark_time": time_mf,
            "Max_Memory_MB": mem_mf
        },
    )

    # Log per-epoch MF loss history
    log_loss_history(
        output_dir = output_dir,
        filename = f"{run_id}/mf_train_{chunksize}_loss_history.csv" , 
        train_mse = train_mse_history, 
        val_mse = val_mse_history)

    # Log MMR experiment results (cosine)
    log_experiment(
        output_dir = output_dir,
        file_name="mmr_train_experiment_log.csv",
        params={ "Run_id": run_id,
                "Dataset_name": dataset,
                "Datasize": chunksize,
                "Top_k": top_k,
                "Similarity_type": "cosine",
                "Relevance_weight": relevance_weight,
                "Diveristy_weight": diversity_weight,
                "Best_lambda": best_lambda_cosine,
                "Best_score": best_score_cosine,
                "Benchmark_time": time_cos,
                "Max_Memory_MB": mem_cos},
    )

    # Log MMR experiment results (jaccard)
    log_experiment(
        output_dir = output_dir,
        file_name="mmr_train_experiment_log.csv",
        params={ "Run_id": run_id,
                "Dataset_name": dataset,
                "Datasize": chunksize,
                "Top_k": top_k,
                "Similarity_type": "jaccard",
                "Relevance_weight": relevance_weight,
                "Diveristy_weight": diversity_weight,
                "Best_lambda": best_lambda_jaccard,
                "Best_score": best_score_jaccard,
                "Benchmark_time": time_jac,
                "Max_Memory_MB": mem_jac}
    )

    print(f"Pipeline for {dataset} train finished successfully!")

    return best_lambda_cosine, best_lambda_jaccard, mf, filtered_user_ids, filtered_item_ids


def run_test_pipeline(
    run_id,
    ratings_path,
    item_path,
    output_dir=None,
    dataset=None,
    top_n=10,
    chunksize=10000,
    top_k=20,
    best_lambda_cosine = 0.7,
    best_lambda_jaccard = 0.7,
    trained_mf_model=None,
    train_filtered_user_ids=None,
    train_filtered_item_ids=None
):
    
    print(f"Start {dataset} test pipeline ")

    # Ensure the output directory exists for saving results
    os.makedirs(output_dir, exist_ok=True)

    # Load the user-item rating matrix and item genre metadata
    item_user_rating, genre_map, all_genres= load_and_prepare_matrix(
        ratings_path, item_path)
    
    # Align the user matrix to only include users seen during training
    R_filtered, filtered_df = align_matrix_to_user(
        matrix_df=item_user_rating,
        filtered_user_ids=train_filtered_user_ids
    )   

    filtered_item_ids = train_filtered_item_ids

    # Extract predicted ratings for filtered users and items from the trained MF model
    filtered_user_ids, predicted_ratings = get_filtered_predictions(
        trained_mf_model, 
        filtered_df, 
        train_filtered_user_ids, 
        # train_filtered_item_ids 
        )


    # Generate top-N recommendations for each user from MF predictions
    get_top_n_recommendations_MF(
        predicted_ratings=predicted_ratings,
        R_filtered=R_filtered,
        filtered_user_ids=filtered_user_ids,
        filtered_item_ids=filtered_item_ids,
        top_n=top_n,
        save_path=os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv"))
    
    # Use the top-N MF recommendations as the candidate list for MMR
    candidate_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv")

    predicted_ratings_top_n, user_history_top_n, candidate_items = build_mmr_input(
    candidate_list_csv = candidate_path,
    R_filtered = R_filtered,
    filtered_user_ids = filtered_user_ids,
    filtered_item_ids = filtered_item_ids
    )

    # Save the full MF predictions to CSV
    save_mf_predictions(
        trained_mf_model=trained_mf_model,
        train_user_ids=train_filtered_user_ids,
        train_item_ids=train_filtered_item_ids,
        ground_truth_path=ground_truth_path,
        output_path=os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_predictions.csv")
    )

    # Build MMR models using cosine and jaccard similarity based on candidate items
    builder_cosine = mmr_builder_factory(
        item_ids=candidate_items,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings_top_n,
        similarity_type="cosine"
    )
    mmr_cosine = builder_cosine(best_lambda_cosine)

    builder_jaccard = mmr_builder_factory(
        item_ids=candidate_items,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings_top_n,
        similarity_type="jaccard"
    )
    mmr_jaccard = builder_jaccard(best_lambda_jaccard)

    # Run MMR re-ranking for each user to balance relevance and diversity
    all_recs_cosine = run_mmr(
        mmr_model = mmr_cosine,
        R_filtered = R_filtered,
        user_history = user_history_top_n,
        top_k = top_k)
    
    all_recs_jaccard = run_mmr(
        mmr_model = mmr_jaccard,
        R_filtered = R_filtered,
        user_history = user_history_top_n,
        top_k = top_k)
    

    # Process and save the final MMR results to CSV
    process_save_mmr(all_recs = all_recs_cosine,
                    user_ids=filtered_user_ids,
                    item_ids=candidate_items,
                    predicted_ratings=predicted_ratings_top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_cosine_top_{top_n}.csv"))


    process_save_mmr(all_recs = all_recs_jaccard,
                    user_ids=filtered_user_ids,
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
    LAMBDA_PARAM = 0.7
    RANDOM_STATE = 42

    #base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

    #load MovieLens data
    dataset_movie = "movies"
    movies_ratings_train_file= os.path.join(base_dir, "data", "INPUT_TRAIN","ratings_10K_movies_train.csv")
    movies_ratings_val_file = os.path.join(base_dir, "data", "INPUT_VAL","ratings_10K_movies_val.csv")
    movies_ratings_test_path = os.path.join(base_dir, "data", "INPUT_TEST","ratings_10K_movies_test.csv")
    movies_item_file_path = os.path.join(base_dir,"data", "INPUT_datasets", "Input_movies_dataset_100k", "movies_100K.csv")
    ground_truth_path = movies_ratings_test_path
    movies_output_dir = os.path.join(base_dir,"data", "OUTPUT_datasets", "MMR", "movies_test")

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
    #(0.8, 0.2),
    # (0.6, 0.4),
    #(0.5, 0.5),
    (0.4, 0.6),
    #(0.2, 0.8),
    #(0.0, 1.0),
    ]


    for REL_WEIGHT, DIV_WEIGHT in weight_pairs:
        print(f"\n=== Running pipeline with weights: "f"relevance={REL_WEIGHT}, diversity={DIV_WEIGHT} ===")

        # run pipeline for movies
        run_movie_id = generate_run_id()
        (
            movies_best_lambda_cosine, 
            movies_best_lambda_jaccard, 
            movies_mf_trained, 
            movies_train_user_ids, 
            movies_train_item_ids
        ) = run_train_pipeline (
            run_id = run_movie_id,
            ratings_train_path = movies_ratings_train_file,
            ratings_val_path= movies_ratings_val_file,
            ground_truth_path=ground_truth_path,
            item_path = movies_item_file_path,
            output_dir = movies_output_dir,
            top_k = TOP_K,
            chunksize= CHUNK_SIZE,
            n_epochs= N_EPOCHS,
            relevance_weight=REL_WEIGHT,
            diversity_weight=DIV_WEIGHT,
            dataset=dataset_movie,
            random_state=RANDOM_STATE)

        run_test_pipeline(
            run_id = run_movie_id,
            ratings_path=movies_ratings_test_path,
            item_path=movies_item_file_path,
            output_dir=movies_output_dir,
            dataset=dataset_movie,
            top_n=TOP_N,
            top_k=TOP_K,
            chunksize=CHUNK_SIZE,
            best_lambda_cosine = movies_best_lambda_cosine,
            best_lambda_jaccard = movies_best_lambda_jaccard,
            trained_mf_model = movies_mf_trained,
            train_filtered_user_ids=movies_train_user_ids,
            train_filtered_item_ids=movies_train_item_ids
        )


    #     #RUN pipeline for books
    #     run_book_id = generate_run_id()
    #     (
    #         books_best_lambda_cosine, 
    #         books_best_lambda_jaccard, 
    #         books_mf_trained, 
    #         books_train_user_ids, 
    #         books_train_item_ids
    #         ) = run_train_pipeline (
    #         run_id = run_book_id,
    #         ratings_train_path = books_ratings_train_file,
    #         ratings_val_path= books_ratings_val_file,
    #         item_path = books_item_file_path,
    #         output_dir = books_output_dir,
    #         top_k = TOP_K,
    #         chunksize= CHUNK_SIZE,
    #         n_epochs= N_EPOCHS,
    #         relevance_weight=REL_WEIGHT,
    #         diversity_weight=DIV_WEIGHT,
    #         dataset=dataset_books,
    #         random_state=RANDOM_STATE)

    #     run_test_pipeline(
    #         run_id = run_book_id,
    #         ratings_path=books_ratings_test_path,
    #         item_path=books_item_file_path,
    #         output_dir=books_output_dir,
    #         dataset=dataset_books,
    #         top_n=TOP_N,
    #         top_k=TOP_K,
    #         chunksize=CHUNK_SIZE,
    #         best_lambda_cosine = books_best_lambda_cosine,
    #         best_lambda_jaccard = books_best_lambda_jaccard,
    #         trained_mf_model = books_mf_trained,
    #         train_filtered_user_ids=books_train_user_ids,
    #         train_filtered_item_ids=books_train_item_ids
    #     )


