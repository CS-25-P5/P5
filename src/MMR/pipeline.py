import numpy as np
from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF, tune_mf, train_mf_with_best_params, align_train_val_matrices,align_test_matrix,align_matrix_to_filtered_items, get_aligned_predictions
from MMR import MMR, mmr_builder_factory, tune_mmr_lambda, run_mmr, process_save_mmr
import os
import pandas as pd
import time
import tracemalloc
import datetime
import csv


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


def run_train_pipeline(
    run_id,
    ratings_train_path, 
    ratings_val_path , 
    item_path, 
    output_dir=None, 
    dataset=None, 
    top_n=10, 
    top_k=20, 
    chunksize=10000, 
    n_epochs=50, 
    relevance_weight=0.6,
    diversity_weight=0.4,
    random_state = 42
    ):

    os.makedirs(output_dir, exist_ok=True)


    #load data
    item_user_rating_train, genre_map, all_genres, title_to_id = load_and_prepare_matrix( 
        ratings_train_path, item_path,nrows_items=chunksize)
    

    item_user_rating_val, genre_map, all_genres, title_to_id = load_and_prepare_matrix( 
    ratings_val_path, item_path, nrows_items=chunksize)

    # align train anf val to the same items
    train_aligned, val_aligned = align_train_val_matrices(item_user_rating_train, item_user_rating_val )

    #Convert to numpy arrays
    R_train = train_aligned.values
    R_val = val_aligned.values


    # Remove users with not interactions 
    R_filtered_train, filtered_user_ids, filtered_item_titles = filter_empty_users_data(
    R = R_train,
    user_ids=train_aligned.index,
    item_titles = train_aligned.columns
    )

    R_filtered_val, val_data_filtered = align_matrix_to_filtered_items(item_user_rating_val, filtered_item_titles, filtered_user_ids)

    
    # TRAIN MF
    # Tune MF parameters 
    tracemalloc.start()
    start_time_mf = time.time()
    best_params = tune_mf(
        R_train=R_filtered_train, 
        R_val = R_filtered_val, 
        n_epochs=n_epochs)
    end_time_mf = time.time()
    time_mf = end_time_mf - start_time_mf
    mem_mf = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()


    # Train MF with best hyperparameters
    mf, predicted_ratings, train_rmse, random_state = train_mf_with_best_params(
        R_filtered_train, 
        best_params, 
        n_epochs=n_epochs, 
        random_state = random_state)
    
    val_rmse = mf.compute_rmse(R_filtered_val, predicted_ratings)

    # Attach filtered item titles to MF model
    mf.item_titles = filtered_item_titles

    # Get top-N candidates for MF
    get_top_n_recommendations_MF(
        genre_map, 
        predicted_ratings,
        R_filtered_train, 
        filtered_user_ids, 
        filtered_item_titles, 
        top_n=top_n,  
        title_to_id = title_to_id,
        save_path = os.path.join(output_dir,f"{run_id}/mf_train_{chunksize}_predictions.csv"))


    #TUNE MMR lambda
    # Create a builder for cosine similarity
    builder_cosine = mmr_builder_factory(
        item_titles=filtered_item_titles,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="cosine"
    )

    tracemalloc.start()
    start_time_cos = time.time()
    best_lambda_cosine, best_score_cosine = tune_mmr_lambda(
        mmr_builder = builder_cosine,
        predicted_ratings = predicted_ratings,
        R_filtered=R_filtered_train,
        val_data=val_data_filtered,
        item_titles = filtered_item_titles,
        k_eval = top_k,
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight
    )
    end_time_cos = time.time()
    time_cos = end_time_cos - start_time_cos
    mem_cos = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()


    # Repeat for jaccard similarity
    builder_jaccard = mmr_builder_factory(
        item_titles=filtered_item_titles,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="jaccard"
    )

    tracemalloc.start()
    start_time_jac = time.time()
    best_lambda_jaccard, best_score_jaccard = tune_mmr_lambda(
        mmr_builder=builder_jaccard,
        predicted_ratings=predicted_ratings,
        R_filtered=R_filtered_train,
        val_data=val_data_filtered,
        item_titles=filtered_item_titles,
        k_eval=top_k,
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight
    )
    end_time_jac = time.time()
    time_jac = end_time_jac - start_time_jac
    mem_jac = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.start()


    # Build Final MMR models with best lambda
    mmr_cosine = builder_cosine(best_lambda_cosine)

    mmr_jaccard = builder_jaccard(best_lambda_jaccard)

    # Run MMR
    all_recs_cosine = run_mmr(mmr_model = mmr_cosine, 
            R_filtered = R_filtered_train , 
            top_k = top_k)
    

    all_recs_jaccard = run_mmr(mmr_model = mmr_jaccard, 
            R_filtered = R_filtered_train, 
            top_k = top_k)
    
    

    # Process and Save MMR result
    process_save_mmr(all_recs = all_recs_cosine, 
                    item_user_rating = item_user_rating_train,
                    item_titles = filtered_item_titles, 
                    predicted_ratings = predicted_ratings, 
                    genre_map = genre_map, 
                    title_to_id = title_to_id,
                    top_n = top_n, 
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_train_{chunksize}_cosine_predictions.csv"))
    

    process_save_mmr(all_recs = all_recs_jaccard, 
                    item_user_rating = item_user_rating_train,
                    item_titles = filtered_item_titles, 
                    predicted_ratings = predicted_ratings, 
                    genre_map = genre_map, 
                    title_to_id = title_to_id,
                    top_n = top_n, 
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_train_{chunksize}_jaccard_predictions.csv"))
    

    

    #LOG MF DATA
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
            "Train_rmse": train_rmse,
            "Val_rmse": val_rmse,
            "Benchmark_time": time_mf,
            "Max_Memory_MB": mem_mf
        },
    )

    # LOG MMR DATA
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
    

    print("Pipeline for train finished successfully!")

    return best_lambda_cosine, best_lambda_jaccard, mf

    

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
    trained_mf_model=None
):
    
    os.makedirs(output_dir, exist_ok=True)


    # Load and prepare data
    item_user_rating, genre_map, all_genres,title_to_id = load_and_prepare_matrix(
        ratings_path, item_path, nrows_items=chunksize)

    R_aligned_test, aligned_test_items = align_test_matrix(item_user_rating, trained_mf_model)

    R_filtered, filtered_user_ids, filtered_item_titles = filter_empty_users_data(
        R = R_aligned_test,
        item_titles = aligned_test_items,
        user_ids = item_user_rating.index )
    

    predicted_ratings = get_aligned_predictions(trained_mf_model, filtered_item_titles)



    # Get top-N candidates for MMR
    save_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_predictions.csv")
    get_top_n_recommendations_MF(
        genre_map, 
        predicted_ratings, 
        R_filtered, 
        filtered_user_ids, 
        filtered_item_titles, 
        top_n=top_n, 
        title_to_id = title_to_id,
        save_path=save_path)
    
    # Create a builder for cosine similarity
    builder_cosine = mmr_builder_factory(
        item_titles=filtered_item_titles,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="cosine"
    )

    mmr_cosine = builder_cosine(best_lambda_cosine)

    builder_jaccard = mmr_builder_factory(
        item_titles=filtered_item_titles,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="jaccard"
    )

    mmr_jaccard = builder_jaccard(best_lambda_jaccard)


    # Run MMR
    tracemalloc.start()
    start_time_cos = time.time()
    all_recs_cosine = run_mmr(
        mmr_model = mmr_cosine, 
        R_filtered = R_filtered , 
        top_k = top_k)
    end_time_cos = time.time()

    time_cos = end_time_cos - start_time_cos
    mem_cos = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()
    
    tracemalloc.start()
    start_time_jac = time.time()
    all_recs_jaccard = run_mmr(
        mmr_model = mmr_jaccard, 
        R_filtered = R_filtered , 
        top_k = top_k)
    end_time_jac = time.time()

    time_jac = end_time_jac - start_time_jac
    mem_jac = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()
    

    # Process and Save MMR result
    process_save_mmr(all_recs = all_recs_cosine, 
                    item_user_rating = item_user_rating,
                    item_titles = filtered_item_titles, 
                    predicted_ratings = predicted_ratings, 
                    genre_map = genre_map, 
                    title_to_id = title_to_id,
                    top_n = top_n, 
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_cosine_predictions.csv"))
    

    process_save_mmr(all_recs = all_recs_jaccard, 
                    item_user_rating = item_user_rating,
                    item_titles = filtered_item_titles, 
                    predicted_ratings = predicted_ratings, 
                    genre_map = genre_map, 
                    title_to_id = title_to_id,
                    top_n = top_n, 
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_jaccard_predictions.csv"))
    



    # LOG MMR DATA
    log_experiment(
        output_dir = output_dir,
        file_name="mmr_test_experiment_log.csv",
        params={ "Run_id": run_id,
                "Dataset_name": dataset,
                "Datasize": chunksize,
                "Similarity_type": "cosine",
                "Benchmark_time": time_cos,
                "Max_Memory_MB": mem_cos
                }

    )


    log_experiment(
        output_dir = output_dir,
        file_name="mmr_test_experiment_log.csv",
        params={ "Run_id": run_id,
                "Dataset_name": dataset,
                "Datasize": chunksize,
                "Similarity_type": "jaccard",
                "Benchmark_time": time_jac,
                "Max_Memory_MB": mem_jac
                }
    )



    print("Pipeline for test finished successfully!")




if __name__ == "__main__":
    # PARAMETER
    TOP_N = 10
    CHUNK_SIZE = 100000
    K = 20
    ALPHA = 0.01
    LAMDA_ = 0.1
    N_EPOCHS = 50
    TOP_K = 20
    LAMBDA_PARAM = 0.7
    RELEVANCE_WEIGHT = 0.6
    DIVERSITY_WEIGHT = 0.4
    RANDOM_STATE = 42


    #load data
    dataset_movie = "movies"
    folder_movie = "MovieLens"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_train_file= os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_movie}_ratings_{CHUNK_SIZE}_test.csv")
    item_file_path = os.path.join(base_dir, f"../datasets/{folder_movie}", f"{dataset_movie}.csv")

    output_dir = os.path.join(base_dir,f"../datasets/mmr_data/{dataset_movie}")

    run_movie_id = generate_run_id()

    best_lambda_cosine, best_lambda_jaccard, mf_trained = run_train_pipeline(
        run_id = run_movie_id,
        ratings_train_path = ratings_train_file,
        ratings_val_path= ratings_val_file,
        item_path = item_file_path,
        output_dir = output_dir,
        top_n = TOP_N,
        top_k = TOP_K,
        chunksize= CHUNK_SIZE,
        n_epochs= N_EPOCHS,
        relevance_weight=RELEVANCE_WEIGHT,
        diversity_weight=DIVERSITY_WEIGHT,
        dataset=dataset_movie,
        random_state=RANDOM_STATE)
    

    #Run MF pipeline for test dataset
    run_test_pipeline(
        run_id = run_movie_id,
        ratings_path=ratings_test_path,
        item_path=item_file_path,
        output_dir=output_dir,
        dataset=dataset_movie, 
        top_n=TOP_N,
        top_k=TOP_K, 
        chunksize=CHUNK_SIZE,
        best_lambda_cosine = best_lambda_cosine,
        best_lambda_jaccard = best_lambda_jaccard,
        trained_mf_model = mf_trained
    )





    #load data
    dataset_books = "books"
    folder_books = "GoodBooks"
    ratings_train_file= os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_train.csv")
    ratings_val_file = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_val.csv")
    ratings_test_path = os.path.join(base_dir, "../datasets/mmr_data", f"{dataset_books}_ratings_{CHUNK_SIZE}_test.csv")
    item_file_path = os.path.join(base_dir, f"../datasets/{folder_books}", f"{dataset_books}.csv")

    output_dir = os.path.join(base_dir,f"../datasets/mmr_data/{dataset_books}")


    run_book_id = generate_run_id()

    best_lambda_cosine, best_lambda_jaccard, mf_trained = run_train_pipeline(
        run_id = run_book_id,
        ratings_train_path = ratings_train_file,
        ratings_val_path= ratings_val_file,
        item_path = item_file_path,
        output_dir = output_dir,
        top_n = TOP_N,
        top_k = TOP_K,
        chunksize= CHUNK_SIZE,
        n_epochs= N_EPOCHS,
        relevance_weight=0.6,
        diversity_weight=0.4,
        dataset=dataset_books,
        random_state=RANDOM_STATE)
    

        

    # Run MF pipeline for test dataset
    run_test_pipeline(
        run_id = run_book_id,
        ratings_path=ratings_test_path,
        item_path=item_file_path,
        output_dir=output_dir,
        dataset=dataset_books,
        top_n=TOP_N,
        top_k=TOP_K,
        chunksize=CHUNK_SIZE,
        best_lambda_cosine = best_lambda_cosine,
        best_lambda_jaccard = best_lambda_jaccard,
        trained_mf_model = mf_trained
    )