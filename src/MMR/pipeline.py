import numpy as np
from MF import load_and_prepare_matrix, get_top_n_recommendations_MF, tune_mf, train_mf_with_best_params, save_mf_predictions
from MMR import mmr_builder_factory, tune_mmr_lambda, run_mmr, process_save_mmr
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

    print(f"Starting pipeline for {dataset} train")

    os.makedirs(output_dir, exist_ok=True)

    item_user_rating_train, genre_map, all_genres, id_to_title = load_and_prepare_matrix(
        ratings_train_path, item_path)

    item_user_rating_val, _, _, _  = load_and_prepare_matrix(
    ratings_val_path, item_path,)


    (
        R_filtered_train,
        R_filtered_val,
        val_data_filtered,
        filtered_user_ids,
        filtered_item_ids,
        filtered_item_titles
    )= prepare_train_val_matrices(
        item_user_rating_train, 
        item_user_rating_val,
        id_to_title=id_to_title
    )


    # TRAIN MF
    # Tune MF parameters
    tracemalloc.start()
    start_time_mf = time.time()
    best_params = tune_mf(
        R_train = R_filtered_train,
        R_val = R_filtered_val,
        n_epochs = n_epochs)
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
    mf.item_ids = filtered_item_ids

    # Get top-N candidates for MF
    # get_top_n_recommendations_MF(
    #     genre_map=genre_map,
    #     predicted_ratings=predicted_ratings,
    #     R_filtered=R_filtered_train,
    #     filtered_user_ids=filtered_user_ids,
    #     filtered_item_ids=filtered_item_ids,
    #     top_n=top_n,
    #     id_to_title = id_to_title,
    #     save_path = os.path.join(output_dir,f"{run_id}/mf_train_{chunksize}_top_{top_n}.csv"))


    #TUNE MMR lambda
    # Create a builder for cosine similarity
    builder_cosine = mmr_builder_factory(
        item_ids=filtered_item_ids,
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
        item_ids=filtered_item_ids,
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
    tracemalloc.stop()


    # Build Final MMR models with best lambda
    # mmr_cosine = builder_cosine(best_lambda_cosine)

    # mmr_jaccard = builder_jaccard(best_lambda_jaccard)

    # # Run MMR
    # all_recs_cosine = run_mmr(mmr_model = mmr_cosine,
    #         R_filtered = R_filtered_train ,
    #         top_k = top_k)


    # all_recs_jaccard = run_mmr(mmr_model = mmr_jaccard,
    #         R_filtered = R_filtered_train,
    #         top_k = top_k)



    # Process and Save MMR result
    # process_save_mmr(all_recs = all_recs_cosine,
    #                 item_user_rating = item_user_rating_train,
    #                 item_ids = filtered_item_ids,
    #                 predicted_ratings = predicted_ratings,
    #                 genre_map = genre_map,
    #                 id_to_title = id_to_title,
    #                 top_n = top_n,
    #                 output_file_path = os.path.join(output_dir,f"{run_id}/mmr_train_{chunksize}_cosine_top_{top_n}.csv"))


    # process_save_mmr(all_recs = all_recs_jaccard,
    #                 item_user_rating = item_user_rating_train,
    #                 item_ids = filtered_item_ids,
    #                 predicted_ratings = predicted_ratings,
    #                 genre_map = genre_map,
    #                 id_to_title = id_to_title,
    #                 top_n = top_n,
    #                 output_file_path = os.path.join(output_dir,f"{run_id}/mmr_train_{chunksize}_jaccard_top_{top_n}.csv"))


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
    # Create output directory for this run
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    item_user_rating, genre_map, all_genres, id_to_title = load_and_prepare_matrix(
        ratings_path, item_path)
    
    # Use your existing function to align the matrix!
    R_filtered, filtered_df = align_matrix_to_items(
        matrix_df=item_user_rating,
        filtered_item_ids=train_filtered_item_ids,
        filtered_user_ids=train_filtered_user_ids
    )
    

    filtered_user_ids, filtered_item_ids, predicted_ratings = get_filtered_predictions(
        trained_mf_model, filtered_df, train_filtered_user_ids, train_filtered_item_ids)


    # Get top-N candidates for MMR
    mf_top_n_path = os.path.join(output_dir, f"{run_id}/mf_test_{chunksize}_top_{top_n}.csv")


    get_top_n_recommendations_MF(
        genre_map=genre_map,
        predicted_ratings=predicted_ratings,
        R_filtered=R_filtered,
        filtered_user_ids=filtered_user_ids,
        filtered_item_ids=filtered_item_ids,
        id_to_title=id_to_title,
        top_n=top_n,
        save_path=mf_top_n_path)
    

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

    # Create a builder for cosine similarity
    builder_cosine = mmr_builder_factory(
        item_ids=filtered_item_ids,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="cosine"
    )

    mmr_cosine = builder_cosine(best_lambda_cosine)

    builder_jaccard = mmr_builder_factory(
        item_ids=filtered_item_ids,
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
                    item_user_rating=item_user_rating,
                    item_ids=filtered_item_ids,
                    predicted_ratings=predicted_ratings,
                    genre_map=genre_map,
                    id_to_title=id_to_title,
                    top_n=top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_cosine_top_{top_n}.csv"))


    process_save_mmr(all_recs = all_recs_jaccard,
                    item_user_rating=item_user_rating,
                    item_ids=filtered_item_ids,
                    predicted_ratings=predicted_ratings,
                    genre_map=genre_map,
                    id_to_title=id_to_title,
                    top_n=top_n,
                    output_file_path = os.path.join(output_dir,f"{run_id}/mmr_test_{chunksize}_jaccard_top_{top_n}.csv"))
    

    print(f"Log MMR data")

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

    print(f"Pipeline for {dataset} test finished successfully!")


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

    best_lambda_cosine, best_lambda_jaccard, mf_trained, train_user_ids, train_item_ids = run_train_pipeline(
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
        trained_mf_model = mf_trained,
        train_filtered_user_ids=train_user_ids,
        train_filtered_item_ids=train_item_ids
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

    best_lambda_cosine, best_lambda_jaccard, mf_trained, train_user_ids, train_item_ids = run_train_pipeline(
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


    #Run MF pipeline for test dataset
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
        trained_mf_model = mf_trained,
        train_filtered_user_ids=train_user_ids,
        train_filtered_item_ids=train_item_ids
    )