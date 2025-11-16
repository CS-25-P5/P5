from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF, save_mf_predictions
from MMR import mmr, process_mmr, save_mmr_results
import os
import pandas as pd


#TRAIN

# parameter 
top_n = 10
chunksizeMovies = 50000

k = 20
alpha = 0.01
lamda_ = 0.1
n_epochs = 50
top_k = 20

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets", "ratings_train.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")

movie_user_rating, genre_map, all_genres = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=chunksizeMovies)

R = movie_user_rating.values

R_filtered, filtered_movie_titles = filter_empty_users_data(R, movie_user_rating.columns )


# Train the model
mf = MatrixFactorization(R_filtered, k, alpha, lamda_ , n_epochs)
mf.train()

# Full predicted rating matrix
predicted_ratings = mf.full_prediction()

# Get top-N candidates for MMR
all_recommendations = get_top_n_recommendations_MF(genre_map, predicted_ratings, R_filtered, movie_user_rating.index, filtered_movie_titles, top_n=top_n)

save_mf_predictions(all_recommendations,genre_map, output_path="src/datasets/mmr_data/mf_train_predictions.csv")

# print top 10 movies of MMR
lambda_param = 0.7
#movie_embeddings = mf.Q
movie_titles = movie_user_rating.columns.tolist()





#store all recommendations in a list for cosine
mmr_cos_recommendations_list = []

# loop over mutiple users
for user_idx, user_id in enumerate(movie_user_rating.index):
    user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

    mmr_indices = mmr(
        user_id= user_idx,
        predicted_ratings=predicted_ratings,  
        genre_map = genre_map,
        movie_titles = movie_titles,
        user_history = user_history,
        lambda_param =lambda_param,
        top_k=top_k,
        similarity_type="cosine", 
        all_genres=all_genres
                    )
    
    process_mmr(user_id, user_idx, mmr_indices, movie_titles, genre_map, predicted_ratings, mmr_cos_recommendations_list, top_n=top_n)

    

save_mmr_results(base_dir, mmr_cos_recommendations_list, similarity_type="cosine")

print("DONE with MMR for cosine:)")




#store all recommendations in a list for jaccard
mmr_jac_recommendations_list = []

# loop over mutiple users
for user_idx, user_id in enumerate(movie_user_rating.index):
    user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

    mmr_indices = mmr(
        user_id= user_idx,
        predicted_ratings=predicted_ratings,  
        genre_map = genre_map,
        movie_titles = movie_titles,
        user_history = user_history,
        lambda_param =lambda_param,
        top_k=top_k,
        similarity_type="jaccard", 
        all_genres=all_genres
                    )
    
    process_mmr(user_id, user_idx, mmr_indices, movie_titles, genre_map, predicted_ratings, mmr_jac_recommendations_list, top_n=top_n)


save_mmr_results(base_dir, mmr_jac_recommendations_list, similarity_type="jaccard")

print("DONE with MMR for jaccard:)")