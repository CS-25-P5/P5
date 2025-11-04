from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF, save_mf_predictions
from MMR import mmr
import os
import pandas as pd


#TRAIN


# parameter 
top_n = 10
chunksizeMovies = 50000

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets", "ratings_train.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")

movie_user_rating = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=chunksizeMovies)

R = movie_user_rating.values

R_filtered, filtered_user_ids, filtered_movie_titles = filter_empty_users_data(R, movie_user_rating.index, movie_user_rating.columns )


# Train the model
mf = MatrixFactorization(R_filtered, k = 20, alpha=0.01, lamda_=0.1, n_epochs=50)
mf.train()

# Full predicted rating matrix
predicted_ratings = mf.full_prediction()


# Get top-N candidates for MMR
all_recommendations = get_top_n_recommendations_MF(predicted_ratings, R_filtered, filtered_user_ids, filtered_movie_titles, top_n=top_n)
save_mf_predictions(all_recommendations, output_path="src/datasets/mf_train_predictions.csv")

# print top 10 movies of MMR
lambda_param = 0.7
movie_embeddings = mf.Q
movie_titles = movie_user_rating.columns.tolist()

#store all recommendations in a list
mmr_recommendations_list = []

# loop over mutiple users
for user_idx, user_id in enumerate(movie_user_rating.index):
    user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

    mmr_indices = mmr(
        user_id= user_idx,
        predicted_ratings=predicted_ratings,
        movie_embeddings = movie_embeddings,
        user_history = user_history,
        lambda_param =lambda_param,
        top_k=top_n
                    )
    

    for rank, idx in enumerate(mmr_indices, start=1):
        mmr_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'movieTitle': movie_titles[idx],
            'predictedRating': predicted_ratings[user_idx, idx]

        })
    

#     print("--------------------------------------------------")
#     print(f"Top {top_n} diverse movie recommendations for user {user_id} (MMR) with predicted ratings:")
#     for rank, idx in enumerate(mmr_indices, start=1):
#         movie = movie_titles[idx]
#         rating = predicted_ratings[user_idx, idx]
#         print(f"{rank}. {movie} — Predicted rating: {rating:.2f}")

# print("--------------------------------------------------")


mmr_df = pd.DataFrame(mmr_recommendations_list)

#save to csv
output_file_path = os.path.join(base_dir, "../datasets/mmr_train_recommendations.csv")
mmr_df.to_csv(output_file_path, index=False)

print("DONE with MMR :)")
