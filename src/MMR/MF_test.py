from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF, save_mf_predictions
import os

# parameter 
top_n = 10
chunksizeMovies = 50000

k = 20
alpha = 0.01
lamda_ = 0.1
n_epochs = 50

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets", "ratings_test.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")

movie_user_rating, genre_map, all_genres = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=chunksizeMovies)

R = movie_user_rating.values

R_filtered, filtered_movie_titles = filter_empty_users_data(R,movie_user_rating.columns )

# Train the model
mf = MatrixFactorization(R_filtered, k, alpha, lamda_ , n_epochs)
mf.train()

# Full predicted rating matrix
predicted_ratings = mf.full_prediction()

# Get top-N candidates for MMR
all_recommendations = get_top_n_recommendations_MF(genre_map, predicted_ratings, R_filtered, movie_user_rating.index, filtered_movie_titles, top_n=top_n)

save_mf_predictions(all_recommendations,genre_map, output_path="src/datasets/mmr_data/mf_test_predictions.csv")

print("Finish MF test")
