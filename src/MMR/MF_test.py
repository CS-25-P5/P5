from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF, save_mf_predictions
import os



def run_mf_pipeline(
    ratings_path, item_path, 
    user_col="userId", item_col="itemId", rating_col="rating",
    category_col="genre", title_col = "title", 
    output_dir=None, top_n=10, chunksize=10000,
    k=20, alpha=0.01, lambda_=0.1, n_epochs=50
):
  
  os.makedirs(output_dir, exist_ok=True)


  # Load and prepare data
  item_user_rating, genre_map, all_genres = load_and_prepare_matrix(
    ratings_path, item_path, user_col=user_col, item_col=item_col, 
    rating_col=rating_col, title_col=title_col, category_col=category_col, nrows_items=chunksize)

  R = item_user_rating.values

  R_filtered, filtered_movie_titles = filter_empty_users_data(R,item_user_rating.columns )

  # Train the model
  mf = MatrixFactorization(R_filtered, k, alpha, lambda_ , n_epochs)
  mf.train()


  # Full predicted rating matrix
  predicted_ratings = mf.full_prediction()

  # Get top-N candidates for MMR
  save_path = os.path.join(output_dir, f"mf_test_predictions.csv")
  get_top_n_recommendations_MF(
    genre_map, predicted_ratings, R_filtered, 
    item_user_rating.index, filtered_movie_titles, 
    top_n=top_n, save_path=save_path)
  

  print("Pipeline for MF train finished successfully!")




# PARAMETERS
TOP_N = 10
CHUNK_SIZE = 1000000

K = 20
ALPHA = 0.01
LAMDA_ = 0.1
N_EPOCHS = 50
DATASET_NAME = "movie"

#load data
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets/mmr_data", f"ratings_{CHUNK_SIZE}_test.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")

output_dir = os.path.join(base_dir,"../datasets/mmr_data/movie")


# Run MF pipeline for test dataset
run_mf_pipeline(
    ratings_path=ratings_file_path,
    item_path=movies_file_path,
    output_dir=output_dir,
    user_col="userId",
    item_col="movieId", 
    rating_col="rating",
    category_col="genres", 
    title_col = "title",
    top_n=TOP_N,
    chunksize=CHUNK_SIZE,
    k=K,
    alpha=ALPHA,
    lambda_=LAMDA_,
    n_epochs=N_EPOCHS
)



# movie_user_rating, genre_map, all_genres = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=CHUNK_SIZE_MOVIES)

# R = movie_user_rating.values

# R_filtered, filtered_movie_titles = filter_empty_users_data(R,movie_user_rating.columns )

# # Train the model
# mf = MatrixFactorization(R_filtered, k, alpha, lamda_ , n_epochs)
# mf.train()

# # Full predicted rating matrix
# predicted_ratings = mf.full_prediction()

# # Get top-N candidates for MMR
# all_recommendations = get_top_n_recommendations_MF(
#   genre_map, predicted_ratings, R_filtered, 
#   movie_user_rating.index, filtered_movie_titles, 
#   top_n=top_n, save_path="src/datasets/mmr_data/mf_test_predictions.csv")


# print("Finish MF test")
