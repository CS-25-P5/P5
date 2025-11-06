import os
import sys

# 🔧 Add this to make sure Python can find the 'MMR' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MMR.MF import (
    MatrixFactorization,
    load_and_prepare_matrix,
    filter_empty_users_data,
    get_top_n_recommendations_MF,
    save_mf_predictions,
)

from DPP import dpp_recommendations
import os
import pandas as pd
import numpy as np


# ============================================================
# Parameters
# ============================================================
top_n = 10
chunksizeMovies = 50000

# ============================================================
# Load data
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "../datasets", "ratings_train.csv")
movies_file_path = os.path.join(base_dir, "../datasets", "movies.csv")

movie_user_rating = load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=chunksizeMovies)
R = movie_user_rating.values

R_filtered, filtered_user_ids, filtered_movie_titles = filter_empty_users_data(
    R, movie_user_rating.index, movie_user_rating.columns
)

# ============================================================
# Train Matrix Factorization model
# ============================================================
mf = MatrixFactorization(R_filtered, k=20, alpha=0.01, lamda_=0.1, n_epochs=50)
mf.train()

# Full predicted rating matrix
predicted_ratings = mf.full_prediction()

# ============================================================
# Ensure output folder exists before saving CSV
# ============================================================
# Save top-N recommendations from MF
all_recommendations = get_top_n_recommendations_MF(
    predicted_ratings, R_filtered, filtered_user_ids, filtered_movie_titles, top_n=top_n
)

output_dir = os.path.join(base_dir, "../datasets")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "DPP_train_predictions.csv")
save_mf_predictions(all_recommendations, output_path= output_path)



# ============================================================
# Run DPP Re-ranking
# ============================================================
movie_embeddings = mf.Q
movie_titles = movie_user_rating.columns.tolist()

dpp_recommendations_list = []

# Loop over all users
for user_idx, user_id in enumerate(movie_user_rating.index):
    user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

    # Get DPP recommendations for this user
    dpp_indices = dpp_recommendations(
        user_id=user_idx,
        predicted_ratings=predicted_ratings,
        movie_embeddings=movie_embeddings,
        user_history=user_history,
        top_k=top_n
    )

    # Store recommendations
    for rank, idx in enumerate(dpp_indices, start=1):
        dpp_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'movieTitle': movie_titles[idx],
            'predictedRating': predicted_ratings[user_idx, idx]
        })

# ============================================================
# Save results
# ============================================================
dpp_df = pd.DataFrame(dpp_recommendations_list)
output_file_path = os.path.join(base_dir, "../datasets/dpp_train_recommendations.csv")
dpp_df.to_csv(output_file_path, index=False)

print("DONE with DPP :)")