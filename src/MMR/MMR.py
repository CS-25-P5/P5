from sklearn.metrics.pairwise import cosine_similarity
import MF

def mmr(user_id, predicted_ratings, movie_embeddings, user_history, lambda_param=0.7, top_k=10):
    relevance_scores = predicted_ratings[user_id, :]
    similarity_matrix = cosine_similarity(movie_embeddings)
    selected_indices = []
    # Only include movies the user hasn't already seen
    remaining_indices = [i for i in range(len(relevance_scores)) if not user_history[i]]

    for _ in range(top_k):
        mmr_scores = []
        for i in remaining_indices:
            if selected_indices:
                diversity = max(similarity_matrix[i][j] for j in selected_indices)
            else:
                diversity = 0.0

            mmr_score = lambda_param * relevance_scores[i] - (1 - lambda_param) * diversity
            mmr_scores.append((i,mmr_score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return selected_indices

# print top 10 movies of MMR
user_id = 0
lambda_param = 0.7
top_k = 10
predicted_ratings = mf.full_prediction()
user_history = (movie_user_rating.iloc[user_id, :] > 0).values
movie_embeddings = mf.Q


mmr_indices = mmr(user_id= user_id,
                  predicted_ratings=predicted_ratings,
                  movie_embeddings = movie_embeddings,
                  user_history = user_history,
                  lambda_param =lambda_param,
                  top_k=top_k
                  )


# Map indices back to titles and get predicted ratings
movie_titles = movie_user_rating.columns.tolist()
recommended_movies = [movie_titles[i] for i in mmr_indices]

print("Top diverse movie recommendations (MMR) with predicted ratings:")
for rank, idx in enumerate(mmr_indices, start=1):
    movie = movie_titles[idx]
    rating = predicted_ratings[user_id, idx]
    print(f"{rank}. {movie} — Predicted rating: {rating:.2f}")