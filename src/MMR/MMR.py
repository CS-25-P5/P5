from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF

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

