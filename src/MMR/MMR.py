from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os


def mmr(user_id, predicted_ratings, genre_map, movie_titles, user_history, lambda_param=0.7, top_k=10):
    relevance_scores = predicted_ratings[user_id, :]
    selected_indices = []
    # Only include movies the user hasn't already seen
    remaining_indices = [i for i in range(len(relevance_scores)) if not user_history[i]]

    for _ in range(top_k):
        mmr_scores = []
        for i in remaining_indices:
            if selected_indices:
                diversity = max(jaccard_similiarity(genre_map[movie_titles[i]], genre_map[movie_titles[j]])
                                for j in selected_indices)
            else:
                diversity = 0.0

            mmr_score = lambda_param * relevance_scores[i] - (1 - lambda_param) * diversity
            mmr_scores.append((i,mmr_score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return selected_indices



# Diversification post-preocessing 

def jaccard_similiarity(genres_i, genres_j):
    #jaccard similiary between genres of two items
    if not genres_i or not genres_j:
        return 0
    
    return len(genres_i & genres_j) /len(genres_i | genres_j)
   


def process_mmr(user_id, user_idx, mmr_indices, movie_titles, genre_map, predicted_ratings, mmr_recommendations_list, top_n=10):
    for rank, idx in enumerate(mmr_indices, start = 1):
        movie = movie_titles[idx]
        # hangle missing genres
        movie_genres = genre_map.get(movie, set())
        genres = ",".join(movie_genres)


        mmr_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'title': movie,
            'predictedRating': predicted_ratings[user_idx, idx],
            'genres':genres

        })
    
    # print("--------------------------------------------------")
    # print(f"Top {top_n} diverse movie recommendations for user {user_id} (MMR) with predicted ratings:")
    # for rank, idx in enumerate(mmr_indices, start=1):
    #     movie = movie_titles[idx]
    #     genres = ",".join(genre_map.get(movie, []))
    #     rating = predicted_ratings[user_idx, idx]
    #     print(f"{rank}. {movie} — Predicted rating: {rating:.2f} | genres : {genres}")

    # print("--------------------------------------------------")