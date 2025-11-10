from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os


def mmr(user_id, predicted_ratings, genre_map, movie_titles, user_history, lambda_param=0.7, top_k=10, similarity_type="jaccard", all_genres=None):
    relevance_scores = predicted_ratings[user_id, :]
    selected_indices = []
    # Only include movies the user hasn't already seen
    remaining_indices = [i for i in range(len(relevance_scores)) if not user_history[i]]

    for _ in range(top_k):
        mmr_scores = []
        for i in remaining_indices:
            if selected_indices:
                if similarity_type == "jaccard":
                    diversity = max(jaccard_similiarity(genre_map[movie_titles[i]], genre_map[movie_titles[j]])
                                for j in selected_indices)
                elif similarity_type == "cosine":
                    diversity = max(cosine_similarity(
                        genre_map[movie_titles[i]], 
                        genre_map[movie_titles[j]], 
                        all_genres
                        )
                        for j in selected_indices)
                else:
                    raise ValueError("Invalid similairty_type")
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
   

def cosine_similarity(genres_i, genres_j, all_genres):
    # convert genres to binary vectors
    vec_i = np.array([1 if g in genres_i else 0 for g in all_genres])
    vec_j = np.array([1 if g in genres_j else 0 for g in all_genres])

    # Handle case where both are zero vectors
    if not np.any(vec_i) or not np.any(vec_j):
        return 0.0
    
    # compute cosine similarity
    return np.dot(vec_i, vec_j)/(np.linalg.norm(vec_i) * np.linalg.norm(vec_j) )



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


def save_mmr_results(base_dir, mmr_recommendations_list, similarity_type="jaccard"):
    # create output dataframe
    mmr_df = pd.DataFrame(mmr_recommendations_list)

    #save to csv
    output_file_path = os.path.join(base_dir, f"../datasets/mmr_data/mmr_train_{similarity_type}_recommendations.csv")

    mmr_df.to_csv(output_file_path, index=False)

    
    print(f"MMR results saved: {output_file_path}")


    