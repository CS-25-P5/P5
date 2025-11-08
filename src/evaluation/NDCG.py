import pandas as pd
import os
import numpy as np

#ndcgv@K (Normalized Discounted Cumulative Gain), evaluates the quality of ranking (takes into concideration where a recommended item is in the list), calculates an average score across all uses
def ndcg_at_k(all_recommendations, test, movie_to_idx, k=10): #needs a dictionary where all user IDs are the key and the recommendation list what is keyed to the key, the test data, a dictionary of movie  to idx, and the number of recommendations (10)
    def dcg(scores): #helper function for calculating dcg, the scores are a lst of relevance scores of items at position 1,2,3...
        return np.sum((2 ** np.array(scores) - 1) / np.log2(np.arange(2, len(scores) + 2)))

    ndcgs = []
    for user, rec_titles in all_recommendations.items():
        user_test_movies = test[test.userId == user]['movieId'].tolist()
        user_test_idx = {movie_to_idx[movieId] for movieId in user_test_movies if movieId in movie_to_idx} #identifies relevant movies rated by user
        if not user_test_idx:
            continue

        # Binary relevance: 1 if in test set
        recs = rec_titles[:k]
        rel = [1 if title in recs else 0 for title in recs]
        ideal_rel = sorted(rel, reverse=True)

        dcg_val = dcg(rel)
        idcg_val = dcg(ideal_rel)
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)

    return np.mean(ndcgs) if ndcgs else 0.0