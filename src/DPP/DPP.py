import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from MMR.MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF


def build_dpp_kernel(relevance_scores, item_embeddings, epsilon=1e-8):

    # Step 1: ensure numerical stability
    scores = np.array(relevance_scores, dtype=float)
    scores = scores - np.min(scores) + epsilon
    q = np.sqrt(scores)

    # Step 2: normalize item embeddings
    X = np.array(item_embeddings, dtype=float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + epsilon)

    # Step 3: cosine similarity matrix
    S = np.dot(X, X.T)
    S = np.clip(S, -1.0, 1.0)

    # Step 4: build kernel
    K = np.outer(q, q) * S
    K += np.eye(len(K)) * epsilon  # diagonal stability
    return K


def dpp_greedy_selection(K, candidate_indices, top_k):

    #Greedy MAP approximation for DPP subset selection.

    selected = []
    remaining = list(candidate_indices)

    for _ in range(min(top_k, len(remaining))):
        best_idx = None
        best_logdet = -np.inf

        for i in remaining:
            if not selected:
                val = K[i, i]
                sign, logdet = np.linalg.slogdet(np.array([[val]]))
            else:
                subset = selected + [i]
                subK = K[np.ix_(subset, subset)]
                sign, logdet = np.linalg.slogdet(subK)
            if sign > 0 and logdet > best_logdet:
                best_logdet = logdet
                best_idx = i

        if best_idx is None:
            best_idx = remaining[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def dpp_recommendations(user_id, predicted_ratings, movie_embeddings, user_history, top_k=10):

    #Generate DPP-based diverse recommendations for a user.
    # Step 1: get relevance scores for user
    relevance = predicted_ratings[user_id, :]

    # Step 2: only include items the user has NOT seen
    candidate_indices = [i for i in range(len(relevance)) if not user_history[i]]

    # Step 3: build DPP kernel using all items (relevance + embeddings)
    K = build_dpp_kernel(relevance, movie_embeddings)

    # Step 4: select top_k diverse items only from unseen ones
    selected_indices = dpp_greedy_selection(K, candidate_indices, top_k)

    return selected_indices