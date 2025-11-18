import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from MMR.MF import MatrixFactorization, load_and_prepare_matrix, filter_empty_users_data, get_top_n_recommendations_MF


def build_dpp_kernel(relevance_scores, genre_map, movie_titles, all_genres, similarity_type="cosine", epsilon=1e-8):

    # ensure numerical stability
    scores = np.array(relevance_scores, dtype=float)
    scores = scores - np.min(scores) + epsilon
    q = np.sqrt(scores)

    # build feature matrix (genre binary vectors - d_i)
    X = []
    for title in movie_titles:
        genres = genre_map.get(title, set())
        X.append([1 if g in genres else 0 for g in all_genres])
    X = np.array(X, dtype=float)

    # compute similarity matrix (Sim(d_i,d_j))
    if similarity_type == "cosine":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + epsilon)
        S = np.dot(X, X.T)
    elif similarity_type == "jaccard":
        intersection = np.dot(X, X.T)
        union = np.expand_dims(X.sum(axis=1), 1) + np.expand_dims(X.sum(axis=1), 0) - intersection
        S = intersection / (union + epsilon)
    else:
        raise ValueError("Invalid similarity_type. Use 'cosine' or 'jaccard'.")


    # build kernel
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


def dpp_recommendations(user_id, predicted_ratings, movie_titles, genre_map, user_history, all_genres, top_k=10, similarity_type="cosine"):

    #Generate DPP-based diverse recommendations for a user.
    # get relevance scores for user
    relevance = predicted_ratings[user_id, :]

    #only include items the user has NOT seen
    candidate_indices = [i for i in range(len(relevance)) if not user_history[i]]

    #build DPP kernel using all items (relevance + embeddings)
    K = build_dpp_kernel(
        relevance_scores = relevance,
        genre_map = genre_map,
        movie_titles = movie_titles,
        all_genres = all_genres,
        similarity_type = similarity_type
    )

    # Select top_k diverse items
    selected_indices = dpp_greedy_selection(K, candidate_indices, top_k)

    return selected_indices