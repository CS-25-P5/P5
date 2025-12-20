import numpy as np
import pandas as pd
import os

class DPP:
    def __init__(self, item_ids, feature_map, all_features, predicted_ratings,
                 similarity_type="cosine", epsilon=1e-8):
        self.item_ids = item_ids
        self.feature_map = feature_map
        self.all_features = all_features
        self.predicted_ratings = predicted_ratings
        self.similarity_type = similarity_type
        self.epsilon = epsilon

        # build a feature/genre matrix
        self.feature_matrix = self.build_feature_matrix()

        #Build similarity matrix
        if similarity_type == 'jaccard':
            self.sim_matrix = self.jaccard_matrix()
        elif similarity_type == 'cosine':
            self.sim_matrix = self.cosine_matrix()
        else:
            raise ValueError("Invalid similarity type")


     # Build binary feature/genre matrix (movies × genres)
    def build_feature_matrix(self):
        feature_index = {g: i for i, g in enumerate(self.all_features)}

        mat = np.zeros((len(self.item_ids), len(self.all_features)), dtype=float)

        for i, item_ids in enumerate(self.item_ids):
            for g in self.feature_map.get(item_ids, []):
                mat[i, feature_index[g]] = 1.0

        return mat

     # Jaccard similarity

    def jaccard_matrix(self):
         row_sums = self.feature_matrix.sum(axis=1, keepdims=True)
         intersection = self.feature_matrix @ self.feature_matrix.T
         union = row_sums + row_sums.T - intersection

         return intersection / (union + 1e-12)

    #Cosine similarity
    def cosine_matrix(self):
        norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        denom = norms @ norms.T + 1e-12

        return (self.feature_matrix @ self.feature_matrix.T) / denom

    # Build full DPP kernel K = diag(q) * S * diag(q)
    def build_kernel(self, relevance_scores, candidate_indices):

        scores = relevance_scores.astype(float)
        scores = scores - np.min(scores) + self.epsilon
        q = np.sqrt(scores)

        # subset similarity matrix
        S = self.sim_matrix[np.ix_(candidate_indices, candidate_indices)]

        assert q.shape[0] == S.shape[0], "q and S dimension mismatch"
        assert S.shape[0] == len(scores), "Kernel size mismatch — wrong indices"


        L = np.outer(q, q) * S
        L += np.eye(len(L)) * self.epsilon
        return L

    # DPP Greedy MAP selection
    def dpp_greedy(self, L, top_k):

        #Greedy MAP approximation for DPP subset selection.
        n = L.shape[0]
        cis = np.zeros((top_k, n), dtype=L.dtype)
        di2s = np.diag(L).copy()
        selected = []

        for t in range(top_k):
            j = np.argmax(di2s)
            if di2s[j] <= 1e-12:
                break

            selected.append(j)
            dj = np.sqrt(di2s[j])

            if t == 0:
                cis[t] = L[j] / dj
            else:
                proj = cis[:t, j] @ cis[:t]
                cis[t] = (L[j] - proj) / dj

            di2s -= cis[t] ** 2
            di2s[j] = -np.inf

        return selected

    # DPP recommendations for a user
    def dpp(self, user_id, user_history,candidate_indices=None, top_k=10, top_m=100):

        # Ensure user_history matches predicted_ratings length
        if len(user_history) > self.predicted_ratings.shape[1]:
            user_history = user_history[:self.predicted_ratings.shape[1]]

        # Default: all items
        if candidate_indices is None:
            candidate_indices = np.arange(len(user_history))


        # relevance only for candidate items
        relevance = self.predicted_ratings[user_id, candidate_indices]
        if len(relevance) > top_m:
            top_m_idx = np.argsort(-relevance)[:top_m]  # descending
            candidate_indices = [candidate_indices[i] for i in top_m_idx]
            relevance = self.predicted_ratings[user_id, candidate_indices]


        # kernel only for candidate items
        L = self.build_kernel(relevance, candidate_indices)


        selected_local = self.dpp_greedy(L, top_k)

        # map back to global indices
        candidate_indices = np.asarray(candidate_indices)
        selected = candidate_indices[selected_local]

        return selected



def build_dpp_models(movie_titles, genre_map, all_features, predicted_ratings, similarity_type ="cosine"):


    model = DPP(
        item_ids=movie_titles,
        feature_map=genre_map,
        all_features=all_features,
        predicted_ratings=predicted_ratings,
        similarity_type = similarity_type
    )


    return model



# Add DPP results to list
def process_dpp(user_id, user_idx, dpp_indices, item_ids, feature_map,
                predicted_ratings, dpp_recommendations_list, itemid_to_col, top_n=10):

    for rank, col_idx in enumerate(dpp_indices[:top_n], start=1):
        item_id = int(item_ids[col_idx])
        score =  predicted_ratings[user_idx, col_idx]


        dpp_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            "itemId": item_id,
            'predictedRating': score,
        })



# Run DPP for all users

def get_recommendations_for_dpp(dpp_model, movie_user_rating, item_ids, genre_map,
                                predicted_ratings,
                                top_k, top_n, similarity_type, candidate_items_per_user=None,
                                user_history_per_user=None):

    results = []
    itemid_to_col = {item_id: idx for idx, item_id in enumerate(item_ids)}

    for user_idx, user_id in enumerate(movie_user_rating.index):
        # --- Get candidate items for this user ---
        if candidate_items_per_user is not None:
            # Check bounds to avoid IndexError
            if user_idx >= len(candidate_items_per_user):
                # Fallback: select all unrated items
                user_history_mask = (movie_user_rating.iloc[user_idx, :] > 0).values
                candidate_items_user = [
                    item_id for idx, item_id in enumerate(movie_user_rating.columns)
                    if not user_history_mask[idx] and item_id in itemid_to_col
                ]
            else:
                candidate_items_user = candidate_items_per_user[user_idx]
        else:
            # No candidate list given: select all unrated items
            user_history_mask = (movie_user_rating.iloc[user_idx, :] > 0).values
            candidate_items_user = [
                item_id for idx, item_id in enumerate(movie_user_rating.columns)
                if not user_history_mask[idx] and item_id in itemid_to_col
            ]

        if user_history_per_user is not None:
            user_history = user_history_per_user[user_idx]
        else:
            user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

        # --- Map candidate items to global indices ---
        candidate_indices = [itemid_to_col[item] for item in candidate_items_user if item in itemid_to_col]

        if len(candidate_indices) == 0:
            continue

        top_m = min(200, len(candidate_indices))

        # --- Run DPP selection ---
        dpp_indices = dpp_model.dpp(
            user_id=user_idx,
            user_history=user_history,
            candidate_indices=candidate_indices,
            top_k=top_k,
            top_m=top_m
        )


        process_dpp(
            user_id=user_id,
            user_idx=user_idx,
            dpp_indices=dpp_indices,
            item_ids=item_ids,
            feature_map=genre_map,
            predicted_ratings=predicted_ratings,
            dpp_recommendations_list=results,
            itemid_to_col=itemid_to_col,
            top_n=top_n
        )

    print("DPP diagnostics:")
    print("Users with recommendations:", len(set(r['userId'] for r in results)))
    print("Unique recommended items:", len(set(r['itemId'] for r in results)))
    print("Total recommendations:", len(results))
    print(f"Done DPP for {similarity_type}")
    return results





def save_DPP(dpp_recommendations_list, output_dir):
    dpp_df = pd.DataFrame(dpp_recommendations_list)
    dpp_df.to_csv(output_dir, index=False)
    print(f"Test predictions saved to: {output_dir}")
    print("DONE with DPP :)")

