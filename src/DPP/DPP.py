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

        scores = np.array(relevance_scores, dtype=float)
        scores = scores - np.min(scores) + self.epsilon
        q = np.sqrt(scores)

        # subset similarity matrix
        sim_sub = self.sim_matrix[np.ix_(candidate_indices, candidate_indices)]

        K = np.outer(q, q) * self.sim_matrix
        K += np.eye(len(K)) * self.epsilon
        return K

    # DPP Greedy MAP selection
    def dpp_greedy(self, K, candidate_indices, top_k):

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

    # DPP recommendations for a user
    def dpp(self, user_id, user_history, top_k=10):

        # Ensure user_history matches predicted_ratings length
        if len(user_history) > self.predicted_ratings.shape[1]:
            user_history = user_history[:self.predicted_ratings.shape[1]]

        candidate_indices = np.where(~user_history)[0]

        # relevance only for candidate items
        relevance = self.predicted_ratings[user_id, candidate_indices]

        # kernel only for candidate items
        K = self.build_kernel(relevance, candidate_indices)

        # but greedy needs indices 0..len-1 within K
        selected_local = self.dpp_greedy(K, list(range(len(candidate_indices))), top_k)

        # map back to original item indices
        selected = [candidate_indices[i] for i in selected_local]

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
def process_dpp(user_id, user_idx, dpp_indices, item_names, feature_map,
                predicted_ratings, dpp_recommendations_list, item_to_id,  top_n=10):

    for rank, idx in enumerate(dpp_indices[:top_n], start=1):
        item = item_names[idx]
        title = item_to_id.get(item, "")
        feature = ",".join(feature_map.get(item, set()))

        dpp_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            "itemId": item,
            'title': title,
            'predictedRating': predicted_ratings[user_idx, idx],
            'features': feature
        })



# Run DPP for all users

def get_recommendations_for_dpp(dpp_model, movie_user_rating, movie_titles, genre_map,
                                predicted_ratings, item_to_id, top_k, top_n, similarity_type):

    results = []

    for user_idx, user_id in enumerate(movie_user_rating.index):
        user_history = (movie_user_rating.iloc[user_idx, :] > 0).values

        dpp_indices = dpp_model.dpp(
            user_id=user_idx,
            user_history=user_history,
            top_k=top_k
        )

        process_dpp(
            user_id=user_id,
            user_idx=user_idx,
            dpp_indices=dpp_indices,
            item_names = movie_titles,
            feature_map=genre_map,
            predicted_ratings=predicted_ratings,
            dpp_recommendations_list=results,
            item_to_id=item_to_id,
            top_n=top_n
        )

    print(f"Done DPP for {similarity_type}")
    return results





def save_DPP(dpp_recommendations_list, output_dir):
    dpp_df = pd.DataFrame(dpp_recommendations_list)
    dpp_df.to_csv(output_dir, index=False)
    print(f"MF test predictions saved to: {output_dir}")
    print("DONE with DPP :)")

