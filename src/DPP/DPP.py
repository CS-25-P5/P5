import numpy as np
import pandas as pd
import os

class DPP:
    def __init__(self, titles, feature_map, all_features, predicted_ratings,
                 similarity_type="cosine", epsilon=1e-8):
        self.titles = titles
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

        mat = np.zeros((len(self.titles), len(self.all_features)), dtype=float)

        for i, title in enumerate(self.titles):
            for g in self.feature_map.get(title, []):
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
    def build_kernel(self, relevance_scores):

        scores = np.array(relevance_scores, dtype=float)
        scores = scores - np.min(scores) + self.epsilon
        q = np.sqrt(scores)

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

        relevance = self.predicted_ratings[user_id, :]
        candidate_indices = np.where(~user_history)[0].copy()

        K = self.build_kernel(relevance)
        selected = self.dpp_greedy(K, candidate_indices, top_k)

        return selected




def build_dpp_models(movie_titles, genre_map, all_features, predicted_ratings):
    dpp_cosine = DPP(
        titles=movie_titles,
        feature_map=genre_map,
        all_features=all_features,
        predicted_ratings=predicted_ratings,
        similarity_type ="cosine"
    )

    dpp_jaccard = DPP(
        titles=movie_titles,
        feature_map=genre_map,
        all_features=all_features,
        predicted_ratings=predicted_ratings,
        similarity_type="jaccard"
    )

    return dpp_cosine, dpp_jaccard



# Add DPP results to list
def process_dpp(user_id, user_idx, dpp_indices, item_names, feature_map,
                predicted_ratings, dpp_recommendations_list, item_to_id,  top_n=10):

    for rank, idx in enumerate(dpp_indices[:top_n], start=1):
        item = item_names[idx]
        feature = ",".join(feature_map.get(item, set()))

        dpp_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            "itemId": item_to_id.get(item, ""),
            'title': item,
            'predictedRating': predicted_ratings[user_idx, idx],
            'features': feature
        })



# Run DPP for all users

def get_recommendations_for_dpp(dpp_model, movie_user_rating, movie_titles, genre_map,
                                predicted_ratings, item_to_id, top_k, top_n, output_dir, similarity_type):

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
            item_titles= movie_titles,
            genre_map=genre_map,
            predicted_ratings=predicted_ratings,
            results_list=results,
            item_to_id=item_to_id,
            top_n=top_n
        )

    save_DPP(results, output_dir, similarity_type)
    print(f"Done DPP for {similarity_type}")







def save_DPP(dpp_recommendations_list, base_dir, similarity_type):
    dpp_df = pd.DataFrame(dpp_recommendations_list)
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"dpp_{similarity_type}_recommendations.csv")
    dpp_df.to_csv(output_file_path, index=False)
    print("DONE with DPP :)")