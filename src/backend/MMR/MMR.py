import pandas as pd
import os
import numpy as np
from rectools.metrics import (
    NDCG, IntraListDiversity
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator



class MMR:
    def __init__(self, item_ids, genre_map, all_genres, predicted_ratings, similarity_type='jaccard', lambda_param=0.7):
        # List of all item IDS
        self.item_ids = item_ids
        # mapping from itemid to set of genres
        self.genre_map = genre_map
        # list of all unique genres in the datset
        self.all_genres = all_genres  
        # predicted rating from MF model         
        self.predicted_ratings = predicted_ratings
        self.similarity_type = similarity_type
        # tradeoff parameter between revance and diversity
        self.lambda_param = lambda_param

        # Build binary genre vectors(rows: items, columns: genres)
        self.genre_vectors = self.build_genre_vectors()

        # Compute similarity matrix between items based on genres
        if similarity_type == 'jaccard':
            self.sim_matrix = self.jaccard_matrix()
        elif similarity_type == 'cosine':
            self.sim_matrix = self.consine_matrix()
        else:
            raise ValueError("Invalid similarity type")


    #  Build binary item-genre matrix:
    def build_genre_vectors(self):
        # Map each genre to a unique column index
        genre_index = {}
        for idx, genre in enumerate(self.all_genres):
            genre_index[genre] = idx

        # create a 2-dimensional matrix (numpy array) filled with zeros
        mat = np.zeros((len(self.item_ids), len(self.all_genres)), dtype=np.float32)

        # fill matrix so it indicates which genres each movie belong to
        for i, item_id in enumerate(self.item_ids):
            # fallback to empty set if item_id missing in genre_map
            genres = self.genre_map.get(item_id, set())  
            for g in genres:
                mat[i, genre_index[g]] = 1.0

        return mat
    
    # Jaccard similarity   
    def jaccard_matrix(self):
        # Count how many genres each item has
        row_sums = self.genre_vectors.sum(axis=1, keepdims=True)

        # counts how many genres two items share
        intersection = self.genre_vectors @ self.genre_vectors.T

        # get the union
        union = row_sums + row_sums.T - intersection
        # 1e-12 prevents division by zero
        return intersection/(union + 1e-12)
    
    # Cosine similarity
    def consine_matrix(self):
        # calculate the norm
        norms = np.linalg.norm(self.genre_vectors, axis=1, keepdims=True)
        # calculate the denominator 
        denom = norms * norms.T + 1e-12
        return (self.genre_vectors @ self.genre_vectors.T) / denom
    

    # genreate top-k recomemndtions   
    def mmr(self, user_id, user_history, top_k=10):
        # predicted score for each item
        relevance = self.predicted_ratings[user_id]

        # Convert user history to boolean array
        user_history = np.array(user_history, dtype=bool)

        # Only consider items that are unseen and have positive predicted rating
        remaining = np.where((~user_history) & (relevance > 0))[0].copy()
        selected = []

        # Normalize relevance to [0, 1] for stable scoring
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)


        #Repeat up to top_k times or until no remaining items left.
        for _ in range(min(top_k, len(remaining))):
            if len(selected) == 0:
                # First item is purely based on relevance
                mmr_scores = self.lambda_param * relevance[remaining]
            else:
                # Compute max similarity to already selected items (diversity)
                diversity = self.sim_matrix[np.ix_(remaining, selected)].max(axis=1)

                # MMR score balances relevance and diversity
                mmr_scores = self.lambda_param * relevance[remaining] - (1- self.lambda_param) * diversity

            # Select the item with the highest MMR score (best tradeoff between relevance and diversity)
            best_idx = np.argmax(mmr_scores)

            #  Map from remaining list index to dataset index
            best_item = remaining[best_idx]

            #add item to selected
            selected.append(best_item)

            # remove item just picked
            remaining = np.delete(remaining, best_idx)

        return selected


def run_mmr(mmr_model, user_ids, R_filtered, top_k, user_history = None):
    all_recs = []
    # Loop over all user indices corresponding to user_ids
    for idx, user_id in enumerate(user_ids):
        # Get user's interacted items (from user_history or R_filtered)
        user_histories = np.atleast_1d(
            user_history[idx] if user_history is not None
            else (R_filtered[idx, :] > 0)
        )
        # Run the MMR algorithm for this user to get top_k recommendations
        rec_indices = mmr_model.mmr(idx, user_histories, top_k)
        all_recs.append(rec_indices)
    
    return all_recs



# ==========================
# Create an MMR model with fixed data and the provided lambda_param
# ==========================
def mmr_builder_factory(
        item_ids, 
        genre_map, 
        all_genres, 
        predicted_ratings, 
        similarity_type="cosine"):

    def builder(lambda_param):
        return MMR(
            item_ids = item_ids,
            genre_map=genre_map,
            all_genres=all_genres,
            predicted_ratings=predicted_ratings,
            similarity_type=similarity_type,
            lambda_param=lambda_param
            )
    return builder


# ==========================
# FIne tune hymperparameter lambda with grid-search
# ==========================
def tune_mmr_lambda(
        mmr_builder,        # function that builds an MMR model given lambda
        predicted_ratings,  # full predicted rating matrix from MF model
        R_filtered,         # user-item rating matrix 
        val_data,           # validation set of user-item interactions 
        item_ids,           # list of all item IDs
        top_k=10,
        relevance_weight = 0.6,
        diversity_weight = 0.4
        ):
    
    # Define candidate lambda values to test
    lambda_grid = np.linspace(0, 1, 21)
    best_lambda = None
    best_score = -np.inf

    # builds mmr model to extract genre vectors
    sample_mmr = mmr_builder(lambda_grid[0])
    genre_matrix = sample_mmr.genre_vectors


    # Build genre DataFrame for computing diversity (ILD)
    genre_df = pd.DataFrame(
        genre_matrix,
        index = item_ids,
        columns=sample_mmr.all_genres
    )

    # replace NaN and ensure 0/1 values
    genre_df = genre_df.fillna(0).astype(int)

    # Create Hamming distance calculator for ILD metric
    distance_calc = PairwiseHammingDistanceCalculator(genre_df)

    # Define evaluation metrics
    ndcg_metric = NDCG(k=top_k)
    ild_metric = IntraListDiversity(k=top_k, distance_calculator=distance_calc)

    # Prepare validation interactions in Rectools format
    val_array = val_data.values
    rows, cols = np.where(val_array > 0)


    # Map column indices to actual item IDs
    mapped_item_ids = []
    for c in cols:
        if c < len(item_ids):  # Safety check
            mapped_item_ids.append(item_ids[c])
        else:
            print(f"Warning: Column index {c} out of bounds for item_ids")
            mapped_item_ids.append(None)

    # Create DataFrame of user-item interactions from validation matrix
    interactions_df = pd.DataFrame({
        "user_id": rows,
        "item_id": mapped_item_ids,  
        "rating": val_array[rows, cols]
    })

    # Store raw NDGC and ILD for normalization
    ndcg_vals = []
    ild_vals = []

    # Loop over cnadidata lambda values
    for lam in lambda_grid:
        #Build new MMR model for each lambda
        mmr_model = mmr_builder(lam)

        # genreate top-k recommendation for each user
        user_recs = {}
        for user_idx in range(predicted_ratings.shape[0]):
            # mark already seen items
            user_history = (R_filtered[user_idx, :] > 0)
            rec_indices = mmr_model.mmr(user_idx, user_history, top_k =top_k)

            # Map recommended indices to item IDs
            recs = []
            for idx in rec_indices:
                if idx < len(item_ids):  # Safety check
                    recs.append(item_ids[idx])
                else:
                    print(f"Warning: Recommendation index {idx} out of bounds")
            user_recs[user_idx] = recs

        # Convert recommendations to Rectools DataFrame format
        recs_recods = []
        for user, items, in user_recs.items():
            for rank, item in enumerate(items, start=1):
                recs_recods.append((user,item,rank))

        # COnvert recomemndation rectools format
        recs_df = pd.DataFrame(recs_recods, columns=["user", "item", "rank"])


        # Compute relevance metric (NDCG)
        ndcg_val = ndcg_metric.calc_per_user(
            reco=recs_df.rename(
                columns={"user": "user_id", "item": "item_id", "rank": "rank"}),
                interactions=interactions_df
        ).mean()

        #Compute diversity metric (ILD)
        ild_val = ild_metric.calc_per_user(
            reco=recs_df.rename(
                columns={"user": "user_id", "item": "item_id", "rank": "rank"})
        ).mean()

        # Store raw values for normalization
        ndcg_vals.append(ndcg_val)
        ild_vals.append(ild_val)


    #  Normalize metrics to [0,1] for fair weighting
    ndcg_min, ndcg_max = min(ndcg_vals), max(ndcg_vals)
    ild_min, ild_max = min(ild_vals), max(ild_vals)

    # Compute combined score for each lambda and find the best one
    for i, lam in enumerate(lambda_grid):
        ndcg_norm = (ndcg_vals[i] - ndcg_min) / (ndcg_max - ndcg_min + 1e-8)
        ild_norm = (ild_vals[i] - ild_min) / (ild_max - ild_min + 1e-8)
        # transform ild_norm since higher similarity -> lower diversity
        ild_as_similarity = 1 - ild_norm
        score = relevance_weight * ndcg_norm + diversity_weight * ild_as_similarity

        if score > best_score:
            best_score = score
            best_lambda = lam


    print(f"Best lambda: {best_lambda} with score {best_score:.4f}")

    return best_lambda, best_score
            




# ==========================
# Save Top-10 of every user as CSV
# ==========================
def process_save_mmr(all_recs, user_ids, item_ids, predicted_ratings, top_n = 10, output_file_path = None):
    results = []

    # Loop through all users and their recommended item indices
    for user_idx, rec_indices in enumerate(all_recs):
        user_id = user_ids[user_idx]
        # Process the user's top-10 MMR recommendations and append to results
        process_mmr(
            user_id=user_id,
            user_idx=user_idx,
            mmr_indices=rec_indices,
            item_ids=item_ids,
            predicted_ratings=predicted_ratings,
            mmr_recommendations_list=results, 
            top_n = top_n)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    #save to csv
    mmr_df = pd.DataFrame(results)
    mmr_df.to_csv(output_file_path, index=False)
    print(f"MMR results saved: {output_file_path}")




def process_mmr(user_id, user_idx, mmr_indices, item_ids, predicted_ratings, mmr_recommendations_list, top_n=10):

    for rank, col_idx in enumerate(mmr_indices[:top_n], start = 1):
        # Skip if the predicted index is out of bounds
        if col_idx >= len(item_ids):
            print(f"Warning: col_idx {col_idx} out of bounds for user {user_id}, skipping")
            continue  # skip invalid index

        # Get the actual item ID from the column index
        item_id = int(item_ids[col_idx])

        # Get the predicted rating score for this user-item pair
        score =  predicted_ratings[user_idx, col_idx]
        
        # Append recommendation info to the results list
        mmr_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'itemId': str(item_id),
            'predictedRating': score,
        })
    