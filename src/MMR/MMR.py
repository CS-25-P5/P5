import rectools
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from rectools.metrics import (
    NDCG, IntraListDiversity
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator



class MMR:
    def __init__(self, item_ids, genre_map, all_genres, predicted_ratings, similarity_type='jaccard', lambda_param=0.7):
        self.item_ids = item_ids
        self.genre_map = genre_map
        self.all_genres = all_genres
        self.predicted_ratings = predicted_ratings
        self.similarity_type = similarity_type
        self.lambda_param = lambda_param

        # Build genre vector
        self.genre_vectors = self.build_genre_vectors()

        #Build similarity matrix
        if similarity_type == 'jaccard':
            self.sim_matrix = self.jaccard_matrix()
        elif similarity_type == 'cosine':
            self.sim_matrix = self.consine_matrix()
        else:
            raise ValueError("Invalid similarity type")


    # create binary_matrix a 2d array (rows -> movies, col -> genres)
    def build_genre_vectors(self):
        # assign index to each genre
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
    
        
    def jaccard_matrix(self):
        # Count how many genres each movie has
        row_sums = self.genre_vectors.sum(axis=1, keepdims=True)

        # counts how many genres two movies share
        intersection = self.genre_vectors @ self.genre_vectors.T

        # get the union
        union = row_sums + row_sums.T - intersection
        # 1e-12 prevents division by zero
        return intersection/(union + 1e-12)
    
    def consine_matrix(self):
        # calculate the norm
        norms = np.linalg.norm(self.genre_vectors, axis=1, keepdims=True)
        # calculate the denominator 
        denom = norms * norms.T + 1e-12
        return (self.genre_vectors @ self.genre_vectors.T) / denom
    

        
    def mmr(self, user_id, user_history, top_k=10):
        # predicted score for each item
        relevance = self.predicted_ratings[user_id]
        # items the user hasn't seen
        remaining = np.where(~user_history)[0].copy()  #~user_history flips True/False
        # store all indices of items chosen by MMR
        selected = []

        #Repeat up to top_k times or until no remaining items left.
        for _ in range(min(top_k, len(remaining))):
            # if no items are selected 
            if len(selected) == 0:
                mmr_scores = self.lambda_param * relevance[remaining]
            else:
                # get similarity bewteen remaning[i] and selected[j]
                # np.idx_ builds a grid
                diversity = self.sim_matrix[np.ix_(remaining, selected)].max(axis=1)
                mmr_scores = self.lambda_param * relevance[remaining] - (1- self.lambda_param) * diversity

            # select item with highest relvance/diversity score
            best_idx = np.argmax(mmr_scores)

            # get the actual index of item in datset
            best_item = remaining[best_idx]

            #add item to selected
            selected.append(best_item)
            # remove item just picked
            remaining = np.delete(remaining, best_idx)

        return selected


def run_mmr(mmr_model, R_filtered, top_k):
    all_recs = []
    for user_idx in range(R_filtered.shape[0]):
        user_history = (R_filtered[user_idx, :] > 0)
        rec_indices = mmr_model.mmr(user_idx, user_history, top_k)
        all_recs.append(rec_indices)
    
    return all_recs

def process_save_mmr(all_recs, item_user_rating, item_ids, predicted_ratings, genre_map, id_to_title, top_n, output_file_path):
    results = []
    for user_idx, rec_indices in enumerate(all_recs):
        user_id = item_user_rating.index[user_idx]
        process_mmr(
            user_id=user_id,
            user_idx=user_idx,
            mmr_indices=rec_indices,
            item_ids=item_ids,
            genre_map=genre_map,
            predicted_ratings=predicted_ratings,
            mmr_recommendations_list=results,
            id_to_title=id_to_title,
            top_n=top_n)

    # save result as csv
    save_mmr_results(results, output_file_path)
    print(f"Done MMR for {output_file_path}")




def process_mmr(user_id, user_idx, mmr_indices, item_ids, genre_map, predicted_ratings, mmr_recommendations_list, id_to_title, top_n=10):

    for rank, idx in enumerate(mmr_indices[:top_n], start = 1):
        item_id = item_ids[idx]
        title = id_to_title.get(item_id, "")
        # handle missing genres
        item_genres = genre_map.get(item_id, set())
        genres = ",".join(item_genres)


        mmr_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'itemId': item_id ,
            'title': title,
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


def save_mmr_results(mmr_recommendations_list, output_file_path):
    # create output dataframe
    mmr_df = pd.DataFrame(mmr_recommendations_list)

    #save to csv
    mmr_df.to_csv(output_file_path, index=False)
 
    print(f"MMR results saved: {output_file_path}")



def tune_mmr_lambda(
        mmr_builder,
        predicted_ratings,
        R_filtered,
        val_data,
        item_ids,
        k_eval=10,
        relevance_weight = 0.6,
        diversity_weight = 0.4
        ):
    
    lambda_grid = np.linspace(0, 1, 21)
    best_lambda = None
    best_score = -np.inf

    # builds mmr model to extract gnere vectors
    sample_mmr = mmr_builder(lambda_grid[0])
    genre_matrix = sample_mmr.genre_vectors


    # Build DataFrame for rectools distnace calculation
    genre_df = pd.DataFrame(
        genre_matrix,
        index = item_ids,
        columns=sample_mmr.all_genres
    )

    # Replace NaN with 0 in genre matrix
    genre_df = genre_df.fillna(0)

    # Ensure matrix is int (0/1)
    genre_df = genre_df.astype(int)

    # Create Hamming distance calculator for ILD
    distance_calc = PairwiseHammingDistanceCalculator(genre_df)

    # Define metrices
    ndcg_metric = NDCG(k=k_eval)
    ild_metric = IntraListDiversity(k=k_eval, distance_calculator=distance_calc)


    val_array = val_data.values
    rows, cols = np.where(val_array > 0)

    print(f"Validation data: {len(rows)} ratings")
    print(f"Item IDs length: {len(item_ids)}")
    print(f"Max column index: {cols.max() if len(cols) > 0 else 0}")

    # Map column indices to actual item IDs
    mapped_item_ids = []
    for c in cols:
        if c < len(item_ids):  # Safety check
            mapped_item_ids.append(item_ids[c])
        else:
            print(f"Warning: Column index {c} out of bounds for item_ids")
            mapped_item_ids.append(None)

    # Convert validation data to Rectools format using titles
    val_array = val_data.values
    rows, cols = np.where(val_array > 0)
    interactions_df = pd.DataFrame({
        "user_id": rows,
        "item_id": mapped_item_ids,  # map column indices to titles
        "rating": val_array[rows, cols]
    })

    # Store raw NDGC and ILD for normalization
    ndcg_vals = []
    ild_vals = []

    # Loop over cnadidata lambda values
    for lam in lambda_grid:
        #Build new MMR model for each lambda
        mmr_model = mmr_builder(lam)

        # Generate recomenndations
        user_recs = {}

        # genreate top-k recommendation for each user
        for user_idx in range(predicted_ratings.shape[0]):
            user_history = (R_filtered[user_idx, :] > 0)
            rec_indices = mmr_model.mmr(user_idx, user_history, top_k =k_eval)
            
       
            recs = []
            for idx in rec_indices:
                if idx < len(item_ids):  # Safety check
                    recs.append(item_ids[idx])
                else:
                    print(f"Warning: Recommendation index {idx} out of bounds")
            user_recs[user_idx] = recs

        # Convert to Rectools input format
        recs_recods = []
        for user, items, in user_recs.items():
            for rank, item in enumerate(items, start=1):
                recs_recods.append((user,item,rank))

        # COnvert recomemndation rectools format
        recs_df = pd.DataFrame(recs_recods, columns=["user", "item", "rank"])


        # compute NDCG
        ndcg_val = ndcg_metric.calc_per_user(
            reco=recs_df.rename(columns={"user": "user_id", "item": "item_id", "rank": "rank"}),
            interactions=interactions_df
        )
        # takes mean over all users - overall relevnace score
        ndcg_val = ndcg_val.mean()


        #compute ILD - overall diverisity score
        ild_val = ild_metric.calc_per_user(
            reco=recs_df.rename(columns={"user": "user_id", "item": "item_id", "rank": "rank"})).mean()

        ndcg_vals.append(ndcg_val)
        ild_vals.append(ild_val)


    # Normalize metrics (0-1 scaling)
    ndcg_min, ndcg_max = min(ndcg_vals), max(ndcg_vals)
    ild_min, ild_max = min(ild_vals), max(ild_vals)

    for i, lam in enumerate(lambda_grid):
        ndcg_norm = (ndcg_vals[i] - ndcg_min) / (ndcg_max - ndcg_min + 1e-8)
        ild_norm = (ild_vals[i] - ild_min) / (ild_max - ild_min + 1e-8)
        score = relevance_weight * ndcg_norm + diversity_weight * ild_norm
        # print(f"Lambda: {lam}, NDGC_raw: {ndcg_vals[i]:.4f}, ILD_raw: {ild_vals[i]:.4f}, "
        #     f"NDGC_norm: {ndcg_norm:.4f}, ILD_norm: {ild_norm:.4f}, Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_lambda = lam


    print(f"Best lambda: {best_lambda} with score {best_score:.4f}")

    return best_lambda, best_score
            


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

        