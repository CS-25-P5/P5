from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os



class MMR:
    def __init__(self, movie_titles, genre_map, all_genres, predicted_ratings, similarity_type='jaccard', lambda_param=0.7):
        self.movie_titles = movie_titles
        self.genre_map = genre_map
        self.all_genres = all_genres
        self.predicted_ratings = predicted_ratings
        self.similiarity_type = similarity_type
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
        mat = np.zeros((len(self.movie_titles), len(self.all_genres)), dtype=np.float32)

        # fill matrix so it indicates which genres each movie belong to
        for i, title in enumerate(self.movie_titles):
            for g in self.genre_map[title]:
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


def build_mmr_models(movie_titles, genre_map, all_genres, predicted_ratings, lambda_param):
    mmr_cosine = MMR(
        movie_titles=movie_titles,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="cosine",
        lambda_param=lambda_param
    )


    mmr_jaccard = MMR(
        movie_titles=movie_titles,
        genre_map=genre_map,
        all_genres=all_genres,
        predicted_ratings=predicted_ratings,
        similarity_type="jaccard",
        lambda_param=lambda_param
    )

    return mmr_cosine, mmr_jaccard


def get_recommendations_for_mmr(mmr_model,R_filtered , item_user_rating, item_titles, genre_map, 
                                predicted_ratings, top_k, top_n, output_dir, similarity_type):
    results = []

    for user_idx, user_id in enumerate(item_user_rating.index):
        user_history = (R_filtered[user_idx, :] > 0)

        mmr_indices = mmr_model.mmr(
            user_id = user_idx,
            user_history = user_history,
            top_k = top_k
        )

        process_mmr(
            user_id, user_idx, mmr_indices, 
            item_titles, genre_map, predicted_ratings, 
            results, top_n)
        
    # save result as csv
    save_mmr_results(results, output_dir, similarity_type)
    print(f"Done MMR for {similarity_type}")



def process_mmr(user_id, user_idx, mmr_indices, item_names, genre_map, predicted_ratings, mmr_recommendations_list, top_n=10):


    for rank, idx in enumerate(mmr_indices[:top_n], start = 1):
        item = item_names[idx]
        # handle missing genres
        item_genres = genre_map.get(item, set())
        genres = ",".join(item_genres)


        mmr_recommendations_list.append({
            'userId': user_id,
            'rank': rank,
            'title': item,
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


def save_mmr_results(mmr_recommendations_list, output_dir, similarity_type="jaccard"):
    # create output dataframe
    mmr_df = pd.DataFrame(mmr_recommendations_list)

    #Ensure directory exists
    os.makedirs(output_dir,exist_ok=True)

    #Build defalut filename
    output_file_path = os.path.join(output_dir, f"mmr_train_{similarity_type}_recommendations.csv")

    #save to csv
    mmr_df.to_csv(output_file_path, index=False)
 
    print(f"MMR results saved: {output_file_path}")





