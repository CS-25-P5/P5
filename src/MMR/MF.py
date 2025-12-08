import pandas as pd
import numpy as np
import os, csv




class MatrixFactorization:
    def __init__(self, R, k=20, alpha=0.01, lamda_=0.1, n_epochs=50, random_state=42, item_ids=None):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.alpha = alpha
        self.lambda_ = lamda_
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.item_ids  = item_ids if item_ids is not None else np.arange(self.num_items)
        np.random.seed(self.random_state)

    def train(self):
        #initialize latent factors and biases
        # scale = 0.1 -> genreated valued will be close to 0, given matrix shape u x k and i xk
        self.P = np.random.normal(scale=0.1, size=(self.num_users,self.k))
        self.Q = np.random.normal(scale=0.1, size=(self.num_items,self.k))
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = np.mean(self.R[self.R>0])


        # Precompute the indices of known ratings
        known_ratings = np.array(np.where(self.R > 0)).T

        loss_history = []

        for epoch in range(self.n_epochs):
            np.random.seed(self.random_state + epoch)
            np.random.shuffle(known_ratings)
            # Collect squraed error for this epoch
            epoch_errors = []
            for u,i in known_ratings:
                prediction = self.predict_single(u,i)
                error = self.R[u,i] - prediction

                #append squared error
                epoch_errors.append(error ** 2)

                #update parameter
                self.b_u[u] += self.alpha * (error - self.lambda_ * self.b_u[u])
                self.b_i[i] += self.alpha * (error - self.lambda_ * self.b_i[i])


                Pu_old = self.P[u, :].copy()
                Qi_old = self.Q[i, :].copy()

                self.P[u, :] += self.alpha * (error * Qi_old - self.lambda_ * Pu_old)
                self.Q[i, :] += self.alpha * (error * Pu_old - self.lambda_ * Qi_old)


            # compute average loss for the epoch
            epoch_loss = np.mean(epoch_errors)
            rmse = np.sqrt(epoch_loss)
            loss_history.append(epoch_loss)

            full_loss = self.compute_loss()
            #print(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {rmse:.4f}, Fullloss: {full_loss: .4f}")

        return rmse, self.random_state


    def predict_single(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)


    def full_prediction(self):
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis: , ] + self.P.dot(self.Q.T)


    def compute_loss(self):
        loss = 0

        #iterate over all users and items
        for u in range(self.num_users):
            for i in range(self.num_items):
                # check if rating exist
                if self.R[u,i] > 0:
                    # adds the squared error for each observed rating
                    loss += (self.R[u,i] - self.predict_single(u,i)) ** 2


        # Regularization
        loss += self.lambda_ * (np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.b_u**2) + np.sum(self.b_i**2))

        return loss


    def compute_rmse(self, R_eval, R_pred):
        users, items = np.where(R_eval > 0)
        squared_errors = (R_eval[users,items] - R_pred[users, items])**2
        return np.sqrt(np.mean(squared_errors))




def load_and_prepare_matrix(ratings_file_path, item_file_path):
    # Check if files exist before loading
    if not os.path.exists(ratings_file_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file_path}")
    if not os.path.exists(item_file_path):
        raise FileNotFoundError(f"Item file not found: {item_file_path}")


    # Load the files
    ratings = pd.read_csv(ratings_file_path)
    items = pd.read_csv(item_file_path)

    ratings['itemId'] = ratings['itemId'].astype(str)
    items['itemId'] = items['itemId'].astype(str)

    #Merge and Clean
    combine = pd.merge(ratings,items, on='itemId')

    # Remove duplicates by averaging ratings per user-movie pair
    combine = combine.groupby(['userId', 'itemId'], as_index=False).agg({'rating': 'mean'})

    # Pivot data into user-item rating matrix
    user_item_matrix = combine.pivot(index='userId', columns='itemId', values='rating').fillna(0)

    #build genre map (title to set of genres )
    genre_map = {}
    id_to_title = {}
    for _,row in items.iterrows():
        genres = row['genres']
        title = row['title']
        item_id = row['itemId']

        id_to_title[item_id] = title

        if isinstance(genres, str):
            genre_set = set(genres.split('|'))
        else:
            genre_set = set()
        genre_map[item_id] = genre_set


    # build all unique genre list
    all_genres = set()
    for genres in genre_map.values():
        all_genres.update(genres)
    all_genres = sorted(all_genres)

    return user_item_matrix, genre_map, all_genres, id_to_title



# def filter_empty_users_data(R, user_ids=None):
#     Filter users (keep users with at least 1 rating)
#     if user_ids is not None:
#         user_filter = R.sum(axis=1) > 0
#         R = R[user_filter, :]
#         user_ids = user_ids[user_filter]
#     return R, user_ids


# def align_train_val_matrices(train_df, val_df):
#     # Find common users AND common items
#     common_users = train_df.index.intersection(val_df.index)
#     common_items = train_df.columns.intersection(val_df.columns)

#     # Align to common users and items only
#     train_aligned = train_df.loc[common_users, common_items]
#     val_aligned = val_df.loc[common_users, common_items]

#     return train_aligned, val_aligned


# def align_test_matrix(item_user_rating, trained_mf_model):
#     # trained_items = np.array(trained_mf_model.item_ids)

#     # # Only keep columns that exist in trained MF model
#     # common_items = [i for i in item_user_rating.columns if i in trained_items]
#     # aligned_test = item_user_rating.reindex(columns=common_items, fill_value=0)

#     # # Convert to numpy
#     # R_aligned_test = aligned_test.values

#     # # Ensure NumPy array of strings for item_ids
#     # common_items_array = np.array(common_items)

#     # return R_aligned_test, common_items_array

#     trained_items = np.array([str(i) for i in trained_mf_model.item_ids])

#     # Only keep columns that exist in trained MF model
#     common_items = [str(i) for i in item_user_rating.columns if str(i) in trained_items]

#     aligned_test = item_user_rating.reindex(columns=common_items, fill_value=0)

#     # Convert to numpy
#     R_aligned_test = aligned_test.values

#     # Ensure NumPy array of strings for item_ids
#     common_items_array = np.array(common_items)

#     print("Aligned test items:", common_items)
#     print("Number of items dropped:", len(item_user_rating.columns) - len(common_items))


#     return R_aligned_test, common_items_array





# def get_aligned_predictions(trained_mf_model, filtered_item_ids):
#     trained_items = np.array([str(i) for i in trained_mf_model.item_ids])
#     filtered_item_ids_str = np.array([str(i) for i in filtered_item_ids])

#     item_indices_in_mf = []
#     for item_id in filtered_item_ids_str:
#         # Check if the item exists in trained MF
#         if item_id in trained_items:
#             index = np.where(trained_items == item_id)[0][0]
#             item_indices_in_mf.append(index)


#     # Get the predicted ratings for the filtered items
#     predicted_ratings_filtered = trained_mf_model.full_prediction()[:, item_indices_in_mf]

#     return predicted_ratings_filtered



def save_mf_top_n(all_recommendations, genre_map, id_to_title, output_path="mf_predictions.csv"):
    # ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for user_id, recs in all_recommendations.items():
        for item_id, score in recs:
            rows.append({
                "userId": user_id,
                "itemId": item_id,
                "title": id_to_title.get(item_id, ""),
                "mf_score":score,
                "genres": ",".join(genre_map.get(item_id,[])) if genre_map else ""
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"MF predictions saved: {output_path}")


def get_top_n_recommendations_MF(genre_map, predicted_ratings, R_filtered, filtered_user_ids, filtered_item_ids,  id_to_title, top_n=10, save_path=None ):

    # Convert filtered_item_ids to NumPy array to allow array indexing
    filtered_item_ids = np.array(filtered_item_ids)

    # store all recomendations for all users
    all_recomenndations = {}

    for user_idx, user_id in enumerate(filtered_user_ids):
        # Get all predicted movie ratings for user
        user_ratings = predicted_ratings[user_idx, :]

        # Boolean series of movie rating status
        already_rated = R_filtered[user_idx, :]> 0

        # Filter out already rated movies
        user_ratings_filtered = np.where(already_rated, -np.inf, user_ratings)

        # get indicies sorted descending
        sorted_indices = np.argsort(user_ratings_filtered)[::-1]

        # Tak top N or fewer if not enough movies
        top_indices = sorted_indices[:min(top_n, len(sorted_indices))]


        # Map top indices to item IDs
        top_items = filtered_item_ids[top_indices]
        top_scores = user_ratings_filtered[top_indices]

        #store a list of (movie, predicted rating)
        all_recomenndations[user_id] = list(zip(top_items, top_scores))

        # MMR-style output for this user
    #     print("--------------------------------------------------------------------")
    #     print(f"Top {top_n} movies for User {user_id} (Matrix Factorization):")
    #     for rank, (movie, score) in enumerate(zip(top_movies, top_scores), start=1):
    #         genres = ",".join(genre_map.get(movie, []))
    #         print(f"{rank}. {movie} — Predicted rating: {score:.2f} | Genres {genres}")
    # print("--------------------------------------------------------------------")

    save_mf_top_n(all_recomenndations, genre_map, id_to_title, save_path)





def tune_mf( R_train, R_val,
             n_epochs=50,
             hyperparams_grid = {
                 "k": [20, 40, 60],
                 "alpha": [0.005, 0.01, 0.02],
                 "lambda_": [0.05, 0.1, 0.2]
             }
             ):

    best_rmse = float('inf')
    best_params = None

    for k in hyperparams_grid["k"]:
        for alpha in hyperparams_grid["alpha"]:
            for lambda_ in hyperparams_grid["lambda_"]:
                mf = MatrixFactorization(R_train, k, alpha, lambda_, n_epochs)
                mf.train()

                # predict on validation set
                pred_val = mf.full_prediction()

                #Compute RMSE correctly
                val_rmse = mf.compute_rmse(R_val,pred_val)

                #print(f"Testing on k={k}, alpha={alpha}, lambda_={lambda_} -> RMSE={val_rmse:.4f}")

                #Keep the best configuration

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_params = {
                        "k":k,
                        "alpha": alpha,
                        "lambda_": lambda_}
                    #print(f"New best params found using VAL RMSE: {best_params}, val_rmse={val_rmse:.4f}")

    print(f"Best MF params: {best_params}, RMSE={best_rmse:.4f}")
    return best_params


def train_mf_with_best_params(R_filtered, best_params, n_epochs=50,  random_state= 42):
    mf= MatrixFactorization(R_filtered, best_params["k"],  best_params["alpha"], best_params["lambda_"], n_epochs, random_state)
    train_rmse, random_state = mf.train()
    predicted_ratings = mf.full_prediction()

    return mf, predicted_ratings, train_rmse, random_state


def save_mf_predictions(trained_mf_model, train_user_ids, train_item_ids, ground_truth_path, output_path="mf_rating_predictions.csv"):
    test_df = pd.read_csv(ground_truth_path)

    # Convert all IDs to strings (clean format)
    test_df['userId'] = test_df['userId'].apply(lambda x: str(int(float(x))))
    test_df['itemId'] = test_df['itemId'].apply(lambda x: str(int(float(x))))
    train_user_ids = [str(int(float(u))) for u in train_user_ids]
    train_item_ids = [str(int(float(i))) for i in train_item_ids]

    # Create mappings from IDs to matrix indices
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(train_user_ids)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(train_item_ids)}

    # Get all MF predictions
    all_predictions = trained_mf_model.full_prediction()

    # Generate predictions for test set
    results = []
    for _, row in test_df.iterrows():
        user_str = row['userId']
        item_str = row['itemId']

        if user_str in user_id_to_idx and item_str in item_id_to_idx:
            user_idx = user_id_to_idx[user_str]
            item_idx = item_id_to_idx[item_str]
            mf_prediction = all_predictions[user_idx, item_idx]
        else:
            # Cold-start user/item fallback
            mf_prediction = 0.0

        results.append({
            'userId': row['userId'],
            'itemId': row['itemId'],
            'true_rating': row['rating'],
            'mf_prediction': mf_prediction
        })

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"MF test predictions saved to: {output_path}")