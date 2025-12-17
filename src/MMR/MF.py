import pandas as pd
import numpy as np
import os, csv




class MatrixFactorization:
    def __init__(self, R, k=20, alpha=0.01, lambda_=0.1, n_epochs=50, random_state=42, item_ids=None):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.alpha = alpha
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.item_ids  = item_ids if item_ids is not None else np.arange(self.num_items)
        np.random.seed(self.random_state)

    def train(self, R_val=None):
        #initialize latent factors and biases
        # scale = 0.1 -> genreated valued will be close to 0, given matrix shape u x k and i xk
        self.P = np.random.normal(scale=0.1, size=(self.num_users,self.k))
        self.Q = np.random.normal(scale=0.1, size=(self.num_items,self.k))
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = np.mean(self.R[self.R>0])

        # Precompute the indices of known ratings
        known_ratings = np.array(np.where(self.R > 0)).T

        train_mse_history = []
        val_mse_history = []

        best_val_rmse = float('inf')
        best_train_rmse = float('inf')
        best_epoch = None
        best_weights = None


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

            # Compute average training loss for the epoch
            train_epoch_mse = np.mean(epoch_errors)
            train_mse_history.append(train_epoch_mse)
            train_epoch_rmse = np.sqrt(train_epoch_mse)


            # Compute validation metrics if validation matrix is provided
            if R_val is not None:
                # R_val_pred = self.full_prediction()
                # val_epoch_mse = self.compute_mse(R_val, R_val_pred)
                # val_mse_history.append(val_epoch_mse)
                # val_epoch_rmse = np.sqrt(val_epoch_mse)

                users, items = np.where(R_val > 0)
                errors = np.array([self.predict_single(u, i) - R_val[u, i] for u, i in zip(users, items)])
                val_epoch_rmse = np.sqrt(np.mean(errors**2))
                val_mse_history.append(np.mean(errors**2))

                # Save best validation weights
                if val_epoch_rmse < best_val_rmse:
                    best_val_rmse = val_epoch_rmse
                    best_train_rmse = train_epoch_rmse
                    best_epoch = epoch + 1
                    # Save the best weights
                    best_weights = {
                        'P': self.P.copy(),
                        'Q': self.Q.copy(),
                        'b_u': self.b_u.copy(),
                        'b_i': self.b_i.copy()
                    }


        # Restore best weights after training
        if best_weights is not None:
            self.P = best_weights['P']
            self.Q = best_weights['Q']
            self.b_u = best_weights['b_u']
            self.b_i = best_weights['b_i']



        print(f"Training complete. Best epoch: {best_epoch}, Best train RMSE: {best_train_rmse:.4f}, Best val RMSE: {best_val_rmse:.4f}")

        return train_mse_history, best_train_rmse, val_mse_history, best_val_rmse, best_epoch


    def predict_single(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)

    def full_prediction(self):
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis: , ] + self.P.dot(self.Q.T)

    # def compute_loss(self):
    #     loss = 0

    #     #iterate over all users and items
    #     for u in range(self.num_users):
    #         for i in range(self.num_items):
    #             # check if rating exist
    #             if self.R[u,i] > 0:
    #                 # adds the squared error for each observed rating
    #                 loss += (self.R[u,i] - self.predict_single(u,i)) ** 2


    #     # Regularization
    #     loss += self.lambda_ * (np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.b_u**2) + np.sum(self.b_i**2))

    #     return loss


    def compute_rmse(self, R_eval, R_pred):
        rmse = np.sqrt(self.compute_mse(R_eval, R_pred))
        return rmse


    def compute_mse(self, R_true, R_pred):
        # Find indices of observed ratings
        users, items = np.where(R_true > 0)
        # Compute squared errors only for observed ratings
        squared_errors = (R_true[users, items] - R_pred[users, items]) ** 2
        # Return mean squared error
        mse = np.mean(squared_errors)

        return mse



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
    # id_to_title = {}
    for _,row in items.iterrows():
        genres = row['genres']
        # title = row['title']
        item_id = row['itemId']

        # id_to_title[item_id] = title

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

    return user_item_matrix, genre_map, all_genres


def process_save_mf(all_recommendations, user_ids, item_ids, predicted_ratings, top_n=10, output_file_path="mf_predictions.csv"):
    results = []

    for user_idx, item_indices  in all_recommendations.items():
        raw_user_id  = user_ids[user_idx]

        process_mf(
            user_id=raw_user_id,
            user_idx=user_idx,
            mf_indices=item_indices,
            item_ids=item_ids,
            predicted_ratings=predicted_ratings,
            mf_recommendations_list=results,
            top_n=top_n
        )

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False)
    print(f"MF predictions saved: {output_file_path}")



def process_mf(user_id, user_idx, mf_indices, item_ids, predicted_ratings,  mf_recommendations_list, top_n=10):

    # mf_item_ids now contains actual item IDs (not indices)
    item_id_to_col = {i: j for j, i in enumerate(item_ids)}

    for rank, item_id in enumerate(mf_indices[:top_n], start=1):
        # item_id = item_ids[idx]
        # score =  predicted_ratings[user_idx, idx]

        col_idx = item_id_to_col[item_id]
        score = predicted_ratings[user_idx, col_idx]

        mf_recommendations_list.append({
            "userId": user_id,
            "rank": rank,
            "itemId": item_id,
            "predictedRating":score,
        })



def get_top_n_recommendations_MF(predicted_ratings, R_filtered, filtered_user_ids, filtered_item_ids, top_n=10, save_path=None ):

    # Convert filtered_item_ids to NumPy array to allow array indexing
    filtered_item_ids = np.array(filtered_item_ids)

    # store all recomendations for all users
    all_recommendations = {}

    for user_idx, _ in enumerate(filtered_user_ids):
        # Get all predicted movie ratings for user
        user_ratings = predicted_ratings[user_idx, :]

        #already_rated = R_filtered[user_idx, :predicted_ratings.shape[1]] > 0

        #already_rated = R_filtered[user_idx, :user_ratings.shape[0]] > 0

        num_pred_items = user_ratings.shape[0]
        num_rated_items = R_filtered.shape[1]

        # Create already_rated mask aligned with predicted_ratings
        already_rated = np.zeros(num_pred_items, dtype=bool)
        # Only fill for items that exist in R_filtered
        items_to_check = min(num_pred_items, num_rated_items)
        already_rated[:items_to_check] = R_filtered[user_idx, :items_to_check] > 0

        ratings_mask = user_ratings <= 0

        # Ensure R_filtered row aligns with predicted_ratings
        # Filter out already rated items
        user_ratings_filtered = np.where(already_rated | ratings_mask, -np.inf, user_ratings)


        # Boolean series of movie rating status
        # already_rated = R_filtered[user_idx, :]> 0
        # valid_mask = user_ratings > 0


        # Filter out already rated items
        #user_ratings_filtered = np.where(already_rated, -np.inf, user_ratings)
        # user_ratings_filtered = np.where(already_rated | ~valid_mask, -np.inf, user_ratings)

        # get indicies sorted descending
        #sorted_indices = np.argsort(user_ratings_filtered)[::-1]

        # # Tak top N or fewer if not enough movies
        # top_indices = sorted_indices[:min(top_n, len(sorted_indices))]


        # Map top indices to item IDs
        # top_items = filtered_item_ids[top_indices]
        # top_scores = user_ratings_filtered[top_indices]

        sorted_indices = np.argsort(user_ratings_filtered)[::-1]
        top_indices = sorted_indices[:top_n]

        #store a list of (movie, predicted rating)
        #all_recommendations[user_id] = list(zip(top_items, top_scores))
        # Map indices to actual item IDs
        top_items = [filtered_item_ids[i] for i in top_indices]
        all_recommendations[user_idx] = top_items
        #all_recommendations[user_idx] = top_indices.tolist()

        #print(f"User {user_id}: {np.sum(~already_rated)} candidate items, top_n requested={top_n}")

        # MMR-style output for this user
    #     print("--------------------------------------------------------------------")
    #     print(f"Top {top_n} movies for User {user_id} (Matrix Factorization):")
    #     for rank, (movie, score) in enumerate(zip(top_movies, top_scores), start=1):
    #         genres = ",".join(genre_map.get(movie, []))
    #         print(f"{rank}. {movie} — Predicted rating: {score:.2f} | Genres {genres}")
    # print("--------------------------------------------------------------------")

    process_save_mf(
        all_recommendations=all_recommendations,
        user_ids=filtered_user_ids,
        item_ids=filtered_item_ids,
        predicted_ratings=predicted_ratings,
        output_file_path=save_path,
        top_n = top_n
    )




def tune_mf(R_train, R_val, n_epochs=50,
            hyperparams_grid = {
                "alpha": [0.005, 0.01, 0.02],
                "lambda_": [0.01, 0.05, 0.1],
                "k": [20, 40, 60],
            }):

    # --- Stage 1: Tune alpha and lambda with fixed k ---
    fixed_k = hyperparams_grid["k"][0]  # pick first k as fixed for stage 1
    best_rmse_stage1 = float("inf")
    best_alpha_lambda = None

    for alpha in hyperparams_grid["alpha"]:
        for lambda_ in hyperparams_grid["lambda_"]:
            mf = MatrixFactorization(
                R_train,
                k=fixed_k,
                alpha=alpha,
                lambda_=lambda_,
                n_epochs=n_epochs,
            )
            _, _, _, val_rmse,_  = mf.train(R_val=R_val)
            # pred_val = mf.full_prediction()
            # val_rmse = mf.compute_rmse(R_val, pred_val)

            if val_rmse < best_rmse_stage1:
                best_rmse_stage1 = val_rmse
                best_alpha_lambda = {"alpha": alpha, "lambda_": lambda_}

    print(f"Best (alpha, lambda) from Stage 1: {best_alpha_lambda}, RMSE={best_rmse_stage1:.4f}")

    # --- Stage 2: Tune k with best alpha and lambda ---
    best_rmse_stage2 = float("inf")
    best_k = None
    for k in hyperparams_grid["k"]:
        mf = MatrixFactorization(
            R_train,
            k=k,
            alpha=best_alpha_lambda["alpha"],
            lambda_=best_alpha_lambda["lambda_"],
            n_epochs=n_epochs,
        )
        _, _, _, val_rmse,_ = mf.train(R_val=R_val)
        # pred_val = mf.full_prediction()
        # val_rmse = mf.compute_rmse(R_val, pred_val)

        if val_rmse < best_rmse_stage2:
            best_rmse_stage2 = val_rmse
            best_k = k

    print(f"Best k from Stage 2: {best_k}, RMSE={best_rmse_stage2:.4f}")

    # --- Combine best hyperparameters ---
    best_params = {
        "alpha": best_alpha_lambda["alpha"],
        "lambda_": best_alpha_lambda["lambda_"],
        "k": best_k
    }

    print(f"Best MF params: {best_params}, RMSE={best_rmse_stage2:.4f}")
    return best_params



# def tune_mf( R_train, R_val,
#         n_epochs=50,
#         hyperparams_grid = {
#         "alpha": [0.005, 0.01, 0.02],
#         "k": [20, 40, 60],
#         "lambda_": [0.01, 0.05, 0.1]
#     }
# ):

#     best_rmse = float('inf')
#     best_params = None

#     for alpha in hyperparams_grid["alpha"]:
#         for k in hyperparams_grid["k"]:
#             for lambda_ in hyperparams_grid["lambda_"]:
#                 mf = MatrixFactorization(R_train, k, alpha, lambda_, n_epochs)
#                 mf.train()

#                 # predict on validation set
#                 pred_val = mf.full_prediction()

#                 #Compute RMSE correctly
#                 val_rmse = mf.compute_rmse(R_val,pred_val)

#                 #print(f"Testing on k={k}, alpha={alpha}, lambda_={lambda_} -> RMSE={val_rmse:.4f}")

#                 #Keep the best configuration

#                 if val_rmse < best_rmse:
#                     best_rmse = val_rmse
#                     best_params = {
#                         "k":k,
#                         "alpha": alpha,
#                         "lambda_": lambda_}
#                     #print(f"New best params found using VAL RMSE: {best_params}, val_rmse={val_rmse:.4f}")

#     print(f"Best MF params: {best_params}, RMSE={best_rmse:.4f}")
#     return best_params


def train_mf_with_best_params(R_filtered, best_params, R_val=None, n_epochs=50,  random_state= 42):
    mf= MatrixFactorization(
        R_filtered,
        best_params["k"],
        best_params["alpha"],
        best_params["lambda_"],
        n_epochs,
        random_state)

    # train and get mse history and best metrics
    train_mse_history, best_train_rmse, val_mse_history, best_val_rmse, best_epoch = mf.train(R_val=R_val)
    predicted_ratings = mf.full_prediction()

    return mf, predicted_ratings,  train_mse_history, best_train_rmse, val_mse_history, best_val_rmse, best_epoch


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
            'predictedRating': mf_prediction
        })

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"MF test predictions saved to: {output_path}")