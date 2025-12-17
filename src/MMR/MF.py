import pandas as pd
import numpy as np
import os

class MatrixFactorization:
    def __init__(self, R, k=20, alpha=0.01, lambda_=0.1, n_epochs=50, random_state=42):
        #user-item rating matrix
        self.R = R                          
        self.num_users, self.num_items = R.shape
        #latent factor
        self.k = k 
        #learning rate                         
        self.alpha = alpha 
        #regularization                  
        self.lambda_ = lambda_             
        self.n_epochs = n_epochs
        self.random_state = random_state
        #ensure repoducible initialization
        np.random.seed(self.random_state)   

    def train(self, R_val=None):
        #Initialize model parameters
        # User latent factors (num_users x k)
        self.P = np.random.normal(scale=0.1, size=(self.num_users,self.k))

        # Item latent factors (num_items x k)
        self.Q = np.random.normal(scale=0.1, size=(self.num_items,self.k))

        # User and item biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        # Global mean of observed ratings
        self.mu = np.mean(self.R[self.R>0])

        # Precompute the indices of known ratings
        known_ratings = np.array(np.where(self.R > 0)).T 

        train_mse_history = []
        val_mse_history = []

        # Track best model according to validation RMSE
        best_val_rmse = float('inf')
        best_train_rmse = float('inf')
        best_epoch = None
        best_weights = None
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle observed ratings 
            np.random.seed(self.random_state + epoch)
            np.random.shuffle(known_ratings)

            epoch_errors = []

            # SGD updates
            for u,i in known_ratings:
                prediction = self.predict_single(u,i)

                # Prediction error
                error = self.R[u,i] - prediction
                epoch_errors.append(error ** 2)

                # Update biases
                self.b_u[u] += self.alpha * (error - self.lambda_ * self.b_u[u])
                self.b_i[i] += self.alpha * (error - self.lambda_ * self.b_i[i])

                # Save old latent vectors for simultaneous update
                Pu_old = self.P[u, :].copy()
                Qi_old = self.Q[i, :].copy()

                # Update latent factors
                self.P[u, :] += self.alpha * (error * Qi_old - self.lambda_ * Pu_old)
                self.Q[i, :] += self.alpha * (error * Pu_old - self.lambda_ * Qi_old)

            # Compute Training metrics
            train_epoch_mse = np.mean(epoch_errors)
            train_mse_history.append(train_epoch_mse)
            train_epoch_rmse = np.sqrt(train_epoch_mse)

            # Compute validation metrics 
            if R_val is not None:
                # Evaluate only on observed validation ratings
                users, items = np.where(R_val > 0)

                errors = np.array([
                    self.predict_single(u, i) - R_val[u, i] 
                    for u, i in zip(users, items)
                ])

                val_epoch_rmse = np.sqrt(np.mean(errors**2))
                val_mse_history.append(np.mean(errors**2))

                #  Save model if validation improves
                if val_epoch_rmse < best_val_rmse:
                    best_val_rmse = val_epoch_rmse
                    best_train_rmse = train_epoch_rmse
                    best_epoch = epoch + 1

                    # Save the best parameters
                    best_weights = {
                        'P': self.P.copy(),
                        'Q': self.Q.copy(),
                        'b_u': self.b_u.copy(),
                        'b_i': self.b_i.copy()
                    }

        # Restore bestmodel
        if best_weights is not None:
            self.P = best_weights['P']
            self.Q = best_weights['Q']
            self.b_u = best_weights['b_u']
            self.b_i = best_weights['b_i']

        print(f"Training complete. Best epoch: {best_epoch}, Best train RMSE: {best_train_rmse:.4f}, Best val RMSE: {best_val_rmse:.4f}")

        return (
            train_mse_history, 
            best_train_rmse, 
            val_mse_history, 
            best_val_rmse, 
            best_epoch)

    def predict_single(self, u, i):
        # Predict the rating for a single user-item pair (u, i).
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)

    def full_prediction(self):
        # Predict the entire rating matrix for all users and items.
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis: , ] + self.P.dot(self.Q.T)


# ==========================
# Tranform ratings and item metadate into a user-item matrix
# Extract genre infromtion for each item
# ==========================
def load_and_prepare_matrix(ratings_file_path, item_file_path):
    # Check if files exist before loading
    if not os.path.exists(ratings_file_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file_path}")
    if not os.path.exists(item_file_path):
        raise FileNotFoundError(f"Item file not found: {item_file_path}")
    
    # Load the ratings and item metdata
    ratings = pd.read_csv(ratings_file_path)
    items = pd.read_csv(item_file_path)

    # Ensure item IDs are strings for consistent merging
    ratings['itemId'] = ratings['itemId'].astype(str)
    items['itemId'] = items['itemId'].astype(str)

    #Merge  ratings with item metadata
    combine = pd.merge(ratings,items, on='itemId')

    # Remove duplicates by averaging ratings per user-movie pair
    combine = combine.groupby(['userId', 'itemId'], as_index=False).agg({'rating': 'mean'})

    # Pivot to create user-item rating matrix, where missing ratings is filled with 0 
    # Where Rows: users, Columns: items, Values: ratings. 
    user_item_matrix = combine.pivot(index='userId', columns='itemId', values='rating').fillna(0)

    #build genre map (itemid maped to set of genres)
    genre_map = {}
    for _,row in items.iterrows():
        genres = row['genres']
        item_id = row['itemId']

        if isinstance(genres, str):
            # Split genre string by '|' into a set
            genre_set = set(genres.split('|'))
        else:
            # Empty set if no genres listed
            genre_set = set()
        genre_map[item_id] = genre_set


    # Build list of all unique genres
    all_genres = set()
    for genres in genre_map.values():
        all_genres.update(genres)
    # Sort for consitent order
    all_genres = sorted(all_genres)

    return user_item_matrix, genre_map, all_genres


# ==========================
# Fine tune MF with hyperparameter alpha, lambda and k and train with best configuration
# ==========================
def tune_mf(R_train, R_val, n_epochs=50,
            hyperparams_grid = { 
                "alpha": [0.005, 0.01, 0.02],
                "lambda_": [0.01, 0.05, 0.1],
                "k": [20, 40, 60],
            }):
    
    # Stage 1: Tune alpha and lambda with fixed k 
    fixed_k = hyperparams_grid["k"][0] 
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

            if val_rmse < best_rmse_stage1:
                best_rmse_stage1 = val_rmse
                best_alpha_lambda = {"alpha": alpha, "lambda_": lambda_}

    print(f"Best (alpha, lambda) from Stage 1: {best_alpha_lambda}, RMSE={best_rmse_stage1:.4f}")

    # Stage 2: Tune k with best alpha and lambda 
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

        if val_rmse < best_rmse_stage2:
            best_rmse_stage2 = val_rmse
            best_k = k

    print(f"Best k from Stage 2: {best_k}, RMSE={best_rmse_stage2:.4f}")

    # Combine best hyperparameters 
    best_params = {
        "alpha": best_alpha_lambda["alpha"],
        "lambda_": best_alpha_lambda["lambda_"],
        "k": best_k
    }

    print(f"Best MF params: {best_params}, RMSE={best_rmse_stage2:.4f}")
    return best_params

def train_mf_with_best_params(R_filtered, best_params, R_val=None, n_epochs=50,  random_state= 42):
    mf= MatrixFactorization(
        R_filtered, 
        best_params["k"],  
        best_params["alpha"], 
        best_params["lambda_"], 
        n_epochs, 
        random_state)
    
    # train and get mse history and best metrics
    (
        train_mse_history, 
        best_train_rmse, 
        val_mse_history, 
        best_val_rmse, 
        best_epoch
    ) = mf.train(R_val=R_val)

    predicted_ratings = mf.full_prediction()

    return (
        mf, 
        predicted_ratings,  
        train_mse_history, 
        best_train_rmse, 
        val_mse_history, 
        best_val_rmse, 
        best_epoch)

# ==========================
# SAVE TOP-N for all users in CSV
# ==========================
def get_top_n_recommendations_MF(
        predicted_ratings,      # predicted ratings for each user-item pair
        R_filtered,             # actual user-item ratings matrix used for filtering
        filtered_user_ids,      # list of user IDs corresponding to rows in predicted_ratings
        filtered_item_ids,      # list of item IDs corresponding to columns in predicted_ratings
        top_n=10, 
        save_path=None 
        ):

    # Convert item IDs to NumPy array for easier indexingg
    filtered_item_ids = np.array(filtered_item_ids)
    all_recommendations = {}

    # Loop through each user
    for user_idx, _ in enumerate(filtered_user_ids):
        # Get predicted ratings for this user
        user_ratings = predicted_ratings[user_idx, :]

        # number of items in predictions
        num_pred_items = user_ratings.shape[0]

        # number of items in actual ratings
        num_rated_items = R_filtered.shape[1]

        # Build a mask for items the user has already rated
        already_rated = np.zeros(num_pred_items, dtype=bool)

        # Avoid index out of bounds by 
        # taking the min items of R_filtered or predicted_ratings
        items_to_check = min(num_pred_items, num_rated_items)

        #  Mark which items the current user has already rated.
        already_rated[:items_to_check] = R_filtered[user_idx, :items_to_check] > 0
            
        ratings_mask = user_ratings <= 0

        # Filter out already rated items
        user_ratings_filtered = np.where(already_rated | ratings_mask, -np.inf, user_ratings)

        # Sort items by predicting rating in descending order
        sorted_indices = np.argsort(user_ratings_filtered)[::-1]
        top_indices = sorted_indices[:top_n]

        # Map indices to actual item IDs
        top_items = [filtered_item_ids[i] for i in top_indices]
        all_recommendations[user_idx] = top_items

    # Save all recommedntion to CSV
    process_save_mf(
        all_recommendations=all_recommendations,
        user_ids=filtered_user_ids,         
        item_ids=filtered_item_ids,          
        predicted_ratings=predicted_ratings, 
        output_file_path=save_path,
        top_n = top_n
    )


def process_save_mf(all_recommendations, user_ids, item_ids, predicted_ratings, top_n=10, output_file_path="mf_predictions.csv"):
    results = []

    # Loop over all usrs and their top-N item indices
    for user_idx, item_indices  in all_recommendations.items():
        # Get the acutal user Id corresponding to the index
        raw_user_id  = user_ids[user_idx]

        # Process user's top_n recommendation and append to results
        process_mf(
            user_id=raw_user_id,
            user_idx=user_idx,
            mf_indices=item_indices,
            item_ids=item_ids,
            predicted_ratings=predicted_ratings,
            mf_recommendations_list=results,
            top_n=top_n
        )

    # Ensure the output directory exists to prevent errors when savings
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Convert results list into a DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False)
    print(f"MF predictions saved: {output_file_path}")



def process_mf(user_id, user_idx, mf_indices, item_ids, predicted_ratings,  mf_recommendations_list, top_n=10):
    # Create a mapping from item ID to column index in predicted_ratings matrix
    item_id_to_col = {i: j for j, i in enumerate(item_ids)}

    # Loop through the top-N recommended item indices for the user
    for rank, item_id in enumerate(mf_indices[:top_n], start=1):
        # Find the corresponding column index in the predicted ratings matrix
        col_idx = item_id_to_col[item_id]

        # Get the predicted rating score for this user-item pair
        score = predicted_ratings[user_idx, col_idx]

        # Append the recommendation info to the list
        mf_recommendations_list.append({
            "userId": user_id, 
            "rank": rank,
            "itemId": item_id,
            "predictedRating":score,
            })


# ==========================
# Save all the predictions
# ==========================
def save_mf_predictions(trained_mf_model, train_user_ids, train_item_ids, ground_truth_path, output_path="mf_rating_predictions.csv"):
    # Load the ground truth test ratings file
    test_df = pd.read_csv(ground_truth_path)

    # Standardize IDs - convert user/item IDs to string format
    # This ensures consistency between train IDs and test IDs
    test_df['userId'] = test_df['userId'].apply(lambda x: str(int(float(x))))
    test_df['itemId'] = test_df['itemId'].apply(lambda x: str(int(float(x))))
    train_user_ids = [str(int(float(u))) for u in train_user_ids]
    train_item_ids = [str(int(float(i))) for i in train_item_ids]

    # Create dictionaries mapping user/item IDs to their row/column indices in MF matrix
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(train_user_ids)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(train_item_ids)}

    # Get the full predicted ratings matrix from the trained MF model
    all_predictions = trained_mf_model.full_prediction()

    # Generate predictions for each entry in the test set
    results = []
    for _, row in test_df.iterrows():
        user_str = row['userId']
        item_str = row['itemId']

        if user_str in user_id_to_idx and item_str in item_id_to_idx:
            # Map user/item ID to corresponding index in predicted ratings matrix
            user_idx = user_id_to_idx[user_str]
            item_idx = item_id_to_idx[item_str]
            mf_prediction = all_predictions[user_idx, item_idx]
        else:
            # Cold-start user/item fallback
            mf_prediction = 0.0  

        # Store the prediction along with true rating
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