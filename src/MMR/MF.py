import pandas as pd
import numpy as np
import os, csv




class MatrixFactorization:
    def __init__(self, R, k=20, alpha=0.01, lamda_=0.1, n_epochs=50, random_state=42):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.alpha = alpha
        self.lambda_ = lamda_
        self.n_epochs = n_epochs
        self.random_state = random_state
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

        return rmse


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




def load_and_prepare_matrix(ratings_file_path, item_file_path,nrows_items=None):
    # Check if files exist before loading
    if not os.path.exists(ratings_file_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file_path}")
    if not os.path.exists(item_file_path):
        raise FileNotFoundError(f"Item file not found: {item_file_path}")

    # Load the files
    ratings = pd.read_csv(ratings_file_path)

    items = pd.read_csv(item_file_path, nrows=nrows_items)
    
    ratings['itemId'] = ratings['itemId'].astype(str)
    items['itemId'] = items['itemId'].astype(str)

    if "movieId" in items.columns and "itemId" not in items.columns:
        items = items.rename(columns={"movieId": "itemId"})

    print("ITEMS COLUMNS:", items.columns)
    print("RATINGS COLUMNS:", ratings.columns)

    #Merge and Clean
    combine = pd.merge(ratings,items, on='itemId')

    # Drop rows without a title
    combine = combine.dropna(subset=['title'])


    # Remove duplicates by averaging ratings per user-movie pair
    combine = combine.groupby(['userId', 'title'], as_index=False).agg({'rating': 'mean'})


    # Pivot data into user-item rating matrix
    user_item_matrix = combine.pivot(index='userId', columns='title', values='rating').fillna(0)


    #build genre map (title to set of genres )
    genre_map = {}
    for _,row in items.iterrows():
        genres = row['genres']
        title = row['title']

        if isinstance(genres, str):
            genre_set = set(genres.split('|'))
        else:
            genre_set = set()
        genre_map[title] = genre_set


    # build all unique genre list
    all_genres = set()
    for genres in genre_map.values():
        all_genres.update(genres)
    all_genres = sorted(all_genres)

    return user_item_matrix, genre_map, all_genres


def filter_empty_users_data(R, movie_titles=None):
    # keep users and movies with at least on rating
    #user_filter = R.sum(axis = 1) > 0

    movie_filter = R.sum(axis = 0) > 0

    R_filtered = R[:, movie_filter]

    filtered_movie_titles = movie_titles[movie_filter] if movie_titles is not None else None

    return R_filtered, filtered_movie_titles


def align_train_val_matrices(train_df, val_df):
    # Find common movies
    train_items = set(train_df.columns)
    val_items = set(val_df.columns)
    common_items = sorted(train_items & val_items)

    if len(common_items) == 0:
        raise ValueError("No overlapping itemIds  between train and validation")


    # SUbset and reorder both matrices
    train_aligned = train_df[common_items]
    val_aligned = val_df[common_items]

    return train_aligned, val_aligned




def save_mf_predictions(all_recommendations, genre_map, output_path="mf_predictions.csv"):
    # ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for user_id, recs in all_recommendations.items():
        for movie, score in recs:
            rows.append({
                "userId": user_id,
                "title": movie,
                "mf_score":score,
                "genres": ",".join(genre_map.get(movie,[])) if genre_map else ""
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"MF predictions saved: {output_path}")


def get_top_n_recommendations_MF(genre_map, predicted_ratings, R_filtered, filtered_user_ids, filtered_movie_titles, top_n=10, save_path=None ):
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


        # Map to movies titles and scors
        top_movies = filtered_movie_titles[top_indices]
        top_scores = user_ratings_filtered[top_indices]

        #store a list of (movie, predicted rating)
        all_recomenndations[user_id] = list(zip(top_movies, top_scores))

        # MMR-style output for this user
    #     print("--------------------------------------------------------------------")
    #     print(f"Top {top_n} movies for User {user_id} (Matrix Factorization):")
    #     for rank, (movie, score) in enumerate(zip(top_movies, top_scores), start=1):
    #         genres = ",".join(genre_map.get(movie, []))
    #         print(f"{rank}. {movie} — Predicted rating: {score:.2f} | Genres {genres}")
    # print("--------------------------------------------------------------------")

    save_mf_predictions(all_recomenndations, genre_map, save_path)





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


def train_mf_with_best_params(R_filtered, best_params, n_epochs=50):
    mf = MatrixFactorization(R_filtered, best_params["k"],  best_params["alpha"], best_params["lambda_"], n_epochs)
    train_rmse = mf.train()
    predicted_ratings = mf.full_prediction()

    return mf, predicted_ratings, train_rmse



def log_mf_experiment(output_dir, params, train_rmse=None, val_rmse=None):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "mf_train_experiments_log.csv")

    # add RMSE to params
    params = params.copy()
    params['train_rmse'] = train_rmse
    params['val_rmse'] = val_rmse

    # check if log file exists
    file_exists = os.path.isfile(log_file)

    # append to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=params.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(params)