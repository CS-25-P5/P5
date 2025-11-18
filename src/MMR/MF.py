import pandas as pd
import numpy as np
import os




class MatrixFactorization:
    def __init__(self, R, k=20, alpha=0.01, lamda_=0.1, n_epochs=50):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.alpha = alpha
        self.lambda_ = lamda_
        self.n_epochs = n_epochs

    def train(self):
        #initialize latent factors and biases
        self.P = np.random.normal(scale=0.1, size=(self.num_users,self.k))
        self.Q = np.random.normal(scale=0.1, size=(self.num_items,self.k))
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = np.mean(self.R[self.R>0])

        # Precompute the indices of known ratings
        known_ratings = np.array(np.where(self.R > 0)).T 

        for epoch in range(self.n_epochs):
            np.random.shuffle(known_ratings)
            for u,i in known_ratings:
                prediction = self.predict_single(u,i)
                error = self.R[u,i] - prediction

                #update parameter
                self.b_u[u] += self.alpha * (error - self.lambda_ * self.b_u[u])
                self.b_i[i] += self.alpha * (error - self.lambda_ * self.b_i[i])


                Pu_old = self.P[u, :].copy()
                Qi_old = self.Q[i, :].copy()

                self.P[u, :] += self.alpha * (error * Qi_old - self.lambda_ * Pu_old)
                self.Q[i, :] += self.alpha * (error * Pu_old - self.lambda_ * Qi_old)

        loss = self.compute_loss()
        print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.4f}")



    def predict_single(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)


    def full_prediction(self):
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis: , ] + self.P.dot(self.Q.T)


    def compute_loss(self):
        loss = 0

        for u in range(self.num_users):
            for i in range(self.num_items):
                if self.R[u,i] > 0:
                    loss += (self.R[u,i] - self.predict_single(u,i)) ** 2


        # Regularization
        loss += self.lambda_ * (np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.b_u**2) + np.sum(self.b_i**2))

        return loss
    


def load_and_prepare_matrix(ratings_file_path, movies_file_path, nrows_movies=None):
    # Check if files exist before loading
    if not os.path.exists(ratings_file_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file_path}")
    if not os.path.exists(movies_file_path):
        raise FileNotFoundError(f"Movies file not found: {movies_file_path}")
    
    # Load the files
    ratings = pd.read_csv(ratings_file_path)
    movies = pd.read_csv(movies_file_path, nrows=nrows_movies)

    #Merge and Clean
    combine_m_r = pd.merge(ratings,movies, on='movieId')
    # drop timestamp and rows
    combine_m_r = combine_m_r.drop('timestamp',axis=1)
    combine_m_r = combine_m_r.dropna(axis=0,subset=['title'])

    # Pivot data into user-item rating matrix
    movie_user_rating = combine_m_r.pivot(index='userId', columns='title', values='rating').fillna(0)


    #build genre map (title to set of genres )
    genre_map = {}
    for _,row in movies.iterrows():
        genres = row['genres']

        if isinstance(genres, str):
            genre_set = set(genres.split('|'))
        else:
            genre_set = set()
        genre_map[row['title']] = genre_set


    # build all unique genre list
    all_genres = set()
    for genres in genre_map.values():
        all_genres.update(genres)
    all_genres = sorted(all_genres)



    return movie_user_rating, genre_map, all_genres


def filter_empty_users_data(R, movie_titles=None):
    # keep users and movies with at least on rating
    #user_filter = R.sum(axis = 1) > 0

    movie_filter = R.sum(axis = 0) > 0

    R_filtered = R[:, movie_filter]

    filtered_movie_titles = movie_titles[movie_filter] if movie_titles is not None else None

    return R_filtered, filtered_movie_titles

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
    







    