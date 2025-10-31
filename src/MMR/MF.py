import pandas as pd
import numpy as np

ratings = pd.read_csv("ratings.csv", nrows=10000)
movies = pd.read_csv("movies.csv", nrows=10000)
movies

combine_m_r = pd.merge(ratings,movies, on='movieId')
combine_m_r


combine_m_r = combine_m_r.drop('timestamp',axis=1)
combine_m_r

combine_m_r = combine_m_r.dropna(axis=0,subset=['title'])
combine_m_r

movie_rc = (combine_m_r.
groupby(by = ['title'])['rating'].
count().
reset_index().
rename(columns= {'rating': 'totalRatingCount'})
[['title', 'totalRatingCount']]
)
movie_rc

rating_with_totRc = combine_m_r.merge(movie_rc, left_on='title', right_on='title', how='left')
rating_with_totRc

# Matrix Factorization

movie_user_rating = rating_with_totRc.pivot(index='userId', columns='title', values='rating').fillna(0)
movie_user_rating

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

        for epoch in range(self.n_epochs):
            for u in range(self.num_users):
                for i in range(self.num_items):
                    if self.R[u,i]> 0:
                        prediction = self.predict_single(u,i)
                        error = self.R[u,i] - prediction

                        #update parameter
                        self.b_u[u] += self.alpha * (error - self.lambda_ * self.b_u[u])
                        self.b_i[i] += self.alpha * (error - self.lambda_ * self.b_i[i])
                        self.P[u, :] += self.alpha * (error * self.Q[i, :] - self.lambda_ * self.P[u,:])
                        self.Q[i, :] += self.alpha * (error * self.P[u, :] - self.lambda_ * self.Q[i,:])

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


R = movie_user_rating.values
mf = MatrixFactorization(R, k = 20, alpha=0.01, lamda_=0.1, n_epochs=50)
mf.train()
mf.full_prediction()

# Full predicted rating matrix
predicted_ratings = mf.full_prediction()

#convert back to DataFram with  movie titels and user IDS

predicted_ratings_df = pd.DataFrame(predicted_ratings,
                                    index = movie_user_rating.index,
                                    columns=movie_user_rating.columns
                                    )
predicted_ratings_df

#printing the movie list
movie_titles = movie_user_rating.columns
user_ids = movie_user_rating.index

u = 0
user_ratings = predicted_ratings[u, :]
#get movie indices sorted by predicted rating (highest first)
recommended_idx = np.argsort(user_ratings)[:: -1]

#map indices to moive titles
recommended_movies = movie_titles[recommended_idx]

#get predicted ratings in order
recommended_scores = user_ratings[recommended_idx]

already_rated = movie_user_rating.iloc[u,:]>0

recommended_idx = [i for i in recommended_idx if not already_rated[i]]

top_n = 10
recommended_idx_top = recommended_idx[:top_n]

# Output in the same style as MMR
print("Top movies from Matrix Factorization (ranked by predicted rating):")
for rank, idx in enumerate(recommended_idx_top, start=1):
    print(f"{rank}. {movie_titles[idx]} — Predicted rating: {user_ratings[idx]:.2f}")

