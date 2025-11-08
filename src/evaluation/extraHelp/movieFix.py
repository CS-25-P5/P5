import pandas as pd
import os

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_path = os.path.join(base_path, "datasets", "ratings_test.csv")
movies_path = os.path.join(base_path, "datasets", "movies.csv")
output_path = os.path.join(base_path, "datasets", "ratings_test_titles.csv")

# Load files
test = pd.read_csv(test_path)
movies = pd.read_csv(movies_path)

# Merge to add titles
test_with_titles = test.merge(movies[['movieId', 'title']], on='movieId', how='left')

# Rename 'title' to 'movie' to match MF predictions
#test_with_titles.rename(columns={'title': 'movie'}, inplace=True)

# Keep only necessary columns (userId, movie, rating, timestamp)
test_with_titles = test_with_titles[['userId', 'title', 'rating', 'timestamp']]

# Save as new CSV
test_with_titles.to_csv(output_path, index=False)

print(f"Saved test ratings with movie titles to: {output_path}")
