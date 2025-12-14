import pandas as pd
import os


def add_genres_to_predictions(predictions_path, movies_path, output_path=None):
    """
    Creates a new CSV file with userId, itemId, predictedRating, and genres
    by merging predictions with movies data based on itemId.

    Args:
        predictions_path: Path to predictions CSV (with userId, itemId, predictedRating)
        movies_path: Path to movies CSV (with itemId, genres)
        output_path: Optional output path. If None, appends '_with_genres' to original filename
    """
    print(f"Loading predictions from {predictions_path}")
    predictions = pd.read_csv(predictions_path)

    print(f"Loading movies from {movies_path}")
    movies = pd.read_csv(movies_path)

    # Keep only needed columns
    predictions = predictions[['userId', 'itemId', 'predictedRating']].copy()
    movies = movies[['itemId', 'genres']].copy()

    # Ensure itemId is string type for consistent matching
    predictions['itemId'] = predictions['itemId'].astype(str).str.strip()
    movies['itemId'] = movies['itemId'].astype(str).str.strip()

    print(f"Merging {len(predictions)} predictions with {len(movies)} movies...")

    # Merge to add genres
    result = predictions.merge(movies, on='itemId', how='left')

    # Check for unmatched items
    unmatched = result['genres'].isna().sum()
    if unmatched > 0:
        print(f"Warning: {unmatched} items could not find matching genres")

    # Generate output path if not provided
    if output_path is None:
        directory = os.path.dirname(predictions_path)
        base_name = os.path.basename(predictions_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(directory, f"{name_without_ext}_with_genres.csv")

    # Save result
    result.to_csv(output_path, index=False)
    print(f"\n✅ Successfully created: {output_path}")
    print(f"   Columns: {list(result.columns)}")
    print(f"   Rows: {len(result)}")

    return output_path


if __name__ == "__main__":
    # === CONFIGURE THESE PATHS ===
    PREDICTIONS_FILE = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\mf_test_100000_top_10.csv"
    MOVIES_FILE = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"

    add_genres_to_predictions(PREDICTIONS_FILE, MOVIES_FILE)