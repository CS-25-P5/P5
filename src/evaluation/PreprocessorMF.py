import pandas as pd
import os


def align_predictions_with_mapping(predictions_path, mapping_path, output_path=None):
    """
    Convert predictions that use titles to use itemIds based on a mapping file.
    """
    print(f"Loading predictions from {predictions_path}")
    predictions = pd.read_csv(predictions_path)

    print(f"Loading mapping from {mapping_path}")
    mapping = pd.read_csv(mapping_path)

    # Ensure string types for matching
    if 'title' in mapping.columns:
        mapping['title'] = mapping['title'].astype(str).str.strip()
    if 'itemId' in mapping.columns:
        mapping['itemId'] = mapping['itemId'].astype(str).str.strip()
    if 'title' in predictions.columns:
        predictions['title'] = predictions['title'].astype(str).str.strip()

    # Validate required columns
    if 'title' not in mapping.columns or 'itemId' not in mapping.columns:
        raise ValueError(f"Mapping file must contain 'itemId' and 'title' columns. Found: {list(mapping.columns)}")
    if 'title' not in predictions.columns:
        raise ValueError(f"Predictions file must contain 'title' column. Found: {list(predictions.columns)}")

    print(f"Converting {len(predictions)} predictions from title to itemId...")

    # Merge predictions with mapping to get itemId
    merged = predictions.merge(
        mapping[['itemId', 'title']],
        on='title',
        how='left'
    )

    # Check for unmatched titles
    unmatched = merged[merged['itemId'].isna()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} predictions could not be matched to itemId")
        print("Example unmatched titles:")
        for title in unmatched['title'].head().tolist():
            print(f"  - '{title}'")

        # Drop unmatched predictions
        merged = merged.dropna(subset=['itemId'])
        print(f"Dropped {len(unmatched)} unmatched predictions")

    # Build the aligned predictions DataFrame
    aligned_predictions = pd.DataFrame()
    aligned_predictions['userId'] = merged['userId']
    aligned_predictions['itemId'] = merged['itemId']

    # Copy rating column (try multiple possible names)
    rating_cols = ['predictedRating', 'rating_pred', 'prediction', 'predicted_rating', 'mf_score', 'rating']
    rating_col_found = None
    for col in rating_cols:
        if col in merged.columns:
            aligned_predictions['rating_pred'] = merged[col]
            rating_col_found = col
            break

    if rating_col_found is None:
        raise ValueError(f"No rating column found in predictions. Tried: {rating_cols}")

    # If there's a rank column, preserve it
    if 'rank' in merged.columns:
        aligned_predictions['rank'] = merged['rank']

    # Add other columns that might be useful
    for col in merged.columns:
        if col not in ['userId', 'itemId', 'title'] + rating_cols:
            aligned_predictions[col] = merged[col]

    print(f"Successfully converted {len(aligned_predictions)} predictions to itemId format")

    # Save aligned predictions
    if output_path is None:
        directory = os.path.dirname(predictions_path)
        base_name = os.path.basename(predictions_path)
        output_path = os.path.join(directory, f"ALIGNED_{base_name}")

    aligned_predictions.to_csv(output_path, index=False)
    print(f"Saved aligned predictions to {output_path}")

    return output_path


if __name__ == "__main__":
    # === CONFIGURE THESE PATHS ===
    MAPPING_FILE = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
    PREDICTIONS_FILE = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\movies\mf_train_predictions_movies.csv"

    align_predictions_with_mapping(PREDICTIONS_FILE, MAPPING_FILE)