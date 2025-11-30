import pandas as pd
import os


def align_predictions_with_mapping(predictions_path, mapping_path, output_path=None):
    """
    Convert predictions that use titles to use itemIds based on a mapping file.
    Saves the aligned file and returns the output path.
    """
    print("=" * 70)
    print(f"📂 Loading predictions: {os.path.abspath(predictions_path)}")
    print(f"📂 Loading mapping:     {os.path.abspath(mapping_path)}")

    # Load data
    predictions = pd.read_csv(predictions_path)
    mapping = pd.read_csv(mapping_path)

    # Debug: Show first few rows
    print("\n📊 Predictions preview:")
    print(predictions.head())
    print("\n📊 Mapping preview:")
    print(mapping.head())

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

    print(f"\n🔄 Converting {len(predictions)} predictions from title to itemId...")

    # Merge predictions with mapping to get itemId
    merged = predictions.merge(
        mapping[['itemId', 'title']],
        on='title',
        how='left'
    )

    # Debug: Show merge results
    print(f"\n📊 After merge - total rows: {len(merged)}")
    print(f"📊 Successfully matched: {len(merged) - merged['itemId'].isna().sum()}")
    print(f"📊 Failed to match: {merged['itemId'].isna().sum()}")

    # Check for unmatched titles
    unmatched = merged[merged['itemId'].isna()]
    if not unmatched.empty:
        print(f"\n⚠️  Warning: {len(unmatched)} predictions could not be matched to itemId")
        print("   Example unmatched titles:")
        for title in unmatched['title'].head().tolist():
            print(f"     - '{title}'")

        # Drop unmatched predictions
        merged = merged.dropna(subset=['itemId'])
        print(f"   Dropped {len(unmatched)} unmatched predictions")

    # Build the aligned predictions DataFrame
    aligned_predictions = pd.DataFrame()
    aligned_predictions['userId'] = merged['userId']
    aligned_predictions['itemId'] = merged['itemId']

    # Copy rating column (try multiple possible names)
    rating_cols = ['predictedRating', 'rating_pred', 'prediction', 'predicted_rating', 'rating']
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

    # Add other columns that might be useful (like genres)
    for col in merged.columns:
        if col not in ['userId', 'itemId', 'title'] + rating_cols:
            aligned_predictions[col] = merged[col]

    print(f"\n✅ Successfully converted {len(aligned_predictions)} predictions to itemId format")

    # Show preview of final output
    print("\n📊 Aligned predictions preview:")
    print(aligned_predictions.head())

    # Determine output path
    if output_path is None:
        # Default: save in same directory as original predictions
        directory = os.path.dirname(predictions_path)
        base_name = os.path.basename(predictions_path)
        output_path = os.path.join(directory, f"aligned_{base_name}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    aligned_predictions.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {os.path.abspath(output_path)}")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    # === CONFIGURE THESE FULL PATHS ===
    MAPPING_FILE = r"/datasets/MovieLens/movies.csv"
    PREDICTIONS_FILE = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\ALIGNED_mf_train_predictions.csv"

    # Run the alignment
    try:
        result_path = align_predictions_with_mapping(
            predictions_path=PREDICTIONS_FILE,
            mapping_path=MAPPING_FILE
        )
        print(f"\n🎉 SUCCESS! Aligned file created at:\n{result_path}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()