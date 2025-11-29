import pandas as pd


def convert_file_to_include_both_ids(input_path, mapping_path, output_path, is_test_data=False, preserve_order=False):
    """
    Convert a file to include both 'itemId' and 'title' columns

    Args:
        preserve_order: If True, keeps original row order instead of sorting
    """
    print(f"\n=== Processing {'TEST DATA' if is_test_data else 'PREDICTIONS'} ===")
    print(f"Input: {input_path}")

    # Load data
    df = pd.read_csv(input_path)
    original_rows = len(df)  # Track original row count

    mapping = pd.read_csv(mapping_path)

    # Clean column names
    df.columns = df.columns.str.strip()
    mapping.columns = mapping.columns.str.strip()

    print(f"Input columns: {df.columns.tolist()}")
    print(f"Input shape: {df.shape}")

    # Find columns
    item_id_col = next((col for col in ['itemId', 'item_id', 'movieId', 'movie_id'] if col in mapping.columns), None)
    title_col = 'title' if 'title' in mapping.columns else None

    if item_id_col is None or title_col is None:
        raise ValueError(f"Mapping file must contain item ID and title columns. Found: {mapping.columns.tolist()}")

    # Ensure string types
    mapping[title_col] = mapping[title_col].astype(str).str.strip()
    mapping[item_id_col] = mapping[item_id_col].astype(str).str.strip()

    if is_test_data:
        # Test data has itemId, need to add title
        print("→ Test data detected - adding 'title' column")

        # Find item ID column in input
        test_item_col = next((col for col in ['itemId', 'item_id', 'movieId', 'movie_id'] if col in df.columns), None)
        if test_item_col is None:
            raise ValueError(f"Test data must contain an item ID column. Found: {df.columns.tolist()}")

        # Ensure string type for matching
        df[test_item_col] = df[test_item_col].astype(str).str.strip()

        # Merge to get titles
        converted = df.merge(mapping[[item_id_col, title_col]],
                             left_on=test_item_col,
                             right_on=item_id_col,
                             how='left')

        # CRITICAL: Only drop the duplicate column if it's different
        if item_id_col != test_item_col:
            converted = converted.drop(columns=[item_id_col])

        # Build final dataframe
        final_df = pd.DataFrame({
            'userId': converted['userId'],
            'itemId': converted[test_item_col],
            'rating': converted['rating'],
            'title': converted[title_col]
        })

    else:
        # Predictions have title, need to add itemId
        print("→ Predictions detected - adding 'itemId' column")

        df['title'] = df['title'].astype(str).str.strip()

        # Merge to get itemIds
        converted = df.merge(mapping[[item_id_col, title_col]],
                             left_on='title',
                             right_on=title_col,
                             how='left')

        # Build final dataframe
        final_df = pd.DataFrame({
            'userId': converted['userId'],
            'title': converted['title'],
            'rating_pred': converted['mf_score'],
            'itemId': converted[item_id_col],
            'genres': converted['genres']
        })

    # Check for missing mappings
    missing_col = 'title' if is_test_data else 'itemId'
    missing = final_df[missing_col].isnull().sum()
    if missing > 0:
        print(f"⚠️  WARNING: {missing} items could not be matched")
        # Debug: show some examples
        missing_examples = final_df[final_df[missing_col].isnull()]['title' if not is_test_data else 'itemId'].unique()[
                           :5]
        print(f"   Example missing: {missing_examples}")

    # Ensure numeric types
    final_df['userId'] = pd.to_numeric(final_df['userId'], errors='coerce')
    if 'rating' in final_df.columns:
        final_df['rating'] = pd.to_numeric(final_df['rating'], errors='coerce')
    if 'rating_pred' in final_df.columns:
        final_df['rating_pred'] = pd.to_numeric(final_df['rating_pred'], errors='coerce')

    # Drop rows that couldn't be matched
    final_df = final_df.dropna(subset=[missing_col])

    print(f"Original rows: {original_rows}, After dropping missing: {len(final_df)}")

    # SORTING: Only sort if preserve_order is False
    if preserve_order:
        print("→ Preserving original order (no sorting)")
    else:
        sort_col = 'rating_pred' if not is_test_data else 'rating'
        final_df = final_df.sort_values(['userId', sort_col], ascending=[True, False])
        print("→ Sorted by userId and rating")

    # Save
    print(f"Saving to: {output_path}")
    print(f"Final shape: {final_df.shape}")
    print(f"Final columns: {final_df.columns.tolist()}")
    print(f"Sample output (first 5 rows):\n{final_df.head()}\n")

    final_df.to_csv(output_path, index=False)
    print("✓ Conversion complete!\n")


# ============================================================================
# USAGE - SET preserve_order=True TO KEEP ORIGINAL ORDER
# ============================================================================
if __name__ == "__main__":
    MAPPING_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
    BASE_DIR = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie"

    # Convert predictions (preserve original order)
    PREDICTIONS_PATH = fr"{BASE_DIR}\mmr_train_cosine_test_recommendations.csv"
    PREDICTIONS_OUTPUT = fr"{BASE_DIR}\mmr_train_cosine_test_recommendations_with_both_ids.csv"
    convert_file_to_include_both_ids(PREDICTIONS_PATH, MAPPING_PATH, PREDICTIONS_OUTPUT,
                                     is_test_data=False, preserve_order=True)

    # Convert test data (preserve original order)
    TEST_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movies_ratings_100000_test.csv"
    TEST_OUTPUT = fr"{BASE_DIR}\test_data_with_both_ids.csv"
    convert_file_to_include_both_ids(TEST_PATH, MAPPING_PATH, TEST_OUTPUT,
                                     is_test_data=True, preserve_order=True)