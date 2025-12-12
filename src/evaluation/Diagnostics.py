import pandas as pd
import os

def _print_data_diagnostics(file_path, file_label="Data", threshold=4.0, is_ground_truth=False):
    """
    Comprehensive diagnostics for either ground truth or predictions files.
    Works with both rating prediction and recommendation formats.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin1')

        print(f"\n{'=' * 60}")
        print(f"{file_label.upper()} DIAGNOSTIC")
        print(f"{'=' * 60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Auto-detect columns
        user_col = next((col for col in df.columns if 'user' in col.lower()), None)
        item_col = next((col for col in df.columns if 'item' in col.lower() or 'movie' in col.lower()), None)
        rating_col = next((col for col in df.columns if 'rating' in col.lower() and 'predict' not in col.lower()), None)
        pred_rating_col = next((col for col in df.columns if 'predicted' in col.lower() or 'pred' in col.lower()), None)

        if user_col:
            print(f"Unique users: {df[user_col].nunique()}")

            # For recommendation format with ranks
            if 'rank' in df.columns:
                rows_per_user = df.groupby(user_col).size()
                print(
                    f"Rows per user - Min: {rows_per_user.min()}, Max: {rows_per_user.max()}, Mean: {rows_per_user.mean():.1f}")
                print(f"Rank range: {df['rank'].min()} - {df['rank'].max()}")

                # Check rank sequence integrity
                if not all(df.groupby(user_col)['rank'].apply(lambda x: sorted(x) == list(range(1, len(x) + 1)))):
                    print("⚠️ WARNING: Ranks are not sequential 1..K for all users")
            else:
                # For rating prediction format
                rows_per_user = df.groupby(user_col).size()
                print(
                    f"Rows per user - Min: {rows_per_user.min()}, Max: {rows_per_user.max()}, Mean: {rows_per_user.mean():.1f}")

        if item_col:
            print(f"Unique items: {df[item_col].nunique()}")

        # Rating distribution (for ground truth or if true ratings exist)
        if is_ground_truth and rating_col:
            print(f"\nRating distribution:")
            print(df[rating_col].value_counts().sort_index().to_string())
            print(f"% ratings ≥ {threshold}: {(df[rating_col] >= threshold).mean():.1%}")

        # Predicted ratings statistics
        if pred_rating_col:
            print(f"\nPredicted rating statistics:")
            print(f"  Min: {df[pred_rating_col].min():.4f}")
            print(f"  Max: {df[pred_rating_col].max():.4f}")
            print(f"  Mean: {df[pred_rating_col].mean():.4f}")
            print(f"  Std: {df[pred_rating_col].std():.4f}")

            # Correlation if both true and predicted ratings exist
            if rating_col and rating_col != pred_rating_col:
                correlation = df[rating_col].corr(df[pred_rating_col])
                print(f"\nCorrelation between true and predicted ratings: {correlation:.4f}")
                if correlation > 0.95:
                    print("⚠️ WARNING: Very high correlation - possible data leakage")
                elif correlation < 0.3:
                    print("⚠️ WARNING: Low correlation - predictions may be random")
                else:
                    print("✓ Reasonable correlation")

        # Data quality
        print(f"\nMissing values:")
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(null_counts[null_counts > 0])
        else:
            print("  None")

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️ WARNING: Found {duplicates} duplicate rows")
        else:
            print(f"✓ No duplicate rows")

        # Sample data
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())

        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")