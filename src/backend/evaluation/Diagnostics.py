import pandas as pd
import os


def _print_data_diagnostics(file_path, file_label="Data", threshold=4.0, is_ground_truth=False, ground_truth_path=None):
    """
    Comprehensive diagnostics for either ground truth or predictions files.
    Works with both rating prediction and recommendation formats.

    Args:
        file_path: Path to the file being diagnosed
        file_label: Label for display purposes
        threshold: Rating threshold for ground truth analysis
        is_ground_truth: Whether this is a ground truth file
        ground_truth_path: Path to ground truth file (optional, for overlap checking with predictions)
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

        # OVERLAP CHECKING WITH GROUND TRUTH
        if not is_ground_truth and ground_truth_path and os.path.exists(ground_truth_path):
            try:
                print(f"\n{'-' * 40}")
                print("OVERLAP ANALYSIS WITH GROUND TRUTH")
                print(f"{'-' * 40}")

                # Load ground truth
                gt_df = pd.read_csv(ground_truth_path, encoding='latin1')

                # Normalize column names for comparison
                gt_user_col = next((col for col in gt_df.columns if 'user' in col.lower()), None)
                gt_item_col = next((col for col in gt_df.columns if 'item' in col.lower() or 'movie' in col.lower()),
                                   None)

                # Detect prediction columns
                pred_user_col = user_col
                pred_item_col = item_col

                if gt_user_col and gt_item_col and pred_user_col and pred_item_col:
                    # Create sets of user-item pairs
                    gt_pairs = set(zip(gt_df[gt_user_col].astype(str), gt_df[gt_item_col].astype(str)))
                    pred_pairs = set(zip(df[pred_user_col].astype(str), df[pred_item_col].astype(str)))

                    # Calculate overlap
                    common_pairs = gt_pairs & pred_pairs
                    overlap_count = len(common_pairs)
                    gt_total = len(gt_pairs)
                    pred_total = len(pred_pairs)

                    print(f"Ground truth pairs: {gt_total:,}")
                    print(f"Prediction pairs: {pred_total:,}")
                    print(f"Common pairs: {overlap_count:,}")

                    if pred_total > 0:
                        overlap_percentage = (overlap_count / pred_total) * 100
                        print(f"Overlap: {overlap_count}/{pred_total} ({overlap_percentage:.1f}% of predictions)")

                    if gt_total > 0:
                        coverage_percentage = (overlap_count / gt_total) * 100
                        print(f"Coverage: {overlap_count}/{gt_total} ({coverage_percentage:.1f}% of ground truth)")

                    # Show sample overlapping pairs
                    if overlap_count > 0:
                        print(f"\nSample overlapping pairs (first 5):")
                        sample_pairs = list(common_pairs)[:5]
                        for i, (user, item) in enumerate(sample_pairs, 1):
                            print(f"  {i}. User: {user}, Item: {item}")

                    # Warning for very low overlap
                    if pred_total > 0 and overlap_count / pred_total < 0.1:
                        print("⚠️ WARNING: Very low overlap (< 10%) - predictions may not align with ground truth!")

            except Exception as e:
                print(f"⚠️ Could not perform overlap analysis: {e}")

        # Sample data
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())

        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
