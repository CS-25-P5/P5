import pandas as pd
import os

def check_overlap(ground_truth_file, mf_test_file, user_col='userId', item_col='itemId'):
    # Load datasets
    ground_truth = pd.read_csv(ground_truth_file)
    mf_test = pd.read_csv(mf_test_file)

    # Ensure consistent types
    ground_truth[user_col] = ground_truth[user_col].astype(int)
    ground_truth[item_col] = ground_truth[item_col].astype(int)
    mf_test[user_col] = mf_test[user_col].astype(int)
    mf_test[item_col] = mf_test[item_col].astype(int)

    # Find which users/items are missing from MF predictions
    missing_users = set(ground_truth[user_col].unique()) - set(mf_test[user_col].unique())
    missing_items = set(ground_truth[item_col].unique()) - set(mf_test[item_col].unique())
    
    print(f"Users in ground-truth but missing in MF predictions: {len(missing_users)}")
    print(f"Items in ground-truth but missing in MF predictions: {len(missing_items)}")
    
    if len(missing_users) > 0:
        print(f"Sample missing users: {list(missing_users)[:10]}")
    if len(missing_items) > 0:
        print(f"Sample missing items: {list(missing_items)[:10]}")

    # Filter ground-truth to only users/items that exist in MF predictions
    ground_truth_filtered = ground_truth[
        ground_truth[user_col].isin(mf_test[user_col].unique()) &
        ground_truth[item_col].isin(mf_test[item_col].unique())
    ]
    print(f"Ground-truth entries with predicted users/items only: {len(ground_truth_filtered)} rows")

    # Merge to find overlap
    overlap = pd.merge(ground_truth_filtered, mf_test, on=[user_col, item_col])

    # Summary
    summary = {
        "total_ground_truth": len(ground_truth),
        "total_ground_truth_filtered": len(ground_truth_filtered),
        "total_mf_test": len(mf_test),
        "overlap_count": len(overlap),
        "overlap_percentage_of_filtered": len(overlap) / len(ground_truth_filtered) * 100 if len(ground_truth_filtered) > 0 else 0
    }

    # Print summary
    print(f"Total ground-truth entries: {summary['total_ground_truth']}")
    print(f"Ground-truth entries for predicted users/items: {summary['total_ground_truth_filtered']}")
    print(f"Total MF test entries: {summary['total_mf_test']}")
    print(f"Number of overlapping entries: {summary['overlap_count']}")
    print(f"Percentage of ground-truth entries present in MF test: {summary['overlap_percentage_of_filtered']:.2f}%")

    return overlap


# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
ground_truth = os.path.join(base_dir, "../datasets/mmr_data", "books_ratings_100000_test.csv")
mf_test = os.path.join(base_dir, "../datasets/mmr_data/books/2025-12-09_05-08-32/mf_test_100000_predictions.csv")
#mf_test = "mf_rating_predictions.csv"

# mf_test = os.path.join(base_dir, "../evaluation/movie/mf_test_100000_predictions.csv")


# overlap_df = check_overlap(ground_truth, mf_test)
# print(overlap_df.head())


# mf_test = os.path.join(base_dir, "../datasets/mmr_data/books/2025-12-09_05-08-32/mf_test_100000_predictions.csv")

# result  = os.path.join(base_dir, "../datasets/mmr_data/books/2025-12-09_05-08-32/mf_test_100000_predictions_ranked.csv")
# df = pd.read_csv(result)

# df["rank"] = df.groupby("userId").cumcount() + 1

# # Reorder columns: put rank after userId
# cols = df.columns.tolist()

# # Ensure "rank" comes right after "userId"
# cols.insert(1, cols.pop(cols.index("rank")))

# df = df[cols]


# df.to_csv(result, index=False)

# Remove rank if it exists
# if "rank" in df.columns:
#     df = df.drop(columns=["rank"])

# df.to_csv(result, index=False)