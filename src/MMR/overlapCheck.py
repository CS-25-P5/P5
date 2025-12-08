import pandas as pd
import os

# def check_overlap(ground_truth_file, mf_test_file, user_col='userId', item_col='itemId'):
#     # Load datasets
#     ground_truth = pd.read_csv(ground_truth_file)
#     mf_test = pd.read_csv(mf_test_file)

#     # Ensure consistent types
#     ground_truth[user_col] = ground_truth[user_col].astype(int)
#     ground_truth[item_col] = ground_truth[item_col].astype(int)
#     mf_test[user_col] = mf_test[user_col].astype(int)
#     mf_test[item_col] = mf_test[item_col].astype(int)

#     # Find which users/items are missing from MF predictions
#     missing_users = set(ground_truth[user_col].unique()) - set(mf_test[user_col].unique())
#     missing_items = set(ground_truth[item_col].unique()) - set(mf_test[item_col].unique())
    
#     print(f"Users in ground-truth but missing in MF predictions: {len(missing_users)}")
#     print(f"Items in ground-truth but missing in MF predictions: {len(missing_items)}")
    
#     if len(missing_users) > 0:
#         print(f"Sample missing users: {list(missing_users)[:10]}")
#     if len(missing_items) > 0:
#         print(f"Sample missing items: {list(missing_items)[:10]}")

#     # Filter ground-truth to only users/items that exist in MF predictions
#     ground_truth_filtered = ground_truth[
#         ground_truth[user_col].isin(mf_test[user_col].unique()) &
#         ground_truth[item_col].isin(mf_test[item_col].unique())
#     ]
#     print(f"Ground-truth entries with predicted users/items only: {len(ground_truth_filtered)} rows")

#     # Merge to find overlap
#     overlap = pd.merge(ground_truth_filtered, mf_test, on=[user_col, item_col])

#     # Summary
#     summary = {
#         "total_ground_truth": len(ground_truth),
#         "total_ground_truth_filtered": len(ground_truth_filtered),
#         "total_mf_test": len(mf_test),
#         "overlap_count": len(overlap),
#         "overlap_percentage_of_filtered": len(overlap) / len(ground_truth_filtered) * 100 if len(ground_truth_filtered) > 0 else 0
#     }

#     # Print summary
#     print(f"Total ground-truth entries: {summary['total_ground_truth']}")
#     print(f"Ground-truth entries for predicted users/items: {summary['total_ground_truth_filtered']}")
#     print(f"Total MF test entries: {summary['total_mf_test']}")
#     print(f"Number of overlapping entries: {summary['overlap_count']}")
#     print(f"Percentage of ground-truth entries present in MF test: {summary['overlap_percentage_of_filtered']:.2f}%")

#     return summary, overlap

def check_overlap(ground_truth_path, mf_predictions_path):
    # Load datasets
    gt = pd.read_csv(ground_truth_path)
    mf_pred = pd.read_csv(mf_predictions_path)

    # Convert IDs to string to ensure consistent types
    gt['userId'] = gt['userId'].astype(str)
    gt['itemId'] = gt['itemId'].astype(str)
    mf_pred['userId'] = mf_pred['userId'].astype(str)
    mf_pred['itemId'] = mf_pred['itemId'].astype(str)

    print(f"Ground-truth entries: {len(gt)}")
    print(f"MF predictions entries: {len(mf_pred)}")

    # Determine users/items present in MF predictions
    pred_users = set(mf_pred['userId'])
    pred_items = set(mf_pred['itemId'])

    # Filter ground-truth to only users/items that MF can predict
    gt_aligned = gt[gt['userId'].isin(pred_users) & gt['itemId'].isin(pred_items)]

    print(f"Users in ground-truth but missing in MF predictions: {len(set(gt['userId']) - pred_users)}")
    print(f"Items in ground-truth but missing in MF predictions: {len(set(gt['itemId']) - pred_items)}")
    print(f"Ground-truth entries after alignment: {len(gt_aligned)}")

    # Merge aligned ground-truth with MF predictions to find overlaps
    merged = gt_aligned.merge(mf_pred, on=['userId', 'itemId'], how='inner', suffixes=('_gt', '_mf'))
    num_overlap = len(merged)

    print(f"Number of overlapping entries: {num_overlap}")
    if len(gt_aligned) > 0:
        print(f"Percentage of aligned ground-truth entries present in MF test: {num_overlap / len(gt_aligned) * 100:.2f}%")
    else:
        print("No overlapping users/items to compare.")

    # Optional: return the merged DataFrame for inspection
    return merged


def debug_user_item_overlap():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load all three files
    train = pd.read_csv(os.path.join(base_dir, "../datasets/mmr_data", "movies_ratings_10000_train.csv"))
    val = pd.read_csv(os.path.join(base_dir, "../datasets/mmr_data", "movies_ratings_10000_val.csv"))
    test = pd.read_csv(os.path.join(base_dir, "../datasets/mmr_data", "movies_ratings_10000_test.csv"))
    
    print("=" * 60)
    print("CHECKING USER/ITEM OVERLAP BETWEEN SPLITS")
    print("=" * 60)
    
    # Convert to strings
    for df in [train, val, test]:
        df['userId'] = df['userId'].astype(str)
        df['itemId'] = df['itemId'].astype(str)
    
    # Check users
    train_users = set(train['userId'])
    val_users = set(val['userId'])
    test_users = set(test['userId'])
    
    print(f"Train users: {len(train_users)}")
    print(f"Val users:   {len(val_users)}")
    print(f"Test users:  {len(test_users)}")
    
    print(f"\nUser overlap:")
    print(f"Train ∩ Val:   {len(train_users & val_users)}")
    print(f"Train ∩ Test:  {len(train_users & test_users)}")
    print(f"Val ∩ Test:    {len(val_users & test_users)}")
    print(f"All three:     {len(train_users & val_users & test_users)}")
    
    # Check items
    train_items = set(train['itemId'])
    test_items = set(test['itemId'])
    
    print(f"\nItem overlap:")
    print(f"Train ∩ Test:  {len(train_items & test_items)}")
    
    # Show sample users from each set
    print(f"\nSample train users: {sorted(train_users)[:10]}")
    print(f"Sample test users:  {sorted(test_users)[:10]}")
    
    # Check if it's a ".0" issue
    print(f"\nChecking for '.0' suffix issue:")
    print(f"Train user samples: {list(train_users)[:5]}")
    print(f"Test user samples:  {list(test_users)[:5]}")
    
    # Maybe users are floats in one, ints in another?
    print(f"\nData types:")
    print(f"Train userId type: {train['userId'].iloc[0]} (type: {type(train['userId'].iloc[0])})")
    print(f"Test userId type:  {test['userId'].iloc[0]} (type: {type(test['userId'].iloc[0])})")

debug_user_item_overlap()

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
ground_truth = os.path.join(base_dir, "../datasets/mmr_data", "movies_ratings_10000_test.csv")
#mf_test = os.path.join(base_dir, "../datasets/mmr_data/movies/2025-12-07_16-03-16/mf_test_10000_predictions.csv")
mf_test = "mf_rating_predictions.csv"

# mf_test = os.path.join(base_dir, "../evaluation/movie/mf_test_100000_predictions.csv")
movies_file = os.path.join(base_dir, "../datasets/MovieLens/movies.csv")

# summary, overlap_df = check_overlap(ground_truth, mf_test)
# print(overlap_df.head())

merged_df = check_overlap(ground_truth, mf_test)
print("\nSample overlapping entries:")
print(merged_df.head())
