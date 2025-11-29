import pandas as pd


def check_data_overlap(predictions_path, ground_truth_path):
    """
    Quick diagnostic to check if predictions and ground truth have overlapping data
    """
    print("=" * 60)
    print("DATA OVERLAP DIAGNOSTIC")
    print("=" * 60)

    # Load data
    pred = pd.read_csv(predictions_path)
    gt = pd.read_csv(ground_truth_path)

    print(f"\nPredictions file: {predictions_path}")
    print(f"Ground truth file: {ground_truth_path}")

    print(f"\nPredictions columns: {pred.columns.tolist()}")
    print(f"Ground truth columns: {gt.columns.tolist()}")

    print(f"\nPredictions shape: {pred.shape}")
    print(f"Ground truth shape: {gt.shape}")

    # Check for itemId and title columns
    print(f"\nPredictions has itemId: {'itemId' in pred.columns}")
    print(f"Predictions has title: {'title' in pred.columns}")
    print(f"Ground truth has itemId: {'itemId' in gt.columns}")
    print(f"Ground truth has title: {'title' in gt.columns}")

    # Convert to strings for comparison
    if 'itemId' in pred.columns and 'itemId' in gt.columns:
        pred_items = pred['itemId'].astype(str).str.strip().unique()
        gt_items = gt['itemId'].astype(str).str.strip().unique()
        print(f"\nUnique items in predictions: {len(pred_items)}")
        print(f"Unique items in ground truth: {len(gt_items)}")

        common_items = set(pred_items) & set(gt_items)
        print(f"Common items: {len(common_items)}")
        if len(common_items) < 10:
            print(f"Common items: {sorted(list(common_items))[:10]}")

    if 'userId' in pred.columns and 'userId' in gt.columns:
        pred_users = pred['userId'].astype(str).str.strip().unique()
        gt_users = gt['userId'].astype(str).str.strip().unique()
        print(f"\nUnique users in predictions: {len(pred_users)}")
        print(f"Unique users in ground truth: {len(gt_users)}")

        common_users = set(pred_users) & set(gt_users)
        print(f"Common users: {len(common_users)}")
        if len(common_users) < 10:
            print(f"Common users: {sorted(list(common_users))[:10]}")

    # Check for exact user-item pair matches
    if 'itemId' in pred.columns and 'itemId' in gt.columns:
        pred_tuples = set(zip(pred['userId'].astype(str).str.strip(), pred['itemId'].astype(str).str.strip()))
        gt_tuples = set(zip(gt['userId'].astype(str).str.strip(), gt['itemId'].astype(str).str.strip()))

        common_pairs = pred_tuples & gt_tuples
        print(f"\nCommon user-item pairs: {len(common_pairs)}")

        if len(common_pairs) == 0:
            print("\n❌ NO MATCHING PAIRS FOUND!")
            print("\nFirst few prediction pairs:")
            for i, pair in enumerate(list(pred_tuples)[:5]):
                print(f"  {pair}")

            print("\nFirst few ground truth pairs:")
            for i, pair in enumerate(list(gt_tuples)[:5]):
                print(f"  {pair}")
        else:
            print("\n✓ Found overlapping data!")
            return True

    return False


if __name__ == "__main__":
    # Configuration
    PRED_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\mf_test_predictions_with_both_ids.csv"
    GT_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\test_data_with_both_ids.csv"

    check_data_overlap(PRED_PATH, GT_PATH)