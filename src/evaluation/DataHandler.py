import pandas as pd
import os


class DataHandler:
    def __init__(self, ground_truth_path, predictions_path):
        # Store the provided paths
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path

        # Load data
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.predictions = pd.read_csv(self.predictions_path)

    def get_merged_data_for_standard_metrics(self, threshold=4.0):
        # Merge datasets using a full OUTER merge
        merged_full = pd.merge(
            self.predictions,
            self.ground_truth,
            on=["userId", "title"],
            how="outer",
            suffixes=('_pred', '_gt')
        )

        # Handle cases where an item is in one dataset but not the other
        # If an item wasn't rated (missing from ground_truth), assume rating is 0.
        merged_full['rating_gt'] = merged_full['rating_gt'].fillna(0)

        # If an item wasn't predicted (missing from predictions), assume predicted rating is 0.
        merged_full['predicted_score'] = merged_full['rating_pred'].fillna(0)
        merged_full['was_predicted'] = merged_full['rating_pred'].notna()  # Flag for predictions made

        # Create true/false columns that says if movie is relevant
        merged_full["true_relevant"] = merged_full["rating_gt"] >= threshold
        merged_full["pred_relevant"] = merged_full["predicted_score"] >= threshold

        return merged_full

    def get_topk_predictions(self, k, threshold=4.0):
        # Sort predictions by user and predicted rating descending
        pred_sorted = self.predictions.sort_values(
            ["userId", "rating"],
            ascending=[True, False]
        )

        # Keep top-k predictions per user
        topk = pred_sorted.groupby("userId").head(k)

        # Merge with ground truth to check relevance
        merged_topk = pd.merge(
            topk,
            self.ground_truth,
            on=["userId", "title"],
            how="left",
            suffixes=('_pred', '_gt')  # FIX: Explicit suffixes for clarity
        )

        # Handle cases where predictions are not in the ground truth
        merged_topk['rating_gt'] = merged_topk['rating_gt'].fillna(0)

        # For Precision@K, we only care if ground truth is relevant
        merged_topk["true_relevant"] = merged_topk["rating_gt"] >= threshold

        return merged_topk

    def get_relevant_counts_per_user(self, threshold=4.0):
        relevant_counts = self.ground_truth[
            self.ground_truth["rating"] >= threshold
            ].groupby("userId")["title"].count()

        return relevant_counts


# XXXXXXXXXXXXXXXXX Test

# Initialize handler
dh = DataHandler(ground_truth_path=r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv",
    predictions_path=r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv")
#fix paths when you get back
# Sample data flows
#rating_pred: Predicted rating (may contain NaN)
# rating_gt: Ground truth rating (0 if missing)
#predicted_score: Final prediction (0-filled)
#was_predicted: Boolean flag

#
# print(" full outer merge (standard metrics)")
# merged_full = dh.get_merged_data_for_standard_metrics(threshold=4.0)
# print("Columns:", merged_full.columns.tolist())
#
# # rating_pred: Predicted rating (from predictions CSV)
# # rating_gt: Ground truth rating (0 if not in ground truth
# # true_relevant: Binary relevance flag
#
# print("\n\n TOP-K MERGE ")
# merged_topk = dh.get_topk_predictions(k=5, threshold=4.0)
# print("Columns:", merged_topk.columns.tolist())
