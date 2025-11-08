import pandas as pd
import os

#for determining recall, precision and F1 of predictions. Precision is the number of correct outcomes divided by the sum of true and false predictions,
# Recall is true positive / (true posetives+false negatives). also called sensetivity
#F1 is the harmonic mean of precision and recall, a measure of the reliability of a model

# Load data
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ground_truth_path = os.path.join(base_path, "datasets", "ratings_test_titles2.csv")
predictions_path = os.path.join(base_path, "datasets/mmr_data", "test_predictions.csv")
ground_truth = pd.read_csv(ground_truth_path)
predictions = pd.read_csv(predictions_path)

# Merge datasets using a full OUTER merge
merged_full = pd.merge(predictions, ground_truth, on=["userId", "title"], how="outer", suffixes=('_pred', '_gt'))

# Handle cases where an item is in one dataset but not the other
# If an item wasn't rated (missing from ground_truth), assume rating is 0.
merged_full['rating_gt'] = merged_full['rating_gt'].fillna(0)

# If an item wasn't predicted (missing from predictions), assume predicted rating is 0.
merged_full['predicted_score'] = merged_full['rating_pred'].fillna(0)
merged_full['was_predicted'] = merged_full['rating_pred'].notna()  # Flag for predictions made

# Define relevance threshold
threshold = 4.0

# Create true/false columns that says if movie is relevant
merged_full["true_relevant"] = merged_full["rating_gt"] >= threshold  # ground truth relevance

merged_full["pred_relevant"] = merged_full["predicted_score"] >= threshold  # predicted relevance


# Per-user precision, recall, and F1
def user_precision_recall_f1_standard(df):
    #Only consider movies that the model made a prediction for when calculating TP/FP
    df_predicted = df[df['was_predicted']]

    #TP: predicted relevant and truly relevant
    tp = ((df_predicted["pred_relevant"]) & (df_predicted["true_relevant"])).sum()

    #FP: predicted relevant but not truly relevant
    fp = ((df_predicted["pred_relevant"]) & (~df_predicted["true_relevant"])).sum()

    #FN: not predicted relevant but truly relevant
    fn = ((~df["pred_relevant"]) & (df["true_relevant"])).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    #F1 Score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({"precision": precision, "recall": recall, "f1": f1})


per_user_standard = merged_full.groupby("userId").apply(user_precision_recall_f1_standard)

macro_precision = per_user_standard["precision"].mean()
macro_recall = per_user_standard["recall"].mean()
macro_f1 = per_user_standard["f1"].mean()

print("Per-user Standard Precision, Recall, and F1:")
print(per_user_standard)
print("\nMacro Standard Precision:", round(macro_precision, 3))
print("Macro Standard Recall:", round(macro_recall, 3))
print("Macro Standard F1:", round(macro_f1, 3))

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Top-K evaluation
k = 5

# Sort predictions by user and predicted rating descending
pred_sorted = predictions.sort_values(["userId", "rating"], ascending=[True, False])

# Keep top-k predictions per user
topk = pred_sorted.groupby("userId").head(k)

# Merge with ground truth to check relevance
merged_topk = pd.merge(topk, ground_truth, on=["userId", "title"], how="left")

# Handle cases where prediction are not in the ground truth
merged_topk["rating_y"] = merged_topk["rating_y"].fillna(0)

# For Precision@K, we only care if ground truth is relevant
merged_topk["true_relevant"] = merged_topk["rating_y"] >= threshold

# Precision@k per user: fraction of top-k that are relevant
precision_at_k = merged_topk.groupby("userId")["true_relevant"].mean()
macro_precision_at_k = precision_at_k.mean()

# Recall@k per user: fraction of all relevant items that appear in top-k
relevant_counts = ground_truth[ground_truth["rating"] >= threshold].groupby("userId")["title"].count()
tp_topk = merged_topk.groupby("userId")["true_relevant"].sum()
recall_at_k = (tp_topk / relevant_counts).fillna(0)
macro_recall_at_k = recall_at_k.mean()

# F1@k per user: harmonic mean of precision@k and recall@k
f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
f1_at_k = f1_at_k.fillna(0)  # Handle division by zero
macro_f1_at_k = f1_at_k.mean()

print(f"\nPrecision@{k} per user:")
print(precision_at_k)
print(f"\nMacro Precision@{k}: {macro_precision_at_k:.3f}")

print(f"\nRecall@{k} per user:")
print(recall_at_k)
print(f"\nMacro Recall@{k}: {macro_recall_at_k:.3f}")

print(f"\nF1@{k} per user:")
print(f1_at_k)
print(f"\nMacro F1@{k}: {macro_f1_at_k:.3f}")