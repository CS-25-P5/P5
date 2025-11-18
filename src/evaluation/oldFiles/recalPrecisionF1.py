import pandas as pd
from DataHandler import DataHandler


# For determining recall, precision and F1 of predictions. Precision is the number of correct outcomes
# divided by the sum of true and false predictions. Recall is true positive / (true positives + false negatives),
# also called sensitivity. F1 is the harmonic mean of precision and recall, a measure of the reliability of a model.
# https://medium.com/@abhishekjainindore24/a-comprehensive-guide-to-performance-metrics-in-machine-learning-4ae5bd8208ce
def user_precision_recall_f1_standard(df):
    # Only consider movies that the model made a prediction for when calculating TP/FP
    df_predicted = df[df['was_predicted']]

    # TP: predicted relevant and truly relevant
    tp = ((df_predicted["pred_relevant"]) & (df_predicted["true_relevant"])).sum()

    # FP: predicted relevant but not truly relevant
    fp = ((df_predicted["pred_relevant"]) & (~df_predicted["true_relevant"])).sum()

    # FN: not predicted relevant but truly relevant
    fn = ((~df["pred_relevant"]) & (df["true_relevant"])).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({"precision": precision, "recall": recall, "f1": f1})


# XXXXXXXXXXXXXXXXX Test
# Define parameters
# threshold = 4.0
# k = 5
#
# # Initialize DataHandler
# data_handler = DataHandler()
#
# # Standard Precision, Recall, and F1
# print("Standard Metrics")
# merged_full = data_handler.get_merged_data_for_standard_metrics(threshold)
# per_user_standard = merged_full.groupby("userId").apply(user_precision_recall_f1_standard)
#
# macro_precision = per_user_standard["precision"].mean()
# macro_recall = per_user_standard["recall"].mean()
# macro_f1 = per_user_standard["f1"].mean()
#
# print("\nPer-user Standard Precision, Recall, and F1:")
# print(per_user_standard)
# print("\nMacro Standard Precision:", round(macro_precision, 3))
# print("Macro Standard Recall:", round(macro_recall, 3))
# print("Macro Standard F1:", round(macro_f1, 3))
#
# #Top-K Evaluation
# print(f"Top-{k} Metrics")
#
# merged_topk = data_handler.get_topk_predictions(k, threshold)
# relevant_counts = data_handler.get_relevant_counts_per_user(threshold)
#
# # Precision@k per user: fraction of top-k that are relevant
# precision_at_k = merged_topk.groupby("userId")["true_relevant"].mean()
# macro_precision_at_k = precision_at_k.mean()
#
# # Recall@k per user: fraction of all relevant items that appear in top-k
# tp_topk = merged_topk.groupby("userId")["true_relevant"].sum()
# recall_at_k = (tp_topk / relevant_counts).fillna(0)
# macro_recall_at_k = recall_at_k.mean()
#
# # F1@k per user: harmonic mean of precision@k and recall@k
# f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
# f1_at_k = f1_at_k.fillna(0)  # Handle division by zero
# macro_f1_at_k = f1_at_k.mean()
#
# print(f"\nPrecision@{k} per user:")
# print(precision_at_k)
# print(f"\nMacro Precision@{k}: {macro_precision_at_k:.3f}")
#
# print(f"\nRecall@{k} per user:")
# print(recall_at_k)
# print(f"\nMacro Recall@{k}: {macro_recall_at_k:.3f}")
#
# print(f"\nF1@{k} per user:")
# print(f1_at_k)
# print(f"\nMacro F1@{k}: {macro_f1_at_k:.3f}")