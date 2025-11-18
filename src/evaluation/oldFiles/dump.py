#XXXXXXXXXXXX Metrics repo XXXXXXXXXXXXXXXX

def calculate_all_metrics(data_handler, threshold=4.0, k=5, item_features=None):
    results = {}
    all_predictions = data_handler.predictions
    ground_truth_data = data_handler.ground_truth

    print("Calculating RMSE and MAE")
    mae, rmse, _ = calculate_accuracy_metrics(all_predictions, ground_truth_data)
    results["RMSE"] = rmse
    results["MAE"] = mae

    print(f"Calculating Precision@{k}, Recall@{k}, F1@{k}")
    merged_topk = data_handler.get_topk_predictions(k, threshold)
    relevant_counts = data_handler.get_relevant_counts_per_user(threshold)

    precision_at_k = merged_topk.groupby("userId")["true_relevant"].mean()
    tp_topk = merged_topk.groupby("userId")["true_relevant"].sum()
    recall_at_k = (tp_topk / relevant_counts).fillna(0)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    results["Precision"] = precision_at_k.mean()
    results["Recall"] = recall_at_k.mean()
    results["F1"] = f1_at_k.fillna(0).mean()

    print(f"Calculating NDCG@{k}")
    per_user_ndcg = merged_topk.groupby("userId").apply(user_ndcg, k=k, include_groups=False)
    results["NDCG"] = per_user_ndcg.mean()

    print(f"Calculating MAP@{k}")
    merged_full = data_handler.get_merged_data_for_standard_metrics(threshold)
    per_user_ap = merged_full.groupby("userId").apply(
        user_average_precision, threshold=threshold, relevant_counts=relevant_counts, include_groups=False
    )
    results["MAP"] = per_user_ap.mean()

    print(f"Calculating MRR@{k}")
    merged_topk_sorted = merged_topk.sort_values(["userId", "rating_pred"], ascending=[True, False])

    def user_mrr(group):
        for rank, is_relevant in enumerate(group["true_relevant"], 1):
            if is_relevant:
                return 1.0 / rank
        return 0.0

    per_user_mrr = merged_topk_sorted.groupby("userId").apply(user_mrr, include_groups=False)
    results["MRR"] = per_user_mrr.mean()

    print(f"Calculating Coverage@{k}")
    coverage_percent, _, _ = calculate_topK_item_coverage(all_predictions, ground_truth_data, k)
    results["Coverage"] = coverage_percent

    print(f"Calculating ILD@{k}")
    if item_features is not None and not item_features.empty:
            per_user_ild = merged_topk.groupby("userId").apply(
                user_intra_list_diversity, item_features_df=item_features, k=k, include_groups=False
            )
            results["ILD"] = per_user_ild.dropna().mean()
    else:
        print("Warning: No item features provided, skipping ILD")
        results["ILD"] = np.nan

    print("Calculating Reverse Gini (Gini Index)")
    gini_score = calculate_gini_index(all_predictions)
    results['Reverse Gini'] = gini_score

    return results

def display_metrics_table(metrics_dict, source_name="Model"):
    metric_order = ["RMSE", "MAE", "Precision", "Recall", "F1", "NDCG", "MAP", "MRR", "Coverage", "ILD", "Reverse Gini"]
    ordered_results = {metric: metrics_dict.get(metric, np.nan) for metric in metric_order}

    df = pd.DataFrame([ordered_results], index=[source_name])
    df_display = df.round(4)

    print("Recommendation evaluation results")
    print(df_display.to_string())

    return df

#map

import pandas as pd
import numpy as np
from DataHandler import DataHandler


def user_average_precision(df, threshold, relevant_counts):
    user_id = df.name  # Get userId from groupby

    # Filter to items the model actually predicted
    df_predicted = df[df['was_predicted']].copy()

    # Handle edge case: no predictions
    if len(df_predicted) == 0:
        return 0.0

    # Sort by predicted score descending (most confident first)
    df_predicted = df_predicted.sort_values('predicted_score', ascending=False)

    # Add rank positions (1-indexed)
    df_predicted['rank'] = range(1, len(df_predicted) + 1)

    # Create binary relevance scores (1 = relevant, 0 = not)
    df_predicted['relevance'] = df_predicted['true_relevant'].astype(int)

    # Calculate cumulative sum of relevant items found so far
    df_predicted['cum_relevant'] = df_predicted['relevance'].cumsum()

    # Calculate Precision@K at each rank position
    df_predicted['precision_at_k'] = df_predicted['cum_relevant'] / df_predicted['rank']

    # Get Precision@K values only at positions where item is relevant
    relevant_precisions = df_predicted[df_predicted['relevance'] == 1]['precision_at_k']

    # Get total number of relevant items for this user from ground truth
    total_relevant = relevant_counts.get(user_id, 0)

    # Handle case: no relevant items in ground truth
    if total_relevant == 0:
        return 0.0

    # Average Precision = sum of precisions at relevant positions / total relevant items
    # NOT the mean of precisions (that would ignore items we didn't find)
    ap = relevant_precisions.sum() / total_relevant

    return ap


# XXXXXXXXXXXXXXXXX Test
# # Define parameters
# threshold = 4.0
#
# # Initialize DataHandler
# data_handler = DataHandler()
#
# print("MAP (Mean Average Precision) Metrics")
# print("=" * 60)
#
# # Get merged data with standard metrics
# merged_full = data_handler.get_merged_data_for_standard_metrics(threshold)
#
# # Get the count of relevant items per user from ground truth
# relevant_counts = data_handler.get_relevant_counts_per_user(threshold)
#
# print("Relevant items per user in ground truth:")
# print(relevant_counts)
# print()
#
# # Calculate Average Precision per user
# per_user_ap = merged_full.groupby("userId").apply(
#     user_average_precision,
#     threshold=threshold,
#     relevant_counts=relevant_counts
# )
#
# # MAP = mean of Average Precision across all users
# map_score = per_user_ap.mean()
#
# print(f"\nAverage Precision per user:")
# print(per_user_ap)
# print(f"\nMAP (Mean Average Precision): {map_score:.3f}")
#
# print(f"\n--- Interpretation ---")
# print(f"- MAP = 1.0: All relevant items ranked before any irrelevant items")
# print(f"- MAP = 0.0: No relevant items found in predictions")
# print(f"- Your model: MAP = {map_score:.3f} ({map_score:.1%} of perfect)")
