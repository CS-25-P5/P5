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
