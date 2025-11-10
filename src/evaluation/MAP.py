import pandas as pd
import numpy as np
from DataHandler import DataHandler

#Doesnt work right
def average_precision_at_k(top_k_recs, relevant_set):
    # If the user has no relevant items, AP is not defined (or 0).
    # We will filter these users out before calling this function.
    if not relevant_set:
        return 0.0

    hits = 0
    sum_of_precisions = 0.0

    # Iterate through the top K recommendations
    for i, rec_item in enumerate(top_k_recs):
        k = i + 1  # Current rank (1-based)

        if rec_item in relevant_set:
            hits += 1
            precision_at_k = hits / k
            sum_of_precisions += precision_at_k

    # The denominator is the total number of relevant items in the ground truth
    num_relevant = len(relevant_set)

    if num_relevant == 0:
        return 0.0

    return sum_of_precisions / num_relevant


def calculate_map_at_k(predictions_df, ground_truth_df, k, threshold=4.0):
    """
    Calculates the Mean Average Precision (MAP) at K for all users.

    Args:
        predictions_df (pd.DataFrame): All predictions.
        ground_truth_df (pd.DataFrame): All ground truth ratings.
        k (int): The number of top items to consider.
        threshold (float): The rating threshold to be considered "relevant".

    Returns:
        float: The MAP@K score.
    """

    # 1. Get the set of relevant items for *all users*
    # We only calculate MAP for users who have at least one relevant item.
    relevant_df = ground_truth_df[ground_truth_df['rating'] >= threshold]
    relevant_sets = relevant_df.groupby('userId')['title'].apply(set)

    if relevant_sets.empty:
        print("Warning: No users have any relevant items at the given threshold.")
        print("MAP@K is 0.")
        return 0.0

    # Create a DataFrame of users who are part of the MAP calculation
    map_users_df = pd.DataFrame(relevant_sets).reset_index()

    # 2. Get the Top-K recommendation lists for *all users*
    pred_sorted = predictions_df.sort_values(
        ["userId", "rating"],
        ascending=[True, False]
    )
    top_k_df = pred_sorted.groupby("userId").head(k)
    top_k_lists = top_k_df.groupby('userId')['title'].apply(list)
    top_k_lists_df = pd.DataFrame(top_k_lists).reset_index()

    # 3. Join the two DataFrames
    # We use a 'left' join starting from the relevant users.
    # This ensures we only evaluate users who have relevant items.
    merged = pd.merge(map_users_df, top_k_lists_df, on='userId', how='left')

    # Handle users who have relevant items but got no recommendations
    # Their 'top_k_list' will be NaN. We replace it with an empty list.
    merged['top_k_list'] = merged['top_k_list'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # 4. Calculate AP for each user
    merged['AP'] = merged.apply(
        lambda row: average_precision_at_k(row['top_k_list'], row['title']),
        axis=1
    )

    # 5. Calculate MAP (the mean of all AP scores)
    map_score = merged['AP'].mean()

    return map_score, merged

#XXXXXXXXXXX
#Test

k = 5
threshold = 4.0

# Initialize DataHandler
try:
    data_handler = DataHandler()
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

all_predictions = data_handler.predictions
ground_truth_data = data_handler.ground_truth

print(f"Calculating MAP@{k} (threshold={threshold})...")

map_score, per_user_ap = calculate_map_at_k(
    all_predictions,
    ground_truth_data,
    k,
    threshold
)

print("\n--- Mean Average Precision (MAP) Results ---")
print(f"Calculated over {len(per_user_ap)} users with at least one relevant item.")
print(f"\nMAP@{k}: {map_score:.4f}")

print("\n--- Example AP@K per user ---")
print(per_user_ap.head())