import pandas as pd
import numpy as np
from DataHandler import DataHandler


def user_ndcg(df, k, ground_truth_df=None):
    """
    Calculate NDCG@k for a single user's predictions.

    Parameters:
    -----------
    df : DataFrame
        User's predictions with columns: rating_pred, rating_true (or rating_gt)
    k : int
        Top-k cutoff
    ground_truth_df : DataFrame, optional
        Full ground truth for this user (needed for proper IDCG calculation)

    Returns:
    --------
    float : NDCG@k score (0.0 to 1.0)
    """
    # Sort by predicted rating and take top-k
    df_sorted = df.sort_values('rating_pred', ascending=False).head(k).copy()

    # Determine which column has the actual ratings
    if 'rating_true' in df_sorted.columns:
        relevance_col = 'rating_true'
    elif 'rating_gt' in df_sorted.columns:
        relevance_col = 'rating_gt'
    else:
        raise ValueError("No rating column found. Expected 'rating_true' or 'rating_gt'")

    # Use actual rating values as relevance
    df_sorted['relevance'] = df_sorted[relevance_col].fillna(0)

    # Calculate DCG: (2^relevance - 1) / log2(rank + 1)
    df_sorted['rank'] = np.arange(1, len(df_sorted) + 1)
    df_sorted['dcg'] = (2 ** df_sorted['relevance'] - 1) / np.log2(df_sorted['rank'] + 1)
    dcg = df_sorted['dcg'].sum()

    # Calculate IDCG using ALL ground truth items
    if ground_truth_df is not None and len(ground_truth_df) > 0:
        # Use full ground truth for ideal ranking
        ideal_relevances = ground_truth_df.sort_values('rating', ascending=False).head(k)['rating'].values
    else:
        # Fallback: use predicted items only
        ideal_relevances = df[relevance_col].fillna(0).sort_values(ascending=False).head(k).values

    # Calculate IDCG
    if len(ideal_relevances) > 0:
        ideal_ranks = np.arange(1, len(ideal_relevances) + 1)
        idcg = np.sum((2 ** ideal_relevances - 1) / np.log2(ideal_ranks + 1))
    else:
        idcg = 0.0

    # Normalize to get NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg





# XXXXXXXXXXXXXXXXX
# Test

# Define parameters
# threshold = 4.0
# k = 5
#
# # Initialize DataHandler
# data_handler = DataHandler()
#
# # Get top-k predictions (sorted by predicted rating)
# merged_topk = data_handler.get_topk_predictions(k, threshold)
#
# # Calculate NDCG@K per user
# per_user_ndcg = merged_topk.groupby("userId").apply(user_ndcg, k=k)
#
# # Macro average across all users
# macro_ndcg = per_user_ndcg.mean()
#
# print(f"\nNDCG@{k} per user:")
# print(per_user_ndcg)
# print(f"\nMacro NDCG@{k}: {macro_ndcg:.3f}")
#
# print(f"- Perfect ranking: NDCG = 1.0")
# print(f"- Random ranking: NDCG ≈ 0.3-0.5 (depending on data)")
# print(f"- Our models ranking quality: NDCG = {macro_ndcg:.3f} ({macro_ndcg:.1%} of perfect)")
