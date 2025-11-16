import pandas as pd
import numpy as np
from DataHandler import DataHandler



def user_ndcg(df, k):
    # Take top-k predictions for this user
    topk_df = df.head(k).copy()

    # Relevance scores (binary: 0 or 1)
    topk_df['relevance'] = topk_df['true_relevant'].astype(int)

    # Calculate rank positions (1-indexed)
    topk_df['rank'] = np.arange(1, len(topk_df) + 1)

    # DCG: Discounted Cumulative Gain (relevance / log2(rank + 1))
    topk_df['dcg'] = topk_df['relevance'] / np.log2(topk_df['rank'] + 1)
    dcg = topk_df['dcg'].sum()

    # IDCG: Ideal DCG (perfect ranking by true relevance)
    ideal_df = df.sort_values('true_relevant', ascending=False).head(len(topk_df))
    ideal_df['relevance'] = ideal_df['true_relevant'].astype(int)
    ideal_df['rank'] = np.arange(1, len(ideal_df) + 1)
    ideal_df['idcg'] = ideal_df['relevance'] / np.log2(ideal_df['rank'] + 1)
    idcg = ideal_df['idcg'].sum()

    # Normalize to get NDCG (handle division by zero for users with no relevant items)
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
