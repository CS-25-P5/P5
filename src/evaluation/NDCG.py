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
threshold = 4.0
k = 5

# Initialize DataHandler
data_handler = DataHandler()

# Get top-k predictions (sorted by predicted rating)
merged_topk = data_handler.get_topk_predictions(k, threshold)

# Calculate NDCG@K per user
per_user_ndcg = merged_topk.groupby("userId").apply(user_ndcg, k=k)

# Macro average across all users
macro_ndcg = per_user_ndcg.mean()

print(f"\nNDCG@{k} per user:")
print(per_user_ndcg)
print(f"\nMacro NDCG@{k}: {macro_ndcg:.3f}")

print(f"- Perfect ranking: NDCG = 1.0")
print(f"- Random ranking: NDCG ≈ 0.3-0.5 (depending on data)")
print(f"- Our models ranking quality: NDCG = {macro_ndcg:.3f} ({macro_ndcg:.1%} of perfect)")

import pandas as pd
import numpy as np
from DataHandler import DataHandler

# comparable across users with different numbers of relevant items.

#ndcgv (Normalized Discounted Cumulative Gain), evaluates the quality of ranking
# takes into concideration where a recommended item is in the list
# NDCG ranges from 0 to 1,
# DCG (Discounted Cumulative Gain) penalizes relevant items that appear lower in the ranking
# using a logarithmic discount factor. IDCG is the theoretical maximum DCG if items were
# perfectly sorted by true relevance. NDCG = DCG / IDCG provides a normalized score
def user_ndcg(df, k, relevant_counts):
    user_id = df.name  # Get userId from groupby

    # Take top-k predictions for this user
    topk_df = df.head(k).copy()

    # Relevance scores (binary: 0 or 1)
    topk_df['relevance'] = topk_df['true_relevant'].astype(int)

    # Calculate rank positions (1-indexed)
    topk_df['rank'] = np.arange(1, len(topk_df) + 1)

    # DCG: Discounted Cumulative Gain (relevance / log2(rank + 1))
    topk_df['dcg'] = topk_df['relevance'] / np.log2(topk_df['rank'] + 1)
    dcg = topk_df['dcg'].sum()

    # IDCG: Ideal DCG based on actual number of relevant items in ground truth
    # Get the number of relevant items this user has in ground truth
    num_relevant = relevant_counts.get(user_id, 0)

    # Ideal scenario: min(k, num_relevant) relevant items at top positions
    ideal_relevant_count = min(k, num_relevant)

    # Calculate IDCG with ideal_relevant_count items all being relevant (value=1)
    if ideal_relevant_count > 0:
        ideal_ranks = np.arange(1, ideal_relevant_count + 1)
        idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
    else:
        idcg = 0.0

    # Normalize to get NDCG (handle division by zero for users with no relevant items)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg


# XXXXXXXXXXXXXXXXX Test

# Define parameters
threshold = 4.0
k = 5

# Initialize DataHandler
data_handler = DataHandler()

# Get top-k predictions (sorted by predicted rating)
merged_topk = data_handler.get_topk_predictions(k, threshold)

# Get the count of relevant items per user from ground truth
relevant_counts = data_handler.get_relevant_counts_per_user(threshold)

print("Relevant items per user in ground truth:")
print(relevant_counts)
print()

# Calculate NDCG@K per user
per_user_ndcg = merged_topk.groupby("userId").apply(
    user_ndcg,
    k=k,
    relevant_counts=relevant_counts
)

# Macro average across all users
macro_ndcg = per_user_ndcg.mean()

print(f"\nNDCG@{k} per user:")
print(per_user_ndcg)
print(f"\nMacro NDCG@{k}: {macro_ndcg:.3f}")

print(f"\n--- Interpretation ---")
print(f"- Perfect ranking: NDCG = 1.0")
print(f"- Random ranking: NDCG ≈ 0.3-0.5 (depending on data)")
print(f"- Your model's ranking quality: NDCG = {macro_ndcg:.3f} ({macro_ndcg:.1%} of perfect)")

# Show detailed breakdown for each user
print(f"\n--- Detailed Breakdown ---")
for user_id in sorted(per_user_ndcg.index):
    user_data = merged_topk[merged_topk['userId'] == user_id].head(k)
    num_relevant_in_topk = user_data['true_relevant'].sum()
    num_relevant_total = relevant_counts.get(user_id, 0)

    print(f"User {user_id}:")
    print(f"  - Relevant items in ground truth: {num_relevant_total}")
    print(f"  - Relevant items found in top-{k}: {num_relevant_in_topk}")
    print(f"  - NDCG@{k}: {per_user_ndcg[user_id]:.3f}")
    print()