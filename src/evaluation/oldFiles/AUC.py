import pandas as pd
import numpy as np
from DataHandler import DataHandler


# Calculates user's AUC (Area Under ROC Curve), which measures the probability that 
# a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant item.
# 
# AUC ranges from 0.0 to 1.0, where:
# - 1.0 = Perfect ranking (all relevant items scored higher than irrelevant)
# - 0.5 = Random ranking (no discrimination)
# - 0.0 = Inverse ranking (all irrelevant scored higher than relevant)
# 
# Formula: AUC = (Σ rank_positives - m*(m+1)/2) / (m*n)
# where m = #positives, n = #negatives, rank_positives = ranks in scored list
def user_auc(df, threshold):
    """
    Calculate AUC for a single user's predictions.

    Args:
        df: User's dataframe with columns: 'predicted_score', 'true_relevant', 'was_predicted'
        threshold: Not directly used but kept for consistency with other metrics

    Returns:
        float: AUC score (0.0 to 1.0)
    """

    # Filter to items that were actually predicted (need scores to rank)
    df_predicted = df[df['was_predicted']].copy()

    # Need at least 2 items and both classes to calculate meaningful AUC
    if len(df_predicted) < 2:
        return 0.0

    # Create binary relevance labels
    df_predicted['relevance'] = df_predicted['true_relevant'].astype(int)

    # Check for edge cases: all positive or all negative
    pos_count = df_predicted['relevance'].sum()
    neg_count = len(df_predicted) - pos_count

    if pos_count == 0 or neg_count == 0:
        return 0.0

    # Sort by predicted score (descending) to simulate ranking
    df_predicted = df_predicted.sort_values('predicted_score', ascending=False)

    # Get ranks (1-indexed) of positive items
    df_predicted['rank'] = range(1, len(df_predicted) + 1)
    pos_ranks = df_predicted[df_predicted['relevance'] == 1]['rank'].values

    # Calculate AUC using the Wilcoxon-Mann-Whitney statistic
    sum_pos_ranks = pos_ranks.sum()
    auc = (sum_pos_ranks - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)

    return auc


# XXXXXXXXXXXXXXXXX Test
# Define parameters
threshold = 4.0

# Initialize DataHandler
data_handler = DataHandler()

# AUC Evaluation
print("AUC (Area Under ROC Curve) Metrics")
print("=" * 50)

# Get merged data with standard metrics
merged_full = data_handler.get_merged_data_for_standard_metrics(threshold)

# Calculate AUC per user using groupby-apply pattern
per_user_auc = merged_full.groupby("userId").apply(user_auc, threshold=threshold)

# Macro-average across all users (ignore NaN from edge cases)
macro_auc = per_user_auc.mean()

print(f"\nAUC per user:")
print(per_user_auc.round(3))
print(f"\nMacro AUC: {macro_auc:.3f}")

print(f"\nInterpretation:")
print(f"- AUC = 1.000: Perfect ranking (all relevant items scored higher)")
print(f"- AUC = 0.500: Random ranking (no discriminative power)")
print(f"- AUC = 0.000: Perfectly inverse ranking")
print(f"- Your model: AUC = {macro_auc:.3f}")
print(f"- Lift over random: {((macro_auc - 0.5) / 0.5 * 100):+.1f}%")