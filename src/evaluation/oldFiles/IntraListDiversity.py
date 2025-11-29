import pandas as pd
import numpy as np
from DataHandler import DataHandler


# Calculates Intra-List Diversity (ILD) for a user's Top-K recommendations.
# ILD measures the average pairwise dissimilarity between items based on their features.
# Higher values (closer to 1.0) indicate more diverse recommendations across categories/genres.
#
# Formula: ILD = (2 / (k*(k-1))) * Σ distance(item_i, item_j) for all i < j
# where distance is cosine distance between item feature vectors (e.g., genre indicators).
def user_intra_list_diversity(df, item_features_df, k):
    # Get top-k item titles for this user (already sorted by rating in get_topk_predictions)
    topk_items = df.head(k)['title'].values

    # Need at least 2 items to calculate diversity
    if len(topk_items) < 2:
        return 0.0

    # Extract feature vectors for these items
    try:
        item_vectors = item_features_df.loc[topk_items].values
    except KeyError:
        # Items not found in feature matrix
        return np.nan

    # Calculate pairwise cosine distances (0=same, 1=different)
    from sklearn.metrics.pairwise import cosine_distances
    distance_matrix = cosine_distances(item_vectors)

    # Extract upper triangle (i < j) to avoid duplicates and self-comparisons
    upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    # Average pairwise distance
    ild = upper_triangle.mean()

    return ild


# XXXXXXXXXXXXXXXXX Test
# Define parameters
# k = 5
#
# # Initialize DataHandler
# data_handler = DataHandler()
#
# # Get top-k predictions (sorted by rating)
# merged_topk = data_handler.get_topk_predictions(k, threshold=4.0)
#
# # CREATE ITEM FEATURE MATRIX
# # Production: Load from CSV with items as rows and binary feature columns
# # Example CSV: title,genre_SciFi,genre_Animation,genre_Drama,genre_Action
# item_features = pd.DataFrame({
#     'title': ['The Matrix', 'Toy Story', 'Inception', 'Joker', 'Interstellar'],
#     'Sci-Fi': [1, 0, 1, 0, 1],
#     'Animation': [0, 1, 0, 0, 0],
#     'Drama': [0, 0, 0, 1, 0],
#     'Action': [1, 0, 1, 0, 1],
# }).set_index('title')
#
# print(f"Intra-List Diversity@{k} Metrics")
# print("=" * 50)
#
# # Calculate ILD per user using groupby-apply
# per_user_ild = merged_topk.groupby("userId").apply(user_intra_list_diversity, item_features_df=item_features, k=k)
#
# # Clean NaN values (users with missing item features)
# per_user_ild = per_user_ild.dropna()
#
# # Macro-average across all users
# macro_ild = per_user_ild.mean()
#
# print(f"\nILD@{k} per user:")
# print(per_user_ild)
# print(f"\nMacro Intra-List Diversity@{k}: {macro_ild:.3f}")
#
# print(f"\nInterpretation:")
# print(f"- Range: 0.0 (no diversity) to 1.0 (maximum diversity)")
# print(f"- Low (<0.30): Items are very similar (e.g., all same genre)")
# print(f"- Medium (0.30-0.60): Moderate diversity")
# print(f"- High (>0.60): Items span different feature sets")
# print(f"- Your model: {macro_ild:.3f} ({macro_ild:.1%} diversity)")


# Simple category diversity alternative (no sklearn required)
def user_category_diversity(df, k, category_col='category'):
    """Simpler: proportion of unique categories in Top-K (0-1 scale)"""
    topk_categories = df.head(k)[category_col].dropna().values
    if len(topk_categories) == 0:
        return 0.0
    return len(set(topk_categories)) / len(topk_categories)

# Add categories if available for simple version
# merged_topk['category'] = merged_topk['title'].map(category_dict)
# per_user_cat_div = merged_topk.groupby("userId").apply(user_category_diversity, k=k)