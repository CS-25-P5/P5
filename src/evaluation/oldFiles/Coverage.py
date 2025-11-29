import pandas as pd
from DataHandler import DataHandler

# Coverage is a metric that measures how much of our item catalog is recommended by our system
# https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/
def calculate_topK_item_coverage(predictions_df, catalog_df, k):
    #  Get the set of all unique items in the catalog
    catalog_items = set(catalog_df['title'].unique())
    total_catalog_size = len(catalog_items)

    # Get the set of all unique items in the Top-K recommendations
    # Sort predictions by user and rating (descending)
    pred_sorted = predictions_df.sort_values(
        ["userId", "rating"],
        ascending=[True, False]
    )

    # Get the top-k predictions for each user
    topk_recs = pred_sorted.groupby("userId").head(k)

    # Find the set of unique items recommended
    recommended_items = set(topk_recs['title'].unique())
    total_recommended_size = len(recommended_items)

    # Calculate coverage
    coverage = (total_recommended_size / total_catalog_size)

    coverage_percent = coverage * 100

    return coverage_percent, total_recommended_size, total_catalog_size


def calculate_overall_coverage(predictions_df, catalog_df):
    # Get the set of all unique items in the catalog
    catalog_items = set(catalog_df['title'].unique())
    total_catalog_size = len(catalog_items)

    # Get the set of all unique items the model made a prediction for
    predicted_items = set(predictions_df['title'].unique())
    total_predicted_size = len(predicted_items)

    # Calculate coverage
    coverage = (total_predicted_size / total_catalog_size)
    coverage_percent = coverage * 100

    return coverage_percent, total_predicted_size, total_catalog_size

# XXXXXXXXXXXXXXXXXXXX
# Test

# Define parameters
# k = 5  # Match the k from your other script
#
# # Initialize DataHandler
# data_handler = DataHandler()
#
# # Get the raw data
# all_predictions = data_handler.predictions
# ground_truth_data = data_handler.ground_truth
#
# # Calculate coverage
# print(f"Calculating Item Coverage @ K={k}...")
#
# coverage_percent, num_recommended, num_catalog = calculate_topK_item_coverage(
#     all_predictions,
#     ground_truth_data,
#     k
# )
#
# print("\nTopK Coverage Results")
# print(f"Total items in catalog (from ground truth): {num_catalog}")
# print(f"Unique items in Top-{k} recommendations: {num_recommended}")
# print(f"Item Coverage @ {k}: {coverage_percent:.2f}%")
#
# # Calculate Overall Coverage
# overall_coverage_percent, num_predicted, num_catalog_overall = calculate_overall_coverage(
#     all_predictions,
#     ground_truth_data
# )
# print("")
# print("Overall Coverage Results")
# print(f"Total items in catalog (from ground truth): {num_catalog_overall}")
# print(f"Unique items with *any* prediction:        {num_predicted}")
# print(f"Overall Item Coverage:                      {overall_coverage_percent:.2f}%")
#
#
