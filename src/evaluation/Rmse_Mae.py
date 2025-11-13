import pandas as pd
import numpy as np
from DataHandler import DataHandler

#metric for how close predicted ratings are to actual ratings, only does it for predictions where a ground truth exists
# https://dev.to/mondal_sabbha/understanding-mae-mse-and-rmse-key-metrics-in-machine-learning-4la2
def calculate_accuracy_metrics(predictions_df, ground_truth_df):
    # Merge dataframes
    # Uses an inner merge because for MAE/RMSE, we only care about items where we have both a prediction and a true rating
    merged_df = pd.merge(
        predictions_df,
        ground_truth_df,
        on=["userId", "title"],
        how="inner",
        suffixes=('_pred', '_gt')
    )

    # 'rating_pred' is the 'rating' column from predictions_df
    # 'rating_true' is the 'rating' column from ground_truth_df
    diff = merged_df['rating_pred'] - merged_df['rating_gt']

    # Calculate MAE (Mean Absolute Error), average of model prediction errors
    mae = diff.abs().mean()

    # Calculate RMSE (Root Mean Square Error), penalizes large errors
    rmse = np.sqrt((diff ** 2).mean())

    count = len(merged_df)

    return mae, rmse, count


# XXXXXXX
# XTest

# Initialize DataHandler
data_handler = DataHandler()

# Get the raw data
all_predictions = data_handler.predictions
ground_truth_data = data_handler.ground_truth

# Calculate metrics
print("MAE and RMSE")

mae, rmse, count = calculate_accuracy_metrics(
    all_predictions,
    ground_truth_data
)

print("\nAccuracy Results")
print(f"Calculated over {count} common ratings.")
print(f"Mean Absolute Error (MAE):   {mae:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

print("MAE: On average, the model's rating prediction is off by ~{:.4f} stars.".format(mae))
