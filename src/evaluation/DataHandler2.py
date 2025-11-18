import pandas as pd

# Container for recommendation data
class ProcessedData:
    def __init__(self, ground_truth, predictions, interactions, recommendations, full_interactions):
        self.ground_truth = ground_truth # user ratings
        self.predictions = predictions # model predictions
        self.interactions = interactions # gt in a specific format for Rectools
        self.recommendations = recommendations #
        self.full_interactions = full_interactions

# Load and transform data into RecTools format.
def load_and_process_data(ground_truth_path, predictions_path):
    # Load data
    gt = pd.read_csv(ground_truth_path)
    pred = pd.read_csv(predictions_path)

    # Convert IDs to strings
    id_columns = ["userId", "title"]
    for col in id_columns:
        if col in gt.columns:
            gt[col] = gt[col].astype(str)
        if col in pred.columns:
            pred[col] = pred[col].astype(str)

    # Convert to RecTools format
    ground_truth = _to_rectools_format(gt, is_ground_truth=True)
    predictions = _to_rectools_format(pred, is_ground_truth=False)

    # Validate
    _validate_columns(ground_truth, predictions)

    # Prepare structures
    interactions = _prepare_interactions(ground_truth)
    recommendations = _prepare_recommendations(predictions)

    return ProcessedData(
        ground_truth=ground_truth, # original GT
        predictions=predictions, # Predictions
        interactions=interactions, # GT filtered down to the 3 columns needed for metrics (UserId, ItemID, Weight)
        recommendations=recommendations, # Predictions with a rank column
        full_interactions=interactions.copy() #GT that shouldnt be touched
    )

# Convert DataFrame to RecTools standard format
def _to_rectools_format(df, is_ground_truth):
    df = df.copy()
    column_map = {
        "userId": "user_id",
        "title": "item_id"
    }

    # might have to change a bit if models use different names
    rating_col = "rating" if is_ground_truth else ("rating_pred" if "rating_pred" in df.columns else "rating")
    if rating_col in df.columns:
        column_map[rating_col] = "weight"

    return df.rename(columns=column_map)

# Validate required columns exist
def _validate_columns(gt, pred):
    required = ["user_id", "item_id", "weight"]
    for df, name in [(gt, "Ground Truth"), (pred, "Predictions")]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")


def _prepare_interactions(df):
    return df[["user_id", "item_id", "weight"]].copy()


def _prepare_recommendations(df):
    recos = df[["user_id", "item_id", "weight"]].copy()
    recos = recos.sort_values(["user_id", "weight"], ascending=[True, False])
    recos["rank"] = recos.groupby("user_id").cumcount() + 1 # add rank column sorted by user Id and descending
    return recos