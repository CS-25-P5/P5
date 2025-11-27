import pandas as pd

import warnings

# Suppress the specific FutureWarning from RecTools
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays.*"
)

# Or suppress ALL FutureWarnings from RecTools (broader)
warnings.filterwarnings("ignore", module="rectools.*")

# Container for recommendation data
class ProcessedData:
    def __init__(self, ground_truth, predictions, interactions, recommendations, full_interactions):
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.interactions = interactions
        self.recommendations = recommendations
        self.full_interactions = full_interactions

# Load and transform data into RecTools format.
def load_and_process_data(ground_truth_path, predictions_path):
    print("Loading raw data")

    # Load data
    gt = pd.read_csv(ground_truth_path)
    pred = pd.read_csv(predictions_path)

    print(f"\nGround truth columns: {list(gt.columns)}")
    print(f"Predictions columns: {list(pred.columns)}")

    # Convert IDs to strings
    print("Converting IDs to strings")

    # Support both itemId and title
    id_columns = ["userId", "itemId", "title"]
    for col in id_columns:
        if col in gt.columns:
            gt[col] = gt[col].astype(str)
            print(f"Converted ground_truth['{col}'] to string")
        if col in pred.columns:
            pred[col] = pred[col].astype(str)
            print(f"Converted predictions['{col}'] to string")

    # Convert to RecTools format
    print("Converting to RecTools format")

    ground_truth = _to_rectools_format(gt, is_ground_truth=True)
    predictions = _to_rectools_format(pred, is_ground_truth=False)

    print(f"Ground truth columns after format: {list(ground_truth.columns)}")
    print(f"Predictions columns after format: {list(predictions.columns)}")

    # Validate
    print("Validating columns")
    _validate_columns(ground_truth, predictions)

    # Prepare structures
    print("Preparing interaction structures")

    interactions = _prepare_interactions(ground_truth)
    recommendations = _prepare_recommendations(predictions)

    print(f"Interactions columns: {list(interactions.columns)}")
    print(f"Recommendations columns: {list(recommendations.columns)}")

    return ProcessedData(
        ground_truth=ground_truth,
        predictions=predictions,
        interactions=interactions,
        recommendations=recommendations,
        full_interactions=interactions.copy()
    )


# Convert DataFrame to RecTools standard format
def _to_rectools_format(df, is_ground_truth):
    df = df.copy()
    original_columns = list(df.columns)

    column_map = {
        "userId": "user_id",
    }

    # Map item identifier to item_id (handle both itemId and title)
    if "itemId" in df.columns:
        column_map["itemId"] = "item_id"
        print(f"Found 'itemId' column, will map to 'item_id'")
    elif "movieId" in df.columns:
        column_map["movieId"] = "item_id"
        print(f"Found 'movieId' column, will map to 'item_id'")
    elif "title" in df.columns:
        column_map["title"] = "item_id"
        print(f"Found 'title' column, will map to 'item_id'")
    else:
        print("Warning no item identifier column found (expected 'itemId', 'movieId', or 'title')")

    # Handle rating column
    if is_ground_truth:
        rating_col = "rating"
        print(f"Ground truth: looking for rating column 'rating'")
    else:
        # For predictions, try multiple possible rating column names
        possible_rating_cols = ["rating_pred", "predictedRating", "prediction", "predicted_rating", "rating","mf_score"]
        rating_col = next((col for col in possible_rating_cols if col in df.columns), None)
        print(f"Predictions: looking for rating column. Found: {rating_col}")

    if rating_col and rating_col in df.columns:
        column_map[rating_col] = "weight"
        print(f"Will map '{rating_col}' to 'weight'")
    else:
        print(
            f"No rating column found. Tried: {possible_rating_cols if not is_ground_truth else ['rating']}")

    print(f"\nColumn mapping: {column_map}")
    df = df.rename(columns=column_map)

    new_columns = list(df.columns)
    print(f"Renamed columns: {original_columns} → {new_columns}")

    # force weight to float
    if "weight" in df.columns:
        df["weight"] = df["weight"].astype(float)
        print("Converted 'weight' column to float")

    return df


# Validate required columns exist
def _validate_columns(gt, pred):
    required = ["user_id", "item_id", "weight"]

    print("\nValidating required columns:")
    for df, name in [(gt, "Ground Truth"), (pred, "Predictions")]:
        missing = [col for col in required if col not in df.columns]
        present = [col for col in required if col in df.columns]
        print(f"{name}: present={present}, missing={missing}")
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        else:
            print(f"{name} has all required columns")

def _prepare_interactions(df):
    print(f"\nPreparing interactions - selecting columns: {['user_id', 'item_id', 'weight']}")
    return df[["user_id", "item_id", "weight"]].copy()


def _prepare_recommendations(df):
    print(f"\nPreparing recommendations - selecting columns: {['user_id', 'item_id', 'weight']}")
    recos = df[["user_id", "item_id", "weight"]].copy()
    recos = recos.sort_values(["user_id", "weight"], ascending=[True, False])
    recos["rank"] = recos.groupby("user_id").cumcount() + 1
    print(f"Added 'rank' column to recommendations. Sample:\n{recos.head()}")
    return recos