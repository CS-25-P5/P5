import pandas as pd
import os

import warnings

# Suppress the specific FutureWarning from RecTools
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays.*"
)

# Container for recommendation data
class ProcessedData:
    def __init__(self, ground_truth, predictions, interactions, recommendations):
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.interactions = interactions
        self.recommendations = recommendations

# Load and transform data into RecTools format.
def load_and_process_data(ground_truth_path, predictions_path):
    # Load ground truth normally
    gt = pd.read_csv(ground_truth_path)

    # Clean predictions on-the-fly
    print(f"Cleaning predictions file: {predictions_path}")

    # Read file, skipping non-CSV lines
    valid_lines = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('userId') or stripped.startswith('user_id') or (
                    stripped and stripped[0].isdigit()):
                valid_lines.append(line)

    # Create a temporary clean file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp:
        temp.writelines(valid_lines)
        temp_path = temp.name

    # Load from cleaned temporary file
    pred = pd.read_csv(temp_path)
    os.unlink(temp_path)  # Delete temp file

    # Continue with your existing code...
    print(f"Predictions loaded: {len(pred)} rows")

    for col in ["userId", "movieId"]:
        if col in gt.columns:
            gt[col] = gt[col].astype(str).str.replace('\.0$', '', regex=True)
        if col in pred.columns:
            pred[col] = pred[col].astype(str).str.replace('\.0$', '', regex=True)

    print("✅ IDs normalized (stripped .0 decimals)")

    # Convert IDs to strings
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

    # ============================================
    # CRITICAL DEBUG: CHECK ACTUAL ID VALUES
    # ============================================
    print("\n" + "=" * 60)
    print("CHECKING FIRST 10 USER-ITEM PAIRS AFTER PROCESSING")
    print("=" * 60)

    # Sample from interactions (ground truth)
    print("\nInteractions (ground truth) sample:")
    print(interactions.head(3).to_string(index=False))

    # Sample from recommendations (predictions)
    print("\nRecommendations (predictions) sample:")
    print(recommendations.head(3).to_string(index=False))

    # Check if ANY pair matches
    interaction_set = set(zip(interactions['user_id'], interactions['item_id']))
    recommendation_set = set(zip(recommendations['user_id'], recommendations['item_id']))

    print(f"\nTotal interaction pairs: {len(interaction_set)}")
    print(f"Total recommendation pairs: {len(recommendation_set)}")

    # Check 3 specific pairs from ground truth
    test_pairs = list(interaction_set)[:3]
    for i, (u, m) in enumerate(test_pairs, 1):
        exists = (u, m) in recommendation_set
        print(f"  Pair {i}: ({u}, {m}) exists in predictions? {exists}")

    common = interaction_set & recommendation_set
    print(f"\nCommon pairs: {len(common)}")
    if len(common) > 0:
        print("First 3 common pairs:", list(common)[:3])
    else:
        print("❌ ZERO OVERLAP FOUND!")
        print("\nInteraction IDs (first 3):", list(interaction_set)[:3])
        print("Recommendation IDs (first 3):", list(recommendation_set)[:3])

    # Check dtypes
    print(f"\nInteraction dtypes:\n{interactions.dtypes}")
    print(f"\nRecommendation dtypes:\n{recommendations.dtypes}")

    # Check if IDs have quotes or spaces
    sample_interaction = interactions.iloc[0]
    sample_recommendation = recommendations.iloc[0]
    print(
        f"\nSample interaction ID repr: user_id='{repr(sample_interaction['user_id'])}', item_id='{repr(sample_interaction['item_id'])}'")
    print(
        f"Sample recommendation ID repr: user_id='{repr(sample_recommendation['user_id'])}', item_id='{repr(sample_recommendation['item_id'])}'")
    # ============================================

    return ProcessedData(
        ground_truth=ground_truth,
        predictions=predictions,
        interactions=interactions,
        recommendations=recommendations
    )


# Convert DataFrame to RecTools standard format
def _to_rectools_format(df, is_ground_truth):
    df = df.copy()

    if not is_ground_truth:
        # Remove any actual rating column included in prediction files
        df = df.drop(columns=["rating"], errors="ignore")

    column_map = {
        "userId": "user_id",
    }


    # Map item identifier to item_id
    if "itemId" in df.columns:
        column_map["itemId"] = "item_id"
    elif "movieId" in df.columns:
        column_map["movieId"] = "item_id"
    elif "title" in df.columns:
        column_map["title"] = "item_id"

    # Handle rating column
    if is_ground_truth:
        rating_col = "rating"
    else:
        possible_rating_cols = ["rating_pred", "predictedRating", "prediction", "predicted_rating", "rating","mf_score", "test_predicted_rating"]
        rating_col = next((col for col in possible_rating_cols if col in df.columns), None)

    if rating_col and rating_col in df.columns:
        column_map[rating_col] = "weight"

    df = df.rename(columns=column_map)

    if not is_ground_truth:
        # Drop any other rating-like columns
        df = df[["user_id", "item_id", "weight"]]

    # Force convert ID columns to string after renaming
    if "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(str)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)

    # Force weight to float
    if "weight" in df.columns:
        df["weight"] = df["weight"].astype(float)

    return df


# Validate required columns exist
def _validate_columns(gt, pred):
    required = ["user_id", "item_id", "weight"]

    #print("\nValidating required columns:")
    for df, name in [(gt, "Ground Truth"), (pred, "Predictions")]:
        missing = [col for col in required if col not in df.columns]
        present = [col for col in required if col in df.columns]
        #print(f"{name}: present={present}, missing={missing}")
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        #else:
            #print(f"{name} has all required columns")


def _prepare_interactions(df):
    #print(f"\nPreparing interactions - selecting columns: {['user_id', 'item_id', 'weight']}")
    interactions = df[["user_id", "item_id", "weight"]].copy()

    #Force consistent dtypes
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["weight"] = interactions["weight"].astype(float)

    #print(f"Interaction dtypes:\n{interactions.dtypes}")
    return interactions


def _prepare_recommendations(df):
    #print(f"\nPreparing recommendations - selecting columns: {['user_id', 'item_id', 'weight']}")
    recos = df[["user_id", "item_id", "weight"]].copy()

    # Force consistent dtypes
    recos["user_id"] = recos["user_id"].astype(str)
    recos["item_id"] = recos["item_id"].astype(str)
    recos["weight"] = recos["weight"].astype(float)

    # If rank already exists, don't add it again
    if "rank" not in recos.columns:
        recos = recos.sort_values(["user_id", "weight"], ascending=[True, False])
        recos["rank"] = recos.groupby("user_id").cumcount() + 1
        #print(f"Added 'rank' column to recommendations.")
    #else:
        #print(f"'rank' column already exists, using provided ranks.")

    #print(f"Recommendations dtypes:\n{recos.dtypes}")
    #print(f"Sample:\n{recos.head()}")
    return recos