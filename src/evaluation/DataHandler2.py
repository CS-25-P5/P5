import pandas as pd
import os
import tempfile
import warnings

# Suppress RecTools FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays.*"
)


class ProcessedData:
    def __init__(self, ground_truth, predictions, interactions, recommendations):
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.interactions = interactions
        self.recommendations = recommendations


def load_and_process_data(ground_truth_path, predictions_path, dataset_type="movies", verbose=True):
    """
    Load and process data for both MovieLens and GoodBooks datasets.

    Args:
        ground_truth_path: Path to ground truth CSV
        predictions_path: Path to predictions CSV
        dataset_type: "movies" or "books" - determines cleaning behavior
        verbose: Whether to print debug information
    """
    gt = pd.read_csv(ground_truth_path, encoding='latin1')
    pred = _load_predictions(predictions_path, dataset_type, verbose)
    gt = _normalize_ids(gt, dataset_type)
    pred = _normalize_ids(pred, dataset_type)

    if verbose:
        print("Converting to RecTools format")

    ground_truth = _to_rectools_format(gt, is_ground_truth=True, dataset_type=dataset_type)
    predictions = _to_rectools_format(pred, is_ground_truth=False, dataset_type=dataset_type)
    _validate_columns(ground_truth, predictions)

    interactions = _prepare_interactions(ground_truth)
    recommendations = _prepare_recommendations(predictions)

    if verbose:
        _print_debug_info(interactions, recommendations)

    return ProcessedData(
        ground_truth=ground_truth,
        predictions=predictions,
        interactions=interactions,
        recommendations=recommendations
    )


def _load_predictions(predictions_path, dataset_type, verbose):
    """Load predictions with appropriate cleaning for dataset type."""
    if dataset_type == "books":
        if verbose:
            print(f"Loading predictions (books mode): {predictions_path}")
        return pd.read_csv(predictions_path, encoding='latin1')

    else:
        if verbose:
            print(f"Cleaning predictions file (movies mode): {predictions_path}")

        valid_lines = []
        with open(predictions_path, 'r', encoding='latin1') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('userId') or stripped.startswith('user_id') or (
                        stripped and stripped[0].isdigit()):
                    valid_lines.append(line)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False,
                                         encoding='latin1') as temp:
            temp.writelines(valid_lines)
            temp_path = temp.name

        try:
            pred = pd.read_csv(temp_path, encoding='latin1')
        finally:
            os.unlink(temp_path)

        if verbose:
            print(f"Predictions loaded: {len(pred)} rows")

        return pred


def _normalize_ids(df, dataset_type):
    """Normalize ID columns by removing .0 decimals and converting to string."""
    df = df.copy()

    if dataset_type == "books":
        user_col = "userId"
        item_col = "itemId"
    else:
        user_col = "userId"
        item_col = "movieId"

    for col in [user_col, item_col]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)

    return df


def _to_rectools_format(df, is_ground_truth, dataset_type):
    """Convert DataFrame to RecTools standard format."""
    df = df.copy()

    if not is_ground_truth:
        df = df.drop(columns=["rating"], errors="ignore")

    column_map = {"userId": "user_id"}

    if dataset_type == "books":
        column_map["itemId"] = "item_id"
    else:
        if "movieId" in df.columns:
            column_map["movieId"] = "item_id"
        elif "itemId" in df.columns:
            column_map["itemId"] = "item_id"
        elif "title" in df.columns:
            column_map["title"] = "item_id"

    if is_ground_truth:
        rating_col = "rating"
    else:
        possible_cols = [
            'recommendation_score',"rating_pred", "predictedRating", "prediction",
            "predicted_rating", "mf_score",
            "test_predicted_rating", "predictedRating", "rating"
        ]
        rating_col = next((col for col in possible_cols if col in df.columns), None)

    if rating_col and rating_col in df.columns:
        column_map[rating_col] = "weight"

    df = df.rename(columns=column_map)

    if "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(str)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)
    if "weight" in df.columns:
        df["weight"] = df["weight"].astype(float)

    return df


def _validate_columns(gt, pred):
    """Validate required columns exist."""
    required = ["user_id", "item_id", "weight"]

    for df, name in [(gt, "Ground Truth"), (pred, "Predictions")]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")


def _prepare_interactions(df):
    """Prepare interactions DataFrame for RecTools."""
    interactions = df[["user_id", "item_id", "weight"]].copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["weight"] = interactions["weight"].astype(float)
    return interactions


def _prepare_recommendations(df):
    """Prepare recommendations DataFrame with ranks."""
    recos = df[["user_id", "item_id", "weight"]].copy()
    recos["user_id"] = recos["user_id"].astype(str)
    recos["item_id"] = recos["item_id"].astype(str)
    recos["weight"] = recos["weight"].astype(float)

    if "rank" not in recos.columns:
        recos = recos.sort_values(["user_id", "weight"], ascending=[True, False])
        recos["rank"] = recos.groupby("user_id").cumcount() + 1

    return recos


def _print_debug_info(interactions, recommendations):
    """Print debug information about data overlap."""
    print("\n" + "=" * 60)
    print("DATASET DEBUG INFORMATION")
    print("=" * 60)

    print("\nInteractions (ground truth) sample:")
    print(interactions.head(3).to_string(index=False))

    print("\nRecommendations (predictions) sample:")
    print(recommendations.head(3).to_string(index=False))

    interaction_set = set(zip(interactions['user_id'], interactions['item_id']))
    recommendation_set = set(zip(recommendations['user_id'], recommendations['item_id']))

    print(f"\nTotal interaction pairs: {len(interaction_set)}")
    print(f"Total recommendation pairs: {len(recommendation_set)}")

    test_pairs = list(zip(interactions['user_id'][:3], interactions['item_id'][:3]))
    for i, (u, m) in enumerate(test_pairs, 1):
        exists = (u, m) in recommendation_set
        print(f"  Pair {i}: ({u}, {m}) exists in predictions? {exists}")

    common = interaction_set & recommendation_set
    print(f"\nCommon pairs: {len(common)}")

    print(f"\nInteraction dtypes:\n{interactions.dtypes}")
    print(f"\nRecommendation dtypes:\n{recommendations.dtypes}")