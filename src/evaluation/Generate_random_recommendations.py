import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Optional


def generate_random_baseline(ground_truth_path: str, output_dir: str, k: int = 10,
                             dataset_type: str = "movies", seed: int = 42, num_runs: int = 1,
                             predictions_path: Optional[str] = None,
                             catalog_path: Optional[str] = None):
    """
    Generate multiple random recommendation runs saved as separate files.

    Uses items from catalog (or ground truth if no catalog provided)
    and users from predictions file (or ground truth if no predictions provided).

    Returns list of generated file paths.
    """
    print(f"\n{'=' * 60}")
    print(f"GENERATING {num_runs} RANDOM BASELINE RUNS (K={k})")
    print(f"{'=' * 60}")

    # Load GROUND TRUTH for USERS (if predictions not provided)
    gt = pd.read_csv(ground_truth_path, encoding='latin1')

    # Detect columns for ground truth
    if dataset_type == "books":
        user_col_gt, item_col_gt = "userId", "itemId"
    else:
        user_col_gt = "userId"
        item_col_gt = "movieId" if "movieId" in gt.columns else "itemId"

    gt[user_col_gt] = gt[user_col_gt].astype(str).str.replace(r'\.0$', '', regex=True)
    if item_col_gt in gt.columns:
        gt[item_col_gt] = gt[item_col_gt].astype(str).str.replace(r'\.0$', '', regex=True)

    # Load CATALOG for ITEMS (critical methodological change)
    if catalog_path and os.path.exists(catalog_path):
        catalog_df = pd.read_csv(catalog_path, encoding='latin1')

        # Find item column in catalog
        possible_item_cols = [col for col in catalog_df.columns if
                              'item' in col.lower() or 'movie' in col.lower() or 'book' in col.lower()]
        if not possible_item_cols:
            raise ValueError(f"No item column found in catalog. Columns: {catalog_df.columns}")

        item_col_cat = possible_item_cols[0]
        catalog_df[item_col_cat] = catalog_df[item_col_cat].astype(str).str.replace(r'\.0$', '', regex=True)
        unique_items = catalog_df[item_col_cat].unique()

        print(f"✓ Loaded catalog: {len(unique_items):,} total items")
    else:
        # FALLBACK: Use items from ground truth (with warning)
        if item_col_gt in gt.columns:
            unique_items = gt[item_col_gt].unique()
            print(f"⚠️  WARNING: No catalog provided. Using {len(unique_items)} items from ground truth.")
            print("   This may inflate baseline performance. Provide catalog_path for proper evaluation.")
        else:
            raise ValueError("No item column found in ground truth and no catalog provided!")

    # Load PREDICTIONS for USERS (if provided)
    if predictions_path and os.path.exists(predictions_path):
        pred = pd.read_csv(predictions_path, encoding='latin1')

        # Auto-detect user column in predictions file
        possible_user_cols = [col for col in pred.columns if 'user' in col.lower()]
        if not possible_user_cols:
            raise ValueError(f"No user column found in predictions file. Columns: {pred.columns}")

        user_col_pred = possible_user_cols[0]
        pred[user_col_pred] = pred[user_col_pred].astype(str).str.replace(r'\.0$', '', regex=True)
        unique_users = pred[user_col_pred].unique()

        print(f"Users from predictions file: {len(unique_users):,}")
    else:
        # Fallback: use users from ground truth
        unique_users = gt[user_col_gt].unique()
        print(f"Users from ground truth: {len(unique_users):,}")

    generated_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for run in range(num_runs):
        print(f"\n--- Run {run + 1:>2}/{num_runs} (seed={seed + run}) ---")
        np.random.seed(seed + run)

        # Generate recommendations
        recommendations = []
        for user in unique_users:
            items = np.random.choice(unique_items, size=min(k, len(unique_items)),
                                     replace=False if k <= len(unique_items) else True)
            scores = np.random.random(len(items))

            for item, score in zip(items, scores):
                recommendations.append({
                    'userId': user,
                    'itemId': item,
                    'recommendation_score': score
                })

        recs_df = pd.DataFrame(recommendations)
        recs_df = recs_df.sort_values(['userId', 'recommendation_score'], ascending=[True, False])

        # Save individual run
        output_path = os.path.join(output_dir, f"random_top{k}_run{run + 1:02d}_Movies.csv")
        os.makedirs(output_dir, exist_ok=True)
        recs_df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_path} ({len(recs_df)} rows)")
        generated_paths.append(output_path)

    print(f"\n🔷 Generated {len(generated_paths)} random baseline files")
    return generated_paths

def generate_popularity_baseline(ground_truth_path, output_dir, k=10, dataset_type="movies"):
    """Generate popularity-based recommendations - single deterministic file"""
    print(f"\n{'=' * 60}")
    print(f"GENERATING POPULARITY BASELINE (K={k})")
    print(f"{'=' * 60}")

    gt = pd.read_csv(ground_truth_path, encoding='latin1')

    # Normalize IDs
    if dataset_type == "books":
        user_col, item_col = "userId", "itemId"
    else:
        user_col = "userId"
        item_col = "movieId" if "movieId" in gt.columns else "itemId"

    gt[user_col] = gt[user_col].astype(str).str.replace(r'\.0$', '', regex=True)
    gt[item_col] = gt[item_col].astype(str).str.replace(r'\.0$', '', regex=True)

    # Get top-K most popular items
    item_popularity = gt[item_col].value_counts().head(k)
    top_items = item_popularity.index.tolist()

    print(f"Top {k} items: {top_items}")
    print(f"Popularity counts: {item_popularity.values}")

    unique_users = gt[user_col].unique()
    print(f"Recommending to {len(unique_users)} users")

    recommendations = []
    for user in unique_users:
        for rank, item in enumerate(top_items):
            recommendations.append({
                'userId': user,
                'itemId': item,
                'recommendation_score': k - rank  # Highest score for most popular
            })

    recs_df = pd.DataFrame(recommendations)
    recs_df = recs_df.sort_values(['userId', 'recommendation_score'], ascending=[True, False])

    # Save popularity baseline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"popularity_top{k}_{timestamp}.csv")
    os.makedirs(output_dir, exist_ok=True)
    recs_df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path} ({len(recs_df)} rows)")
    return [output_path]  # Return as list for consistency


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # BOOKS 100k
    # CATALOG = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"
    # GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books_groundtruth.csv"
    # PREDICTIONS = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\dpp_train_jaccard_recommendations.csv"

    # movies 100k
    CATALOG = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
    GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies_groundtruth.csv"
    PREDICTIONS = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_00-49-51\mf_test_100000_top_10.csv"

    OUTPUT_DIR = r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\random"
    DATASET_TYPE = "movies"

    K = 10
    NUM_RANDOM_RUNS = 25

    # Generate 1 popularity baseline file
    print("\nGENERATING POPULARITY BASELINE")
    pop_paths = generate_popularity_baseline(
        ground_truth_path=GROUND_TRUTH,
        output_dir=OUTPUT_DIR,
        k=K,
        dataset_type=DATASET_TYPE
    )


    # ==================== GENERATE BASELINES ====================
    # Generate 25 random baseline files (using predictions user count + catalog items)
    print("GENERATING RANDOM BASELINES")
    random_paths = generate_random_baseline(
        ground_truth_path=GROUND_TRUTH,
        catalog_path=CATALOG,  # <-- NEW PARAMETER
        predictions_path=PREDICTIONS,
        output_dir=OUTPUT_DIR,
        k=K,
        dataset_type=DATASET_TYPE,
        seed = 12,
        num_runs=NUM_RANDOM_RUNS
    )

    # ==================== OUTPUT INSTRUCTIONS ====================
    print("\n" + "=" * 60)
    print("📋 ADD THESE TO YOUR DataPaths.py MODELS LIST:")
    print("=" * 60)

    # Random baselines (25 entries)
    print("# Random Baselines (25 runs):")
    for path in random_paths:
        print(f"('{path}', 'Random-Baseline'),")

    print("\n# Then compute metrics for each Random-Baseline and average results")
    print("=" * 60)