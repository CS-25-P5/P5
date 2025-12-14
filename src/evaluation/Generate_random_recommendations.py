import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List


def generate_random_baseline(ground_truth_path, output_dir, k=10,
                             dataset_type="movies", seed=42, num_runs=1):
    """
    Generate multiple random recommendation runs saved as separate files.
    Returns list of file paths.
    """
    print(f"\n{'=' * 60}")
    print(f"GENERATING {num_runs} RANDOM BASELINE RUNS (K={k})")
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

    unique_users = gt[user_col].unique()
    unique_items = gt[item_col].unique()

    print(f"Users: {len(unique_users)}, Items: {len(unique_items)}")

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
        output_path = os.path.join(output_dir, f"random_top{k}_run{run + 1:02d}_{timestamp}.csv")
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
    # MOVIES 1M
    GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_1M_movies_test.csv"
    OUTPUT_DIR = r"C:\Users\Jacob\Documents\GitHub\P5\src\baselines\movies\1m"
    DATASET_TYPE = "movies"

    K = 10
    NUM_RANDOM_RUNS = 25  # <-- Generate 25 separate runs

    # ==================== GENERATE BASELINES ====================
    # Generate 25 separate random baseline files
    print("GENERATING RANDOM BASELINES")
    random_paths = generate_random_baseline(
        ground_truth_path=GROUND_TRUTH,
        output_dir=OUTPUT_DIR,
        k=K,
        dataset_type=DATASET_TYPE,
        seed=42,
        num_runs=NUM_RANDOM_RUNS
    )

    # Generate 1 popularity baseline file
    print("\nGENERATING POPULARITY BASELINE")
    pop_paths = generate_popularity_baseline(
        ground_truth_path=GROUND_TRUTH,
        output_dir=OUTPUT_DIR,
        k=K,
        dataset_type=DATASET_TYPE
    )

    # ==================== OUTPUT INSTRUCTIONS ====================
    print("\n" + "=" * 60)
    print("📋 ADD THESE TO YOUR DataPaths.py MODELS LIST:")
    print("=" * 60)

    # Random baselines (25 entries)
    print("# Random Baselines (25 runs):")
    for path in random_paths:
        print(f"('{path}', 'Random-Baseline'),")

    # Popularity baseline (1 entry)
    print("\n# Popularity Baseline (1 run):")
    for path in pop_paths:
        print(f"('{path}', 'Popularity-Baseline'),")

    print("\n# Then in your evaluation script, compute metrics for each Random-Baseline")
    print("# and average the results to get: Mean ± Std Dev")
    print("=" * 60)