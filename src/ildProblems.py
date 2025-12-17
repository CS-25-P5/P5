import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import jaccard
import warnings

warnings.filterwarnings("ignore")


def generate_baselines_for_test(gt_path, catalog_path, k=10):
    """Generate random and popularity baselines to compare against your model"""

    print(f"\n{'=' * 60}")
    print("GENERATING BASELINES FOR ILD COMPARISON")
    print(f"{'=' * 60}")

    # Load data
    gt = pd.read_csv(gt_path)
    catalog = pd.read_csv(catalog_path)

    # Create one-hot encoded features
    all_genres = set()
    for genres in catalog['genres'].fillna('Unknown'):
        all_genres.update(genres.split('|'))
    all_genres = sorted([g for g in all_genres if g not in ['Unknown', '(no genres listed)']])

    item_features = pd.DataFrame(0, index=catalog['itemId'].astype(str), columns=all_genres)
    item_features.index.name = 'item_id'

    for _, row in catalog.iterrows():
        item_id = str(row['itemId'])
        for genre in row['genres'].split('|'):
            if genre in all_genres:
                item_features.at[item_id, genre] = 1

    print(f"Feature matrix: {item_features.shape}")

    # Users and items
    users = gt['userId'].unique()
    all_items = item_features.index.tolist()

    # 1. RANDOM baseline
    print("\nGenerating random baseline...")
    random_recos = []
    for user in users:
        items = np.random.choice(all_items, size=min(k, len(all_items)), replace=False)
        for rank, item in enumerate(items, 1):
            random_recos.append({'user_id': str(user), 'item_id': str(item), 'rank': rank})

    random_df = pd.DataFrame(random_recos)

    # 2. POPULARITY baseline
    print("Generating popularity baseline...")
    item_popularity = gt['movieId'].astype(str).value_counts()
    top_items = item_popularity.head(k).index.tolist()
    print(f"Top {k} popular items: {top_items}")

    popular_recos = []
    for user in users:
        for rank, item in enumerate(top_items, 1):
            popular_recos.append({'user_id': str(user), 'item_id': str(item), 'rank': rank})

    popular_df = pd.DataFrame(popular_recos)

    # Calculate ILD
    def calculate_ild_fast(recos_df, item_features, k=10):
        recos = recos_df.copy()
        recos['item_id'] = recos['item_id'].astype(str)

        # Get items with features
        available = list(set(recos['item_id'].unique()) & set(item_features.index))
        if not available:
            return np.nan

        features = item_features.loc[available].values

        # Distance matrices (both metrics)
        cos_dist = cosine_distances(features)

        # Jaccard using scipy
        from scipy.spatial.distance import pdist, squareform
        jac_dist = squareform(pdist(features, metric='jaccard'))

        item_to_idx = {item: idx for idx, item in enumerate(available)}

        # ILD per user
        jac_values, cos_values = [], []
        for user_id, user_recos in recos.groupby('user_id'):
            items = user_recos['item_id'].tolist()
            items = [i for i in items if i in item_to_idx]

            if len(items) >= 2:
                idxs = [item_to_idx[i] for i in items]

                user_jac = jac_dist[np.ix_(idxs, idxs)]
                user_cos = cos_dist[np.ix_(idxs, idxs)]

                jac_values.append(user_jac[np.triu_indices_from(user_jac, k=1)].mean())
                cos_values.append(user_cos[np.triu_indices_from(user_cos, k=1)].mean())
            else:
                print(f"  User {user_id}: only {len(items)} items, skipping")

        return np.mean(jac_values), np.mean(cos_values)

    print("\nCalculating ILD metrics...")
    random_ild = calculate_ild_fast(random_df, item_features, k)
    popular_ild = calculate_ild_fast(popular_df, item_features, k)

    # Display results
    print(f"\n{'=' * 60}")
    print("BASELINE COMPARISON ON YOUR TEST SET")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Jaccard ILD':<15} {'Cosine ILD':<15}")
    print(f"{'-' * 60}")
    print(f"{'Random':<20} {random_ild[0]:<15.4f} {random_ild[1]:<15.4f}")
    print(f"{'Popularity':<20} {popular_ild[0]:<15.4f} {popular_ild[1]:<15.4f}")
    print(f"{'Your Model':<20} {0.8133:<15.4f} {0.7356:<15.4f}")
    print(f"{'=' * 60}")


# Run it
if __name__ == "__main__":
    generate_baselines_for_test(
        r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\grount_truth_test2.csv",
        r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\movies_test2.csv"
    )