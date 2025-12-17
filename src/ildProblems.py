import pandas as pd  # FIXED: was incorrectly imported as numpy
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances
from sklearn.exceptions import DataConversionWarning
import warnings
import os

warnings.filterwarnings("ignore", category=DataConversionWarning)


def diagnose_movies_ild(ground_truth_path, item_features_path, catalog_path,
                        sample_baseline_path=None, k=10, dataset_type="movies"):
    """
    Diagnose ILD calculation issues specific to MovieLens.
    Expected ILD@10 for random: ~0.75-0.85
    """

    print("=" * 70)
    print("MOVIELENS ILD DIAGNOSTIC TOOL")
    print("=" * 70)

    # 1. Load and analyze catalog
    print("\n[1] CATALOG ANALYSIS")
    try:
        catalog = pd.read_csv(catalog_path)
        print(f"✓ Loaded catalog: {len(catalog)} movies")
    except Exception as e:
        print(f"✗ Error loading catalog: {e}")
        return

    # For MovieLens, ensure movieId is string
    id_col = 'movieId' if 'movieId' in catalog.columns else 'itemId'
    if id_col not in catalog.columns:
        print(f"✗ No 'movieId' or 'itemId' column found in catalog!")
        print(f"   Available columns: {list(catalog.columns)}")
        return

    catalog[id_col] = catalog[id_col].astype(str).str.replace(r'\.0$', '', regex=True)
    catalog = catalog.set_index(id_col)

    # Genre distribution
    if 'genres' in catalog.columns:
        all_genres = catalog['genres'].fillna('Unknown').str.split('|').explode()
        genre_counts = all_genres.value_counts()
        print(f"\nGenre distribution ({len(genre_counts)} genres):")
        for genre, count in genre_counts.head(10).items():
            print(f"  {genre:15s}: {count:5d} movies ({count / len(catalog):5.1%})")

        # Diversity metrics
        genre_entropy = -(genre_counts / len(catalog) * np.log2(genre_counts / len(catalog))).sum()
        print(f"\nGenre entropy: {genre_entropy:.2f} bits (max={np.log2(len(genre_counts)):.2f})")

        if genre_entropy < 3.0:
            print("⚠️  WARNING: Low entropy indicates unbalanced genre distribution")
        else:
            print("✓ Genre distribution is reasonably diverse")
    else:
        print("⚠️  No 'genres' column found in catalog!")
        return

    # 2. Load feature matrix
    print("\n[2] FEATURE MATRIX ANALYSIS")
    try:
        item_features = pd.read_csv(item_features_path, index_col=0)
        print(f"✓ Loaded feature matrix: {item_features.shape}")
    except Exception as e:
        print(f"✗ Error loading item_features: {e}")
        # Try reading without index_col
        try:
            item_features = pd.read_csv(item_features_path)
            if item_features.columns[0] in ['item_id', 'movieId', 'itemId']:
                item_features = item_features.set_index(item_features.columns[0])
                print(f"✓ Loaded feature matrix (fixed index): {item_features.shape}")
            else:
                print(f"✗ Could not determine index column")
                return
        except Exception as e2:
            print(f"✗ Still failed: {e2}")
            return

    print(f"Index name: {item_features.index.name}")
    print(f"First 5 index values: {list(item_features.index[:5])}")
    print(f"Columns: {list(item_features.columns[:10])}{'...' if len(item_features.columns) > 10 else ''}")

    # Normalize index to strings
    item_features.index = item_features.index.astype(str).str.replace(r'\.0$', '', regex=True)

    # Check sparsity
    sparsity = (item_features == 0).mean().mean()
    print(f"Matrix sparsity: {sparsity:.1%}")
    if sparsity > 0.95:
        print("⚠️  Very sparse matrix may cause low distances")
    else:
        print("✓ Sparsity is normal")

    # Check for binary values (0/1)
    unique_vals = np.unique(item_features.values)
    if set(unique_vals).issubset({0, 1}):
        print("✓ Features are binary (0/1)")
    else:
        print(f"⚠️  Features are not binary! Unique values: {unique_vals}")

    # 3. Sample distance calculations
    print("\n[3] SAMPLE DISTANCE CALCULATIONS")

    # Find movies from different genres for testing
    test_movies = []
    test_genres = ['Action', 'Drama', 'Comedy', 'Animation', 'Documentary', 'Horror', 'Sci-Fi']

    for genre in test_genres:
        genre_movies = catalog[catalog['genres'].str.contains(genre, na=False)].index
        if len(genre_movies) > 0:
            # Take first movie that's also in features
            for movie in genre_movies:
                if movie in item_features.index:
                    test_movies.append(movie)
                    print(f"  Found '{genre}' movie: {movie}")
                    break

    # If we didn't find enough movies, just use what's available
    if len(test_movies) < 3:
        available_movies = catalog.index.intersection(item_features.index)
        test_movies = available_movies[:5].tolist()
        print(f"  Using first available movies: {test_movies}")

    print(f"\nTest movies: {test_movies}")

    # Get features for test movies
    try:
        features_sample = item_features.loc[test_movies]
        print(f"\nFeature vectors for test movies:")
        print(features_sample)

        # Calculate distances
        print(f"\n📐 CALCULATING DISTANCES...")

        # Cosine distances
        cosine_dists = cosine_distances(features_sample)
        print(f"Cosine distance matrix ({len(test_movies)} movies):")
        print(np.round(cosine_dists, 3))

        # Check diagonal (should be 0)
        diag_mean = np.diag(cosine_dists).mean()
        if diag_mean > 0.001:
            print(f"⚠️  Diagonal is not zero: {diag_mean}")

        # Check self-similarity
        upper_tri = cosine_dists[np.triu_indices_from(cosine_dists, k=1)]
        if len(upper_tri) > 0:
            avg_off_diag = upper_tri.mean()
            print(f"\nAverage off-diagonal cosine distance: {avg_off_diag:.3f}")

            if avg_off_diag < 0.6:
                print("⚠️  WARNING: Cosine distances between different genres are too low!")
                print("   For MovieLens, this should be 0.7+")
            elif avg_off_diag > 0.7:
                print("✓ Cosine distances look healthy")

        # For binary features, also test Jaccard
        if set(np.unique(features_sample.values)).issubset({0, 1}):
            try:
                from scipy.spatial.distance import pdist, squareform
                jaccard_dists = squareform(pdist(features_sample.values, metric='jaccard'))
                print(f"\nJaccard distance matrix:")
                print(np.round(jaccard_dists, 3))

                jaccard_off_diag = jaccard_dists[np.triu_indices_from(jaccard_dists, k=1)].mean()
                print(f"Average off-diagonal Jaccard distance: {jaccard_off_diag:.3f}")
            except ImportError:
                print("(scipy not available for Jaccard)")

    except KeyError as e:
        print(f"✗ Movie IDs not found in feature matrix: {e}")
        print(f"   Feature matrix has {len(item_features)} items")
        print(f"   First 10: {list(item_features.index[:10])}")
        return

    # 4. Analyze recommendations if provided
    if sample_baseline_path and os.path.exists(sample_baseline_path):
        print("\n[4] RECOMMENDATIONS ANALYSIS")
        try:
            baseline = pd.read_csv(sample_baseline_path)
            baseline['item_id'] = baseline['item_id'].astype(str).str.replace(r'\.0$', '', regex=True)

            print(f"Baseline file: {len(baseline)} recommendations")

            # Check a single user's recommendations
            sample_user = baseline['user_id'].unique()[0]
            user_recos = baseline[baseline['user_id'] == sample_user].head(k)

            print(f"\nSample user {sample_user} top-{k} recommendations:")
            for idx, row in user_recos.iterrows():
                movie_id = row['item_id']
                # Find movie in catalog
                if movie_id in catalog.index:
                    movie_title = catalog.loc[movie_id, 'title'] if 'title' in catalog.columns else movie_id
                    movie_genres = catalog.loc[movie_id, 'genres']
                    print(f"  {movie_id}: {movie_genres:30s} ({movie_title[:30]})")
                else:
                    print(f"  {movie_id}: NOT IN CATALOG!")

            # Check if items exist in feature matrix
            in_features = user_recos['item_id'].isin(item_features.index).sum()
            print(f"\nItems with features: {in_features}/{len(user_recos)}")

            # Check diversity manually
            recos_in_features = user_recos[user_recos['item_id'].isin(item_features.index)]
            if len(recos_in_features) >= 2:
                features_recos = item_features.loc[recos_in_features['item_id']]
                cosine_manual = cosine_distances(features_recos)
                manual_ild = cosine_manual[np.triu_indices_from(cosine_manual, k=1)].mean()
                print(f"Manual ILD for this user: {manual_ild:.3f}")

                if manual_ild < 0.5:
                    print("⚠️  This user's recommendations are NOT diverse!")
                else:
                    print("✓ This user's recommendations show good diversity")

            # Check overall coverage
            total_in_features = baseline['item_id'].isin(item_features.index).sum()
            print(f"\nOverall coverage: {total_in_features}/{len(baseline)} items have features")

        except Exception as e:
            print(f"✗ Error analyzing recommendations: {e}")

    # 5. Test random recommendations
    print("\n[5] RANDOM RECOMMENDATIONS SANITY TEST")

    # Generate truly random recommendations for 100 users
    all_items = item_features.index.tolist()

    print(f"Testing {len(all_items)} items with k={k}")

    ild_values = []
    for user in range(100):
        # Random items
        user_items = np.random.choice(all_items, size=min(k, len(all_items)), replace=False)

        # Features
        features = item_features.loc[user_items].values

        # Cosine distances
        cosine_manual = cosine_distances(features)
        manual_ild = cosine_manual[np.triu_indices_from(cosine_manual, k=1)].mean()
        ild_values.append(manual_ild)

    random_ild = np.mean(ild_values)
    random_ild_std = np.std(ild_values)

    print(f"Expected ILD for random: 0.75-0.85")
    print(f"Calculated ILD: {random_ild:.3f} ± {random_ild_std:.3f}")

    if 0.75 <= random_ild <= 0.85:
        print("✓ Feature matrix and distance calculation are WORKING")
        print("✓ The problem is likely in your _calculate_ild() function")
    else:
        print("✗ Feature matrix or distances are BROKEN")
        print("  Check: ID alignment, feature values, distance calculation")

    # 6. Summary
    print("\n" + "=" * 70)
    print("EXPECTED vs YOUR ILD VALUES")
    print("=" * 70)
    print(f"Random recommendations:     0.75 - 0.85")
    print(f"Popularity-based:           0.60 - 0.70 (less diverse)")
    print(f"Collaborative filtering:    0.65 - 0.75")
    print(f"Content-based:              0.40 - 0.55 (overly specialized)")
    print(f"Your result (~0.5):         ⚠️  TOO LOW for random")

    if 0.75 <= random_ild <= 0.85:
        print(f"\n🎯 TARGET: Replace your _calculate_ild() with the corrected version")
        print("           Your feature matrix is fine, the bug is in the ILD function")
    else:
        print(f"\n🎯 TARGET: Fix item feature matrix first")

    print("=" * 70)
    print("\n🔧 FIXES TO TRY:")
    print("1. If random ILD in [5] is 0.75-0.85:")
    print("   → Replace _calculate_ild() with calculate_correct_ild()")
    print("2. If random ILD in [5] is NOT 0.75-0.85:")
    print("   → Check item ID alignment between features and catalog")
    print("   → Verify genres are correctly one-hot encoded")
    print("   → Ensure no duplicate item IDs in feature matrix")
    print("3. Re-run diagnostic after fixes")
    print("=" * 70)


def calculate_correct_ild(data_handler, item_features, k=10, metric='cosine'):
    """
    Simplified and corrected ILD calculation.
    Should give ~0.80 for MovieLens random recommendations.
    """
    recos = data_handler.recommendations.copy()
    recos['item_id'] = recos['item_id'].astype(str)
    recos = recos[recos['rank'] <= k]

    print(f"\nCalculating ILD@{k} for {len(recos)} recommendations ({recos['user_id'].nunique()} users)")

    # Filter to items that have features
    available_items = list(set(recos['item_id'].unique()) & set(item_features.index.astype(str)))
    if not available_items:
        print("⚠️  No items in recommendations have features!")
        return np.nan

    print(f"Items with features: {len(available_items)}/{recos['item_id'].nunique()} unique items")

    # Create distance matrix
    feature_matrix = item_features.loc[available_items].values

    if metric == 'cosine':
        # Normalize features
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        feature_matrix_normalized = feature_matrix / norms

        # Cosine distance = 1 - cosine similarity
        similarity_matrix = np.dot(feature_matrix_normalized, feature_matrix_normalized.T)
        distance_matrix = 1 - similarity_matrix

        # Ensure it's in [0,1] range
        distance_matrix = np.clip(distance_matrix, 0, 1)

        # Zero out diagonal
        np.fill_diagonal(distance_matrix, 0)

        print(
            f"Distance matrix: {distance_matrix.shape}, range [{distance_matrix.min():.3f}, {distance_matrix.max():.3f}]")

    elif metric == 'jaccard':
        try:
            from scipy.spatial.distance import pdist, squareform
            distance_matrix = squareform(pdist(feature_matrix, metric='jaccard'))
        except ImportError:
            # Manual Jaccard
            n = len(feature_matrix)
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    intersection = np.sum(np.minimum(feature_matrix[i], feature_matrix[j]))
                    union = np.sum(np.maximum(feature_matrix[i], feature_matrix[j]))
                    distance_matrix[i, j] = 1 - (intersection / union) if union > 0 else 0
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Create mapping
    item_to_idx = {item: idx for idx, item in enumerate(available_items)}

    # Calculate ILD per user
    ild_values = []
    skipped_users = 0

    for user_id, user_recos in recos.groupby('user_id'):
        user_items = user_recos['item_id'].astype(str).tolist()
        user_items = [item for item in user_items if item in item_to_idx]

        if len(user_items) >= 2:
            # Get pairwise distances
            idxs = [item_to_idx[item] for item in user_items]
            user_distances = distance_matrix[np.ix_(idxs, idxs)]

            # Average of upper triangle (excluding diagonal)
            upper_tri = user_distances[np.triu_indices_from(user_distances, k=1)]
            if len(upper_tri) > 0:
                ild_values.append(upper_tri.mean())
        else:
            skipped_users += 1

    if skipped_users > 0:
        print(f"Skipped {skipped_users} users with <2 items having features")

    final_ild = np.mean(ild_values) if ild_values else np.nan

    # Sanity check
    if final_ild < 0.5:
        print(f"⚠️  WARNING: ILD={final_ild:.3f} is suspiciously low!")
        print(f"   Expected: 0.75-0.85 for random")
    elif final_ild > 0.7:
        print(f"✓ ILD={final_ild:.3f} looks healthy")

    return final_ild


if __name__ == "__main__":
    # MovieLens dataset paths
    GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies_100k_train.csv"
    CATALOG = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\movies_100k\movies.csv"
    ITEM_FEATURES = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\movies_100k\item_features.csv"
    BASELINE_SAMPLE = None  # Set to a random baseline path if you have one

    diagnose_movies_ild(
        ground_truth_path=GROUND_TRUTH,
        item_features_path=ITEM_FEATURES,
        catalog_path=CATALOG,
        sample_baseline_path=BASELINE_SAMPLE,
        k=10,
        dataset_type="movies"
    )

# Example usage
if __name__ == "__main__":
    # MovieLens dataset paths
    GROUND_TRUTH = r"CC:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies_groundtruth.csv"  # or wherever your training data is
    CATALOG = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
    ITEM_FEATURES = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"

    # Optional: path to one of your random baseline files
    BASELINE_SAMPLE = r"C:\path\to\random_top10_run01.csv"

    diagnose_movies_ild(
        ground_truth_path=GROUND_TRUTH,
        item_features_path=ITEM_FEATURES,
        catalog_path=CATALOG,
        sample_baseline_path=BASELINE_SAMPLE,
        k=10
    )