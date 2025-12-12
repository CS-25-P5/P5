import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rectools.metrics import (
    Precision, Recall, F1Beta, MAP, NDCG, MRR, HitRate,
    IntraListDiversity, CatalogCoverage, calc_metrics
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator
from DataHandler2 import load_and_process_data
import os
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn.metrics.pairwise")

# Import from new modules
from Plotting import plot_individual_metric_charts, plot_rating_distribution
from Diagnostics import _print_data_diagnostics

# Calculate most metrics using RecTools
def calculate_all_metrics(catalog, data_handler, threshold=4.0, k=5, item_features=None,
                         model_name="Unknown", calculate_ild=True):
    results = {}

    # RMSE & MAE (prediction metrics) - with error handling
    print(f"Calculating RMSE and MAE for {model_name}")
    rmse, mae = _calculate_rating_metrics(data_handler, model_name)
    results["RMSE"] = rmse
    results["MAE"] = mae

    # Top-K Metrics (using filtered relevant interactions)
    print(f"Calculating top-{k} metrics for {model_name}")

    # Filter interactions to only include relevant items (rating >= threshold), Necesarry for ranking metrics
    relevant_interactions = data_handler.interactions[
        data_handler.interactions['weight'] >= threshold
        ].copy()

    # Create catalog of all items from ground truth
    gt_items = data_handler.interactions["item_id"].unique()
    pred_items = data_handler.recommendations["item_id"].unique()
    catalog = np.union1d(gt_items, pred_items)
    catalog_size = len(catalog)
    print(f"Catalog size is: {catalog_size}")

    # Create dictionary of metrics to calculate and set their variables
    metrics = {
        f'Precision@{k}': Precision(k=k),  # Proportion of recommended items that are relevant
        f'Recall@{k}': Recall(k=k),  # Proportion of items that where recommended
        f'F1Beta@{k}': F1Beta(k=k, beta=1.0),  # Harmonic mean of Precision and recall
        f'MAP@{k}': MAP(k=k),  # Considers ranking order of relevant items
        f'NDCG@{k}': NDCG(k=k),  # Rewards relevant items appearing earlier in recommendations
        f'MRR@{k}': MRR(k=k),  # focuses on position of the first relevant item
        f'CatalogCoverage@{k}': CatalogCoverage(k=k),  # proportion of catalog items that appear in recommendations
        f'HitRate@{k}': HitRate(k=k), #proportion of users with at least 1 match
    }

    # Calculate metrics
    try:
        metrics_values = calc_metrics(
            metrics=metrics,
            reco=data_handler.recommendations,
            interactions=relevant_interactions,
            catalog=catalog,
            prev_interactions=None,
        )

        # Store calculated metrics in results dictionary
        results[f"HitRate@{k}"] = metrics_values[f'HitRate@{k}']
        results[f"Precision@{k}"] = metrics_values[f'Precision@{k}']
        results[f"Recall@{k}"] = metrics_values[f'Recall@{k}']
        results[f"F1@{k}"] = metrics_values[f'F1Beta@{k}']
        results[f"MAP@{k}"] = metrics_values[f'MAP@{k}']
        results[f"NDCG@{k}"] = metrics_values[f'NDCG@{k}']
        results[f"MRR@{k}"] = metrics_values[f'MRR@{k}']
        results[f"Coverage@{k}"] = metrics_values[f'CatalogCoverage@{k}'] / catalog_size

    except Exception as e:
        print(f"Warning: Error calculating ranking metrics for {model_name}: {e}")
        print("Filling with NaN values and continuing")
        results[f"HitRate@{k}"] = np.nan
        results[f"Precision@{k}"] = np.nan
        results[f"Recall@{k}"] = np.nan
        results[f"F1@{k}"] = np.nan
        results[f"MAP@{k}"] = np.nan
        results[f"NDCG@{k}"] = np.nan
        results[f"MRR@{k}"] = np.nan
        results[f"Coverage@{k}"] = np.nan

    # intra list diversity (ILD)
    # Change this block (around line 95):
    if calculate_ild and item_features is not None and not item_features.empty:
        print(f"Calculating ILD@{k} for {model_name}")
        results[f"ILD@{k}_Hamming"] = _calculate_ild(data_handler, item_features, k, metric='hamming')
        results[f"ILD@{k}_Jaccard"] = _calculate_ild(data_handler, item_features, k, metric='jaccard')
        results[f"ILD@{k}_Cosine"] = _calculate_ild(data_handler, item_features, k, metric='cosine')
    else:
        # Set all three to NaN if skipped
        results[f"ILD@{k}_Hamming"] = np.nan
        results[f"ILD@{k}_Jaccard"] = np.nan
        results[f"ILD@{k}_Cosine"] = np.nan

    # Filter recommendations to only the Top K for Gini calculation
    top_k_recos = data_handler.recommendations[data_handler.recommendations['rank'] <= k]

    # Reverse Gini (Popularity Bias)
    print(f"Calculating Reverse Gini for {model_name}")
    results['Reverse Gini'] = _calculate_reverse_gini(top_k_recos)

    return results

# Calculate RMSE and MAE using scikit-learn - with error handling
def _calculate_rating_metrics(data_handler, model_name="Unknown"):
    try:
        # Merge predictions with ground truth
        merged = pd.merge(
            data_handler.predictions,
            data_handler.interactions,
            on=['user_id', 'item_id'],
            suffixes=('_pred', '_gt')
        )

        # Remove any rows with NaN values
        merged_clean = merged[['weight_gt', 'weight_pred']].dropna()

        if merged_clean.empty:
            print(f"Warning for {model_name}: After merge, no valid rating pairs found (all NaN).")
            return np.nan, np.nan

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(
            merged_clean['weight_gt'],
            merged_clean['weight_pred']
        ))
        mae = mean_absolute_error(
            merged_clean['weight_gt'],
            merged_clean['weight_pred']
        )
        return rmse, mae

    except Exception as e:
        print(f"\nError in RMSE/MAE calculation for {model_name}: {e}")
        print("   Returning NaN for RMSE and MAE to allow other metrics to continue.\n")
        return np.nan, np.nan

# Calculate Intra-List Diversity with RecTools
def _calculate_ild(data_handler, item_features, k, metric='hamming'):
    try:
        recos = data_handler.recommendations.copy()
        recos['item_id'] = recos['item_id'].astype(str)
        available = list(set(recos['item_id'].unique()) & set(item_features.index.astype(str)))

        if not available:
            return np.nan

        features = item_features.loc[available]

        class Calculator:
            def __init__(self, f, m):
                # Suppress the jaccard warning since we know data is binary
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=DataConversionWarning,
                        module="sklearn.metrics.pairwise"
                    )
                    self.d = pairwise_distances(f.values, metric=m)

                self.ids = f.index.tolist()
                self.map = {id_: i for i, id_ in enumerate(self.ids)}

            def get_distances(self, ids):
                # Handle tuple of two arrays (rectools passes this way)
                if isinstance(ids, tuple) and len(ids) == 2:
                    item_0, item_1 = ids
                    # Convert to numpy arrays
                    item_0 = item_0.values if hasattr(item_0, 'values') else np.asarray(item_0)
                    item_1 = item_1.values if hasattr(item_1, 'values') else np.asarray(item_1)
                    # Build distances for each pair
                    result = []
                    for a, b in zip(item_0, item_1):
                        a_val = str(a.item() if hasattr(a, 'item') else a)
                        b_val = str(b.item() if hasattr(b, 'item') else b)
                        if a_val in self.map and b_val in self.map:
                            result.append(self.d[self.map[a_val], self.map[b_val]])
                        else:
                            result.append(0.0)
                    return np.array(result)

                return np.array([])  # Fallback

            def __getitem__(self, ids):
                return self.get_distances(ids)

        calc = Calculator(features, metric)
        ild = IntraListDiversity(k=k, distance_calculator=calc).calc_per_user(reco=recos)
        return ild.mean()

    except Exception as e:
        print(f"ILD failed for {metric}: {e}")
        return np.nan

def _calculate_reverse_gini(recommendations):
    item_counts = recommendations['item_id'].value_counts()
    n = len(item_counts)

    if n < 2:
        print("n < 2, returning 1.0")
        return 1.0

    sorted_counts = np.sort(item_counts.values)
    total = sorted_counts.sum()
    proportions = sorted_counts / total

    i = np.arange(1, n + 1)
    weighted_sum = np.sum(i * proportions)
    gini = (2 * weighted_sum - (n + 1)) / (n - 1)

    gini = np.clip(gini, 0.0, 1.0)
    result = 1 - gini

    print(f"Reverse Gini: {result:.6f}")
    return result

# Display metrics table
def display_metrics_table(metrics_dict, source_name="Model", k=5):
    overall_metrics = ["RMSE", "MAE", "Reverse Gini"]
    topk_metrics = [
        f"HitRate@{k}",
        f"Precision@{k}",
        f"Recall@{k}",
        f"F1@{k}",
        f"NDCG@{k}",
        f"MAP@{k}",
        f"MRR@{k}",
        f"Coverage@{k}",
        f"ILD@{k}_Hamming",
        f"ILD@{k}_Jaccard",
        f"ILD@{k}_Cosine"
    ]
    metric_order = overall_metrics + topk_metrics

    ordered_results = {metric: metrics_dict.get(metric, np.nan) for metric in metric_order}
    df = pd.DataFrame([ordered_results], index=[source_name])
    df_display = df.round(4)

    print(df_display.to_string())

    return df

def save_metrics_table_as_file(metrics_df, filename="metrics_results"):
    #metrics_df.to_csv(f"{filename}.csv")
    metrics_df.to_excel(f"{filename}.xlsx")
    print(f"Saved: {filename}.csv, {filename}.xlsx")

#Load movies file and create binary genre features for ILD calculation.
def load_item_features(items_path, dataset_type="movies"):
    """
    Load items file and create binary genre features for ILD calculation.
    Handles both MovieLens and GoodBooks with pipe-separated genres.
    """
    print(f"Loading item features for {dataset_type}")

    # Robust CSV reading
    items = pd.read_csv(items_path, engine='python', on_bad_lines='skip')

    # CRITICAL: Normalize ID column to 'itemId' name
    if 'itemId' not in items.columns and 'item_id' in items.columns:
        items = items.rename(columns={'item_id': 'itemId'})

    # Normalize IDs: strip whitespace and remove .0 decimals
    items['itemId'] = items['itemId'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    # Remove duplicates AFTER normalization
    items = items.drop_duplicates(subset=['itemId'], keep='first')
    print(f"After deduplication: {len(items)} unique items")

    # Clean genres
    items['genres'] = items['genres'].fillna('Unknown')
    items['genres_list'] = items['genres'].str.split('|')

    # Get unique genres
    all_genres = set()
    for genres in items['genres_list']:
        if isinstance(genres, list):
            for g in genres:
                clean_genre = g.strip() if isinstance(g, str) else ''
                if clean_genre and clean_genre.lower() != 'unknown':
                    all_genres.add(clean_genre)

    all_genres = sorted(list(all_genres))
    print(f"Found {len(all_genres)} genres: {all_genres[:10]}...")

    # Create feature matrix with explicit unique index
    unique_ids = items['itemId'].unique()
    item_features = pd.DataFrame(0, index=unique_ids, columns=all_genres)
    item_features.index.name = 'item_id'

    # Fill matrix safely
    items_with_features = 0
    for idx, row in items.iterrows():
        if isinstance(row['genres_list'], list):
            valid_genres = [g.strip() for g in row['genres_list']
                            if g and g.strip() in all_genres]
            if valid_genres:
                # Use .at for safe single-value assignment
                for genre in valid_genres:
                    item_features.at[row['itemId'], genre] = 1
                items_with_features += 1

    print(f"Processed features for {items_with_features}/{len(items)} items")
    print(f"Matrix shape: {item_features.shape}")
    print(f"Index unique: {item_features.index.is_unique}")
    return item_features

def run_model_comparison(ground_truth_path, sources, catalog, threshold=4.0, k=5,
                         item_features=None, output_prefix="comparison",
                         calculate_ild=True, dataset_type="movies"):
    all_results_df = pd.DataFrame()

    print("Metrics calculations")

    for predictions_path, source_name in sources:
        print(f"\nProcessing '{source_name}'")

        try:
            data = load_and_process_data(ground_truth_path, predictions_path, dataset_type=dataset_type, verbose=False)
        except Exception as e:
            print(f"Error loading data for {source_name}: {e}")
            print("Skipping this model")
            continue

        # Pass the calculate_ild parameter
        metrics = calculate_all_metrics(catalog, data, threshold, k, item_features,
                                       source_name, calculate_ild)
        source_df = display_metrics_table(metrics, source_name, k)
        all_results_df = pd.concat([all_results_df, source_df])


    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add timestamp to output filenames
    save_metrics_table_as_file(all_results_df, f"{output_prefix}_results_{timestamp}")
    #plot_individual_metric_charts(all_results_df, output_dir=f"{output_prefix}_individual_charts_{timestamp}")

    print(f"\nProcessed {len(all_results_df)} model(s) with k={k}")
    return all_results_df

if __name__ == "__main__":
    # Import configuration from separate file
    from DataPaths import (
        THRESHOLD, K, CALCULATE_ILD,
        CATALOG_PATH, CATALOG,
        GROUND_TRUTH,
        MODELS
    )

    # Optional: plot rating distribution
    plot_rating_distribution(
        ground_truth_path=GROUND_TRUTH,
        items_path=CATALOG_PATH,  # This provides the genre data
        output_dir="rating_charts"
    )

    # Conditionally load item features (this is the slow part)
    if CALCULATE_ILD:
        print("Loading item features for ILD calculation")
        ITEM_FEATURES = load_item_features(
            r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv", dataset_type="books"
            #r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv", dataset_type="movies"
        )
    else:
        print("Skipping item feature loading (ILD disabled)")
        ITEM_FEATURES = None
    # Run comparison

    # 🔍 UNIFIED DIAGNOSTICS
    # Ground truth
    _print_data_diagnostics(GROUND_TRUTH, file_label="Ground Truth", threshold=THRESHOLD, is_ground_truth=True)

    # Predictions
    for predictions_path, source_name in MODELS:
        _print_data_diagnostics(predictions_path, file_label=f"Model '{source_name}'", threshold=THRESHOLD,
                                is_ground_truth=False)

    results = run_model_comparison(
        ground_truth_path=GROUND_TRUTH,
        sources=MODELS,
        threshold=THRESHOLD,
        k=K,
        item_features=ITEM_FEATURES,  # Can still provide features, but they won't be used
        output_prefix=f"Johannes, gb 100k, top{K}_comparison",
        calculate_ild=CALCULATE_ILD,  #
        catalog=CATALOG,
        #dataset_type="movies"
        dataset_type="books"
    )