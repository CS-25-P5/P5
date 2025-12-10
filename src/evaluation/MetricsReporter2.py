import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rectools.metrics import (
    Precision, Recall, F1Beta, MAP, NDCG, MRR, HitRate,
    IntraListDiversity, CatalogCoverage, calc_metrics
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator
from DataHandler2 import ProcessedData, load_and_process_data
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

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
    catalog = catalog["item_id"].unique()
    catalog_size = 1682 # change when i get the correct movielens file

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
        unique_recommended_items = data_handler.recommendations['item_id'].nunique()
        results["Overall Coverage"] = unique_recommended_items / catalog_size
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
        results["Overall Coverage"] = np.nan

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

    # Reverse Gini (Popularity Bias)
    print(f"Calculating Reverse Gini for {model_name}")
    results['Reverse Gini'] = _calculate_reverse_gini(data_handler.recommendations)

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
                self.d = pairwise_distances(f.values, metric=m)
                self.ids = f.index.tolist()
                self.map = {id_: i for i, id_ in enumerate(self.ids)}

            def get_distances(self, ids):
                # ✅ Handle tuple of two arrays (rectools passes this way)
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

# Calculate Reverse Gini (1-gini)
def _calculate_reverse_gini(recommendations):
    item_counts = recommendations['item_id'].value_counts()
    if len(item_counts) < 2: return 1.0

    values = np.sort(item_counts.values)
    n = len(values)
    cumsum = np.cumsum(values)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return 1 - gini


# Display metrics table
def display_metrics_table(metrics_dict, source_name="Model", k=5):
    overall_metrics = ["RMSE", "MAE", "Overall Coverage", "Reverse Gini"]
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
    metrics_df.to_csv(f"{filename}.csv")
    metrics_df.to_excel(f"{filename}.xlsx")
    print(f"Saved: {filename}.csv, {filename}.xlsx")


# function for creating charts of calculated metrics
def plot_individual_metric_charts(df_metrics, output_dir="metric_charts"):
    # Create charts folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop making plots for each metric
    for metric in df_metrics.columns:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes

        # Create bar chart
        bars = ax.bar(df_metrics.index, df_metrics[metric],
                      color=plt.cm.Set3(np.linspace(0, 1, len(df_metrics.index))))

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)

        # ✅ FIX: Add 15% padding to top of y-axis
        max_height = df_metrics[metric].max()
        if not np.isnan(max_height):
            ax.set_ylim(0, max_height * 1.15)  # 15% padding at top

        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sources', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xticks(range(len(df_metrics.index)))
        ax.set_xticklabels(df_metrics.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save individual chart
        filename = os.path.join(output_dir, f"{metric.replace(' ', '_')}_chart.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Individual metric charts saved in '{output_dir}' directory")


# function for running it all
def run_model_comparison(ground_truth_path, sources, catalog, threshold=4.0, k=5,
                         item_features=None, output_prefix="comparison",
                         calculate_ild=True, dataset_type="movies"):  # Add parameter
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
    plot_individual_metric_charts(all_results_df, output_dir=f"{output_prefix}_individual_charts_{timestamp}")

    print(f"\nProcessed {len(all_results_df)} model(s) with k={k}")
    return all_results_df

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


def plot_rating_distribution(ground_truth_path, items_path, output_dir="rating_charts"):
    """
    Create a bar chart of rating distribution from ground truth data.
    Saves as both PNG and SVG files. Includes rating sparsity and genre count.
    Stats box positioned on the left side.
    """
    # Load data
    gt = pd.read_csv(ground_truth_path)

    # Load items to get genre information
    items = pd.read_csv(items_path, engine='python', on_bad_lines='skip')
    items['genres'] = items['genres'].fillna('Unknown')
    items_with_genres = (items['genres'] != 'Unknown').sum()
    total_items = len(items)

    # Rating distribution
    rating_counts = gt['rating'].value_counts().sort_index()
    rating_percentages = (rating_counts / len(gt) * 100).round(1)

    # Calculate rating sparsity
    num_users = gt["userId"].nunique()
    item_col = "itemId" if "itemId" in gt.columns else "movieId"
    num_items = gt[item_col].nunique()
    total_ratings = len(gt)
    rating_sparsity = (total_ratings / (num_users * num_items)) * 100

    # Calculate genre stats
    if 'genres_list' not in items.columns:
        items['genres_list'] = items['genres'].str.split('|')
    all_genres = set(g for genres in items['genres_list'] if isinstance(genres, list)
                     for g in genres if g and g.strip().lower() != 'unknown')
    num_genres = len(all_genres)
    genre_coverage = (items_with_genres / total_items) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
                  color=plt.cm.Set3(np.linspace(0, 1, len(rating_counts))))

    # Add percentage labels
    for bar, percentage in zip(bars, rating_percentages.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(rating_counts.values) * 0.01,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Styling
    dataset_name = "Books" if "book" in items_path.lower() else "Movies"
    ax.set_title(f'Rating Distribution - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Statistics box (LEFT-SIDED with separate lines for genres)
    stats_text = (f'Total ratings: {total_ratings:,}\n'
                  f'Users: {num_users:,} | Items: {num_items:,}\n'
                  f'Rating sparsity: {rating_sparsity:.3f}%\n'
                  f'Genres: {num_genres}\n'
                  f'Genre coverage: {genre_coverage:.1f}%')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,  # Changed to 0.02 for left side
            fontsize=10, verticalalignment='top', horizontalalignment='left',  # Changed to left align
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    basename = os.path.splitext(os.path.basename(ground_truth_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rating_distribution_{basename}_{timestamp}.svg'),
                bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Rating distribution chart saved to {output_dir}/")
    return rating_counts, rating_percentages

if __name__ == "__main__":
    # Configuration
    THRESHOLD = 4.0  # for the metrics that need to view things in a binary fashion
    K = 5  # recommendations to look at

    # SET THIS TO False TO SKIP ILD AND GENRE LOADING
    CALCULATE_ILD = True # Change to False to skip ILD entirely

    #CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
    CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"
    CATALOG = pd.read_csv(CATALOG_PATH)
    CATALOG = CATALOG.rename(columns={"itemId": "item_id"})

    #Test1
    #GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

    #test2
    #GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\grount_truth_test2.csv"


    #MF - li movoes
    GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movies_ratings_100000_test.csv")
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\movies_ratings_100000_test.csv")

    #MF - li books
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\books_ratings_100000_train.csv")

    #NN - diana
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\ratings_small.csv")
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\GROUNDTRUTH_TEST.csv") # data from diana
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\MMR_evaluation\movies_ratings_100000_test.csv")

    #NN - johannes
    #movies 100k
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\ground_truth")
    #Books
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\ground_truth")
    #movies 1m
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\ground_truth")

    #DPP - movies
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\movies\movies_ratings_100000_test_gt.csv")

    #DPP - books
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\books_ratings_100000_test_gt.csv")

    # Models to compare
    MODELS = [
        #test1
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "Test"),

        #test2
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\test2_predictions.csv", "Test2"),

        #mf
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\ALIGNED_mf_test_predictions.csv", "mf"),

        #MMR - li movies
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\mf_test_100000_predictions.csv", "MF"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\mmr_test_100000_cosine_predictions.csv", "MMR_cosine"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\mmr_test_100000_jaccard_predictions.csv", "MMR_jaccard"),

        #MMR - li books
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\mf_test_100000_predictions.csv", "MF"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\mmr_test_100000_cosine_predictions.csv", "MMR_cosine"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\mmr_test_100000_jaccard_predictions.csv", "MMR_jaccard"),


        #NN
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\predictionNNwithBPR.csv", "NN"),

        #DPP - movies
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\movies\ALIGNED_dpp_train_jaccard_recommendations_movies.csv", "dpp_jaccard"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\movies\ALIGNED_dpp_train_cosine_recommendations_movies.csv", "dpp_cosine"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\movies\mf_train_predictions.csv", "MF"),
        # Add more models: (predictions_path, model_name)

        # DPP - books
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\ALIGNED_dpp_train_jaccard_recommendations.csv","dpp_jaccard"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\ALIGNED_dpp_train_cosine_recommendations.csv","dpp_cosine"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\ALIGNED_mf_train_predictions_books.csv", "MF"),

        #diana data
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_optimizeradam.csv", "One-32-0001"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_optimizeradam.csv","One-32-00003"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_optimizeradam.csv","One-64-0001"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_optimizeradam.csv","One-64-00003"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_optimizeradam.csv","Two-32-0001"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_optimizeradam.csv","Two-32-00003"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_optimizeradam.csv","Two-64-0001"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_optimizeradam.csv","Two-64-00003"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_optimizeradam.csv","Three-32-0001"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_optimizeradam.csv","Three-32-00003"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_optimizeradam.csv","Three-64-0001"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_optimizeradam.csv","Three-64-00003"),

        #MMR nyeste
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\MMR_evaluation\mf_test_100000_predictions.csv", "MMR_MF"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\MMR_evaluation\mmr_test_100000_cosine_predictions", "MMR_Cosine"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\MMR_evaluation\mmr_test_100000_jaccard_predictions.csv", "MMR_Jaccard"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),


        #NN johannes - movies
        #1layer
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
        #
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
        #
        # #2 layers
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
        #
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
        #
        #
        # #3 layers
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
        #
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

        # NN johannes - books
        # 1layer
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),

        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),

        # 2 layers
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),

        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),

        # 3 layers
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

        # NN johannes - movies 1m
        # 1layer
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),

        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),

        # 2 layers
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),

        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),

        # 3 layers
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    ]

    plot_rating_distribution(
        ground_truth_path=GROUND_TRUTH,
        items_path=CATALOG_PATH,  # This provides the genre data
        output_dir="rating_charts"
    )

    # Conditionally load item features (this is the slow part)
    if CALCULATE_ILD:
        print("Loading item features for ILD calculation")
        ITEM_FEATURES = load_item_features(
            r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv", dataset_type="books"  # Must pass this!
            #r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv", dataset_type="movies"
        )
    else:
        print("Skipping item feature loading (ILD disabled)")
        ITEM_FEATURES = None
    # Run comparison

    # 🔍 DEBUG: Check ground truth distribution
    gt_check = pd.read_csv(GROUND_TRUTH)
    print(f"\n{'=' * 60}")
    print("GROUND TRUTH DIAGNOSTIC")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(gt_check)}")
    print(f"Unique users: {gt_check['userId'].nunique()}")

    # Auto-detect item column name
    item_col = 'movieId' if 'movieId' in gt_check.columns else 'itemId'
    print(f"Unique items: {gt_check[item_col].nunique()}")
    print(f"Rating distribution:\n{gt_check['rating'].value_counts().sort_index()}")
    print(f"% ratings ≥ {THRESHOLD}: {(gt_check['rating'] >= THRESHOLD).mean():.1%}")

    # If this is >70%, your threshold is too low
    if (gt_check['rating'] >= THRESHOLD).mean() > 0.7:
        print("⚠️ WARNING: More than 70% of ratings are above threshold!")
        print("  This makes HitRate artificially high. Try threshold=4.0 or 4.5")

    # 🔍 DEBUG: Check one prediction file raw
    sample_pred = pd.read_csv(MODELS[0][0])
    print(f"\n{'=' * 60}")
    print("RAW PREDICTIONS DIAGNOSTIC")
    print(f"{'=' * 60}")
    print(f"Columns: {list(sample_pred.columns)}")
    print(f"First 5 rows:\n{sample_pred.head()}")

    # Check if predictions are sorted by rating (true) vs test_predicted_rating (predicted)
    if 'test_predicted_rating' in sample_pred.columns:
        correlation = sample_pred['rating'].corr(sample_pred['test_predicted_rating'])
        print(f"Correlation between true rating and predicted rating: {correlation:.3f}")
        if correlation > 0.95:
            print("✓ Predictions are well-correlated with true ratings")
        elif correlation < 0.3:
            print("⚠️ WARNING: Low correlation - predictions may be random")

    results = run_model_comparison(
        ground_truth_path=GROUND_TRUTH,
        sources=MODELS,
        threshold=THRESHOLD,
        k=K,
        item_features=ITEM_FEATURES,  # Can still provide features, but they won't be used
        output_prefix=f"top{K}_comparison",
        calculate_ild=CALCULATE_ILD,  #
        catalog=CATALOG,
        dataset_type="books"
        #dataset_type="movies"
    )