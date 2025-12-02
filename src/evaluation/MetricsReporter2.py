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
from rectools.metrics.auc import PartialAUC
from datetime import datetime

# Calculate most metrics using RecTools
def calculate_all_metrics(data_handler, threshold=4.0, k=5, item_features=None,
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
    catalog = data_handler.interactions['item_id'].unique()
    catalog_size = len(catalog)

    # Create dictionary of metrics to calculate and set their variables
    metrics = {
        f'Precision@{k}': Precision(k=k),  # Proportion of recommended items that are relevant
        f'Recall@{k}': Recall(k=k),  # Proportion of items that where recommended
        f'F1Beta@{k}': F1Beta(k=k, beta=1.0),  # Harmonic mean of Precision and recall
        f'MAP@{k}': MAP(k=k),  # Considers ranking order of relevant items
        f'NDCG@{k}': NDCG(k=k),  # Rewards relevant items appearing earlier in recommendations
        f'MRR@{k}': MRR(k=k),  # focuses on position of the first relevant item
        f'PartialAUC@{k}': PartialAUC(k=k),  # Area under curve for top-K items
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
        results[f"PartialAUC@{k}"] = metrics_values[f'PartialAUC@{k}']
        results[f"Coverage@{k}"] = metrics_values[f'CatalogCoverage@{k}'] / catalog_size
        results["Overall Coverage"] = results[f"Coverage@{k}"]
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
        results[f"PartialAUC@{k}"] = np.nan
        results[f"Coverage@{k}"] = np.nan
        results["Overall Coverage"] = np.nan

    # intra list diversity (ILD)
    if calculate_ild and item_features is not None and not item_features.empty:
        print(f"Calculating ILD@{k} for {model_name}")
        results[f"ILD@{k}"] = _calculate_ild(data_handler, item_features, k)
    else:
        if calculate_ild:
            print(f"Skipping ILD for {model_name}: No item features")
        else:
            print(f"Skipping ILD for {model_name}: Disabled by user")
        results[f"ILD@{k}"] = np.nan

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
def _calculate_ild(data_handler, item_features, k):
    try:
        # Ensure matching index types
        recommendations = data_handler.recommendations.copy()
        recommendations['item_id'] = recommendations['item_id'].astype(str)

        # Filter to items that exist in feature matrix
        available_items = list(set(recommendations['item_id'].unique()) &
                               set(item_features.index.astype(str)))

        if not available_items:
            print("Warning: No item feature overlap")
            return np.nan

        filtered_features = item_features.loc[available_items]

        # Calculate distances (raw)
        distance_calc = PairwiseHammingDistanceCalculator(filtered_features)
        ild_metric = IntraListDiversity(k=k, distance_calculator=distance_calc)
        ild_per_user = ild_metric.calc_per_user(reco=recommendations)

        # Normalize to 0-1 range
        max_possible_distance = item_features.shape[1]  # Number of genre features
        normalized_ild = ild_per_user.mean() / max_possible_distance

        print(f"Raw ILD: {ild_per_user.mean():.3f}, Normalized ILD: {normalized_ild:.3f}")
        return normalized_ild

    except Exception as e:
        print(f"ILD calculation failed: {e}")
        return np.nan

    except Exception as e:
        print(f"\nILD calculation failed with error: {e}")
        print(f"Recommendations columns: {list(data_handler.recommendations.columns)}")
        print(f"Recommendations dtypes:\n{data_handler.recommendations.dtypes}")
        print(f"Recommendations sample:\n{data_handler.recommendations.head()}")
        print(f"Item features shape: {item_features.shape}")
        print(f"Item features index sample: {list(item_features.index)[:5]}")
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
        f"PartialAUC@{k}",
        f"Coverage@{k}",
        f"ILD@{k}"
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
def run_model_comparison(ground_truth_path, sources, threshold=4.0, k=5,
                         item_features=None, output_prefix="comparison",
                         calculate_ild=True):  # Add parameter
    all_results_df = pd.DataFrame()

    print("Metrics calculations")

    for predictions_path, source_name in sources:
        print(f"\nProcessing '{source_name}'")

        try:
            data = load_and_process_data(ground_truth_path, predictions_path)
        except Exception as e:
            print(f"Error loading data for {source_name}: {e}")
            print("Skipping this model")
            continue

        # Pass the calculate_ild parameter
        metrics = calculate_all_metrics(data, threshold, k, item_features,
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
def load_item_features(movies_path):

    # Load movies data
    print("starting loading")
    movies = pd.read_csv(movies_path)
    movies['itemId'] = movies['itemId'].astype(str)

    # Split pipe-separated genres and get all unique genres
    movies['genres_list'] = movies['genres'].str.split('|')
    all_genres = sorted(set([g for genres in movies['genres_list'] for g in genres]))

    # Create binary feature matrix
    item_features = pd.DataFrame(0, index=movies['itemId'], columns=all_genres)
    item_features.index.name = 'item_id'

    for idx, row in movies.iterrows():
        item_features.loc[row['itemId'], row['genres_list']] = 1

    return item_features

if __name__ == "__main__":
    # Configuration
    THRESHOLD = 4  # for the metrics that need to view things in a binary fashion
    K = 5  # recommendations to look at

    # SET THIS TO False TO SKIP ILD AND GENRE LOADING
    CALCULATE_ILD = False  # Change to False to skip ILD entirely

    #Test1
    #GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

    #MF
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movies_ratings_100000_test.csv")

    #NN
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\ratings_small.csv")
    GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\GROUNDTRUTH_TEST.csv") # data from diana

    #DPP - movies
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\movies\movies_ratings_100000_test_gt.csv")

    #DPP - books
    #GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\books_ratings_100000_test_gt.csv")

    # Models to compare
    MODELS = [
        #test1
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "Test"),

        #mf
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\ALIGNED_mf_test_predictions.csv", "mf"),

        #MMR
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\aligned_mmr_train_cosine_test_recommendations.csv", "mmr"),

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

        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_optimizeradam.csv", "One-32-0001"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_optimizeradam.csv","One-32-00003"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_optimizeradam.csv","One-64-0001"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_optimizeradam.csv","One-64-00003"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_optimizeradam.csv","Two-32-0001"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_optimizeradam.csv","Two-32-00003"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_optimizeradam.csv","Two-64-0001"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_optimizeradam.csv","Two-64-00003"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_optimizeradam.csv","Three-32-0001"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_optimizeradam.csv","Three-32-00003"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_optimizeradam.csv","Three-64-0001"),
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\Predictions_test_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_optimizeradam.csv","Three-64-00003"),
    ]


    # Conditionally load item features (this is the slow part)
    if CALCULATE_ILD:
        print("Loading item features for ILD calculation...")
        ITEM_FEATURES = load_item_features(
            r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
        )
    else:
        print("Skipping item feature loading (ILD disabled)")
        ITEM_FEATURES = None
    # Run comparison
    # Run comparison without ILD

    # 🔍 DEBUG: Check ground truth distribution
    gt_check = pd.read_csv(GROUND_TRUTH)
    print(f"\n{'=' * 60}")
    print("GROUND TRUTH DIAGNOSTIC")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(gt_check)}")
    print(f"Unique users: {gt_check['userId'].nunique()}")
    print(f"Unique items: {gt_check['movieId'].nunique()}")
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
        calculate_ild=False  # Skip ILD calculation
    )