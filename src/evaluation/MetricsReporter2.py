import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rectools.metrics import (
    Precision, Recall, F1Beta, MAP, NDCG, MRR,
    IntraListDiversity, CatalogCoverage, calc_metrics
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator
from DataHandler2 import ProcessedData, load_and_process_data
import matplotlib.pyplot as plt
import os
from rectools.metrics.auc import PartialAUC

# Calculate most metrics using RecTools
def calculate_all_metrics(data_handler, threshold=4.0, k=5, item_features=None, model_name="Unknown"):
    results = {}

    # RMSE & MAE (prediction metrics) - with error handling
    print(f"Calculating RMSE and MAE for {model_name}")
    rmse, mae = _calculate_rating_metrics(data_handler, model_name)
    results["RMSE"] = rmse
    results["MAE"] = mae

    # Top-K Metrics (using filtered relevant interactions)
    print(f"Calculating top-{k} metrics for {model_name}")

    # Filter interactions to only include relevant items (rating >= threshold), Necesarry for ranking metrics
    relevant_interactions = data_handler.full_interactions[
        data_handler.full_interactions['weight'] >= threshold
        ].copy()

    # Create catalog of all items from ground truth
    catalog = data_handler.full_interactions['item_id'].unique()
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
        print("Filling with NaN values and continuing...")
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
    if item_features is not None and not item_features.empty:
        print(f"Calculating ILD@{k} for {model_name}")
        results[f"ILD@{k}"] = _calculate_ild(data_handler, item_features, k)
    else:
        print(f"Skipping ILD for {model_name}: No item features")
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
            data_handler.full_interactions,
            on=['user_id', 'item_id'],
            suffixes=('_pred', '_gt')
        )

        # Check if merge produced any matches
        if merged.empty:
            print(f"\nWarning for {model_name}: No matching user-item pairs found!")
            print(
                f"Predictions have {len(data_handler.predictions)} rows, {data_handler.predictions['item_id'].nunique()} unique items")
            print(
                f"Ground truth has {len(data_handler.full_interactions)} rows, {data_handler.full_interactions['item_id'].nunique()} unique items")
            print(f"Debug - Sample predictions item_ids: {list(data_handler.predictions['item_id'].unique())[:5]}")
            print(
                f"Debug - Sample ground truth item_ids: {list(data_handler.full_interactions['item_id'].unique())[:5]}")
            print("Returning NaN for RMSE/MAE and continuing...\n")
            return np.nan, np.nan

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

        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae

    except Exception as e:
        print(f"\nError in RMSE/MAE calculation for {model_name}: {e}")
        print("   Returning NaN for RMSE and MAE to allow other metrics to continue.\n")
        return np.nan, np.nan


# Calculate Intra-List Diversity with RecTools
def _calculate_ild(data_handler, item_features, k):
    try:
        if 'title' in item_features.columns:
            item_features = item_features.set_index('title')
        item_features.index.name = 'item_id'

        distance_calc = PairwiseHammingDistanceCalculator(item_features)
        ild_metric = IntraListDiversity(k=k, distance_calculator=distance_calc)

        ild_per_user = ild_metric.calc_per_user(
            reco=data_handler.recommendations,
            catalog=data_handler.full_interactions['item_id'].unique()
        )
        return ild_per_user.mean()
    except Exception as e:
        print(f"ILD calculation failed: {e}")
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
        f"Precision@{k}", f"Recall@{k}", f"F1@{k}",
        f"NDCG@{k}", f"MAP@{k}", f"MRR@{k}",
        f"PartialAUC@{k}", f"Coverage@{k}", f"ILD@{k}"
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
    # create charts folder if it doesnt exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop making plots for each metric
    for metric in df_metrics.columns:
        plt.figure(figsize=(8, 6))

        # Create bar chart
        bars = plt.bar(df_metrics.index, df_metrics[metric],
                       color=plt.cm.Set3(np.linspace(0, 1, len(df_metrics.index))))

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}',
                         ha='center', va='bottom')

        plt.title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Sources', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save individual chart
        filename = os.path.join(output_dir, f"{metric.replace(' ', '_')}_chart.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Individual metric charts saved in '{output_dir}' directory")


# function for running it all
def run_model_comparison(ground_truth_path, sources, threshold=4.0, k=5,
                         item_features=None, output_prefix="comparison"):
    all_results_df = pd.DataFrame()

    print("Metrics calculations")

    for predictions_path, source_name in sources:
        print(f"\nProcessing '{source_name}'")

        try:
            data = load_and_process_data(ground_truth_path, predictions_path)  # from datahandler, load data
        except Exception as e:
            print(f"Error loading data for {source_name}: {e}")
            print("Skipping this model")
            continue

        metrics = calculate_all_metrics(data, threshold, k, item_features, source_name)  # calculate all metrics
        source_df = display_metrics_table(metrics, source_name, k)  # store results
        all_results_df = pd.concat([all_results_df, source_df])  # relevant if more than one model is run

    save_metrics_table_as_file(all_results_df, f"{output_prefix}_results")  # save as a file (excel and csv)
    plot_individual_metric_charts(all_results_df, output_dir=f"{output_prefix}_individual_charts")  # create plots

    print(f"\nProcessed {len(all_results_df)} model(s) with k={k}")  # output results
    return all_results_df


if __name__ == "__main__":
    # Configuration
    THRESHOLD = 4.0  # for the metrics that need to view things in a binary fashion
    K = 5  # recommendations to look at
    # GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"
    # GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movies_ratings_100000_test.csv")
    GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\ratings_small.csv")

    # Models to compare
    MODELS = [
        # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "Test"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\ALIGNED_mf_test_predictions.csv", "mf"),
        #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\aligned_mmr_train_cosine_test_recommendations.csv", "mmr"),
        (
        r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\predictionNNwithBPR.csv",
        "NN")
        # Add more models: (predictions_path, model_name)
    ]

    # Item features for ILD (test)
    ITEM_FEATURES = pd.DataFrame({
        "title": ["The Matrix", "Toy Story", "Inception", "Joker", "Interstellar"],
        "Sci-Fi": [1, 0, 1, 0, 1],
        "Animation": [0, 1, 0, 0, 0],
        "Drama": [0, 0, 0, 1, 0],
        "Action": [1, 0, 1, 0, 1],
    })

    # Run comparison
    results = run_model_comparison(
        ground_truth_path=GROUND_TRUTH,
        sources=MODELS,
        threshold=THRESHOLD,
        k=K,
        item_features=ITEM_FEATURES,
        output_prefix=f"rectools_top{K}_comparison"
    )