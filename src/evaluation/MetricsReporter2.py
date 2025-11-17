import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from rectools.metrics import (
    Precision, Recall, F1Beta, MAP, NDCG, MRR,
    IntraListDiversity, CatalogCoverage, calc_metrics
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator
from DataHandler2 import DataHandler
import matplotlib.pyplot as plt
import os
from rectools.metrics.auc import PartialAUC

def calculate_all_metrics(data_handler, threshold=4.0, k=5, item_features=None):
    """Calculate all metrics using RecTools (aggregated values)"""
    results = {}

    # 1. RMSE & MAE (using FULL dataset)
    print("Calculating RMSE and MAE")
    rmse, mae = _calculate_rating_metrics(data_handler)
    results["RMSE"] = rmse
    results["MAE"] = mae

    # 2. RecTools Metrics (using FILTERED relevant interactions)
    print(f"Calculating top-{k} RecTools metrics")

    # Filter interactions to only include relevant items (rating >= threshold)
    relevant_interactions = data_handler.full_interactions[
        data_handler.full_interactions['weight'].astype(float) >= threshold
        ].copy()

    # Create catalog of all items from ground truth
    catalog = data_handler.full_interactions['item_id'].unique()
    catalog_size = len(catalog)

    # Create dictionary of metrics
    metrics = {
        f'Precision@{k}': Precision(k=k),
        f'Recall@{k}': Recall(k=k),
        f'F1Beta@{k}': F1Beta(k=k, beta=1.0),
        f'MAP@{k}': MAP(k=k),
        f'NDCG@{k}': NDCG(k=k),
        f'MRR@{k}': MRR(k=k),
        f'PartialAUC@{k}': PartialAUC(k=k),  # NEW: RecTools Partial AUC
        f'CatalogCoverage@{k}': CatalogCoverage(k=k),
    }

    # Calculate metrics
    metrics_values = calc_metrics(
        metrics=metrics,
        reco=data_handler.recommendations,
        interactions=relevant_interactions,
        catalog=catalog,
        prev_interactions=None,
    )

    # Assign values
    results[f"Precision@{k}"] = metrics_values[f'Precision@{k}']
    results[f"Recall@{k}"] = metrics_values[f'Recall@{k}']
    results[f"F1@{k}"] = metrics_values[f'F1Beta@{k}']
    results[f"MAP@{k}"] = metrics_values[f'MAP@{k}']
    results[f"NDCG@{k}"] = metrics_values[f'NDCG@{k}']
    results[f"MRR@{k}"] = metrics_values[f'MRR@{k}']
    results[f"PartialAUC@{k}"] = metrics_values[f'PartialAUC@{k}']  # NEW
    results[f"Coverage@{k}"] = metrics_values[f'CatalogCoverage@{k}'] / catalog_size
    results["Overall Coverage"] = results[f"Coverage@{k}"]

    # 3. ILD (Diversity)
    if item_features is not None and not item_features.empty:
        print(f"Calculating ILD@{k}")
        results[f"ILD@{k}"] = _calculate_ild(data_handler, item_features, k)
    else:
        print("Skipping ILD: No item features")
        results[f"ILD@{k}"] = np.nan

    # 4. Reverse Gini (Popularity Bias)
    print("Calculating Reverse Gini")
    results['Reverse Gini'] = _calculate_reverse_gini(data_handler.recommendations)

    return results


def _calculate_rating_metrics(data_handler):
    """Calculate RMSE and MAE using scikit-learn"""
    merged = pd.merge(
        data_handler.predictions,
        data_handler.full_interactions,
        on=['user_id', 'item_id'],
        suffixes=('_pred', '_gt')
    )
    rmse = np.sqrt(mean_squared_error(
        merged['weight_gt'].astype(float),
        merged['weight_pred'].astype(float)
    ))
    mae = mean_absolute_error(
        merged['weight_gt'].astype(float),
        merged['weight_pred'].astype(float)
    )
    return rmse, mae

def _calculate_ild(data_handler, item_features, k):
    """Calculate Intra-List Diversity with RecTools"""
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


def _calculate_reverse_gini(recommendations):
    """Calculate Reverse Gini (1 - Gini coefficient)"""
    item_counts = recommendations['item_id'].value_counts()
    if len(item_counts) < 2: return 1.0

    values = np.sort(item_counts.values)
    n = len(values)
    cumsum = np.cumsum(values)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return 1 - gini


def display_metrics_table(metrics_dict, source_name="Model", k=5):
    """Display metrics with clear separation"""
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


def plot_metrics_comparison(df_metrics, filename="metrics_comparison.png", show_plot=False):
    """Create comparison bar chart"""
    # ... (keep your existing implementation)


def plot_individual_metric_charts(df_metrics, output_dir="metric_charts"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Individual metric charts saved in '{output_dir}' directory")

def run_model_comparison(ground_truth_path, sources, threshold=4.0, k=5,
                         item_features=None, output_prefix="comparison"):
    all_results_df = pd.DataFrame()

    print("Metrics calculations (RecTools)")

    for predictions_path, source_name in sources:
        print(f"\nProcessing '{source_name}'")

        try:
            data_handler = DataHandler(ground_truth_path, predictions_path)
            metrics = calculate_all_metrics(data_handler, threshold, k, item_features)
            source_df = display_metrics_table(metrics, source_name, k)
            all_results_df = pd.concat([all_results_df, source_df])
        except Exception as e:
            print(f"Error: {e}")
            continue

    if all_results_df.empty:
        print("\nNo models processed successfully!")
        return None

    save_metrics_table_as_file(all_results_df, f"{output_prefix}_results")
    plot_individual_metric_charts(all_results_df, output_dir=f"{output_prefix}_individual_charts")

    print(f"\nProcessed {len(all_results_df)} model(s) with k={k}")
    return all_results_df


if __name__ == "__main__":
    # Configuration
    THRESHOLD = 4.0
    K = 5
    GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

    # Models to compare
    MODELS = [
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "MMR"),
        # Add more models as needed: (predictions_path, model_name)
    ]

    # Item features for ILD
    ITEM_FEATURES = pd.DataFrame({
        'title': ['The Matrix', 'Toy Story', 'Inception', 'Joker', 'Interstellar'],
        'Sci-Fi': [1, 0, 1, 0, 1],
        'Animation': [0, 1, 0, 0, 0],
        'Drama': [0, 0, 0, 1, 0],
        'Action': [1, 0, 1, 0, 1],
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