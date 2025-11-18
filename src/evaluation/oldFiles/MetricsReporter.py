import pandas as pd
import numpy as np
import os
from DataHandler import DataHandler
from Coverage import calculate_topK_item_coverage
from IntraListDiversity import user_intra_list_diversity
from MAP import user_average_precision
from NDCG import user_ndcg
from Rmse_Mae import calculate_accuracy_metrics
from ReverseGini import calculate_gini_index
import matplotlib.pyplot as plt

def calculate_all_metrics(data_handler, threshold=4.0, k=5, item_features=None):
    results = {}
    all_predictions = data_handler.predictions
    ground_truth_data = data_handler.ground_truth

    print("Calculating RMSE and MAE")
    mae, rmse, _ = calculate_accuracy_metrics(all_predictions, ground_truth_data)
    results["RMSE"] = rmse
    results["MAE"] = mae

    print(f"Calculating Precision@{k}, Recall@{k}, F1@{k}")
    merged_topk = data_handler.get_topk_predictions(k, threshold)
    relevant_counts = data_handler.get_relevant_counts_per_user(threshold)

    precision_at_k = merged_topk.groupby("userId")["true_relevant"].mean()
    tp_topk = merged_topk.groupby("userId")["true_relevant"].sum()
    recall_at_k = (tp_topk / relevant_counts).fillna(0)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    results["Precision"] = precision_at_k.mean()
    results["Recall"] = recall_at_k.mean()
    results["F1"] = f1_at_k.fillna(0).mean()

    print(f"Calculating NDCG@{k}")
    per_user_ndcg = merged_topk.groupby("userId").apply(user_ndcg, k=k, include_groups=False)
    results["NDCG"] = per_user_ndcg.mean()

    print(f"Calculating MAP@{k}")
    merged_full = data_handler.get_merged_data_for_standard_metrics(threshold)
    per_user_ap = merged_full.groupby("userId").apply(
        user_average_precision, threshold=threshold, relevant_counts=relevant_counts, include_groups=False
    )
    results["MAP"] = per_user_ap.mean()

    print(f"Calculating MRR@{k}")
    merged_topk_sorted = merged_topk.sort_values(["userId", "rating_pred"], ascending=[True, False])

    def user_mrr(group):
        for rank, is_relevant in enumerate(group["true_relevant"], 1):
            if is_relevant:
                return 1.0 / rank
        return 0.0

    per_user_mrr = merged_topk_sorted.groupby("userId").apply(user_mrr, include_groups=False)
    results["MRR"] = per_user_mrr.mean()

    print(f"Calculating Coverage@{k}")
    coverage_percent, _, _ = calculate_topK_item_coverage(all_predictions, ground_truth_data, k)
    results["Coverage"] = coverage_percent

    print(f"Calculating ILD@{k}")
    if item_features is not None and not item_features.empty:
            per_user_ild = merged_topk.groupby("userId").apply(
                user_intra_list_diversity, item_features_df=item_features, k=k, include_groups=False
            )
            results["ILD"] = per_user_ild.dropna().mean()
    else:
        print("Warning: No item features provided, skipping ILD")
        results["ILD"] = np.nan

    print("Calculating Reverse Gini (Gini Index)")
    gini_score = calculate_gini_index(all_predictions)
    results['Reverse Gini'] = gini_score

    return results

def display_metrics_table(metrics_dict, source_name="Model"):
    metric_order = ["RMSE", "MAE", "Precision", "Recall", "F1", "NDCG", "MAP", "MRR", "Coverage", "ILD", "Reverse Gini"]
    ordered_results = {metric: metrics_dict.get(metric, np.nan) for metric in metric_order}

    df = pd.DataFrame([ordered_results], index=[source_name])
    df_display = df.round(4)

    print("Recommendation evaluation results")
    print(df_display.to_string())

    return df

def save_metrics_table_as_file(metrics_dict, source_name="Model", filename="metrics_results"):
    metric_order = ["RMSE", "MAE", "Precision", "Recall", "F1", "NDCG", "MAP", "MRR", "Coverage", "ILD", "Reverse Gini"]
    ordered_results = {metric: metrics_dict.get(metric, np.nan) for metric in metric_order}

    df = pd.DataFrame([ordered_results], index=[source_name])
    df_display = df.round(4)

    # Save as CSV
    csv_file = f"{filename}.csv"
    df_display.to_csv(csv_file)
    print(f"CSV saved: {csv_file}")

    # Save as Excel
    excel_file = f"{filename}.xlsx"
    df_display.to_excel(excel_file)
    print(f"Excel saved: {excel_file}")

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


def run_model_comparison(ground_truth_path, sources, threshold=4.0, k=5, item_features=None,
                         output_prefix="comparison"):
    all_results_df = pd.DataFrame()

    print("Running comparison")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Number of models to compare: {len(sources)}")

    for predictions_path, source_name in sources:
        print(f"\n📊 Processing '{source_name}'...")
        print(f"   Path: {predictions_path}")

        try:
            data_handler = DataHandler(ground_truth_path=ground_truth_path,
                                       predictions_path=predictions_path)

            metrics = calculate_all_metrics(data_handler, threshold=threshold,
                                            k=k, item_features=item_features)

            source_df = display_metrics_table(metrics, source_name)
            all_results_df = pd.concat([all_results_df, source_df])

        except Exception as e:
            print(f"Error processing {source_name}: {e}")
            continue

    if all_results_df.empty:
        print("\nNo models were processed successfully!")
        return None

    print("Generating comparison outputs")

    # Save results
    save_metrics_table_as_file(all_results_df, filename=f"{output_prefix}_results")

    plot_individual_metric_charts(all_results_df,
                                  output_dir=f"{output_prefix}_individual_charts")

    # Print summary
    print("COMPARISON SUMMARY")
    print(f"Successfully processed: {len(all_results_df)} model(s)")
    print(f"Output files saved with prefix: '{output_prefix}'")
    print(f"Files created:")
    print(f"  - {output_prefix}_results.csv")
    print(f"  - {output_prefix}_results.xlsx")
    print(f"  - {output_prefix}_chart.png")
    print(f"  - {output_prefix}_individual_charts/ (directory)")

    return all_results_df

# Configuration
threshold = 4.0
k = 5
ground_truth_path = r"/datasets/ratings_test_titles2.csv"

# Define models to compare: (predictions_path, display_name)
sources = [
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "MMR"),
]

# Load item features
item_features = pd.DataFrame({
    'title': ['The Matrix', 'Toy Story', 'Inception', 'Joker', 'Interstellar'],
    'Sci-Fi': [1, 0, 1, 0, 1],
    'Animation': [0, 1, 0, 0, 0],
    'Drama': [0, 0, 0, 1, 0],
    'Action': [1, 0, 1, 0, 1],
}).set_index('title')
print("Using hardcoded item features. Load from CSV file in production.")

# Run comparison
results_df = run_model_comparison(
    ground_truth_path=ground_truth_path,
    sources=sources,
    threshold=threshold,
    k=k,
    item_features=item_features,
    output_prefix="model_comparison"
)