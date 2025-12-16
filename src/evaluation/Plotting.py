import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import pandas as pd
import re
import glob


def plot_individual_metric_charts(df_metrics, output_dir="metric_charts"):
    """
    Create bar charts for each metric with improved layout to prevent overlapping.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_models = len(df_metrics.index)
    fig_width = max(8, num_models * 0.5 + 5.5)

    for metric in df_metrics.columns:
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        values = df_metrics[metric].values
        model_names = df_metrics.index

        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        valid_names = model_names[valid_mask]

        if len(valid_values) == 0:
            print(f"Warning: no valid values for metric '{metric}', skipping chart")
            continue

        bars = ax.bar(range(len(valid_names)), valid_values,
                      color=plt.cm.Set3(np.linspace(0, 1, len(valid_names))),
                      width=0.8)

        show_all_labels = len(valid_names) <= 10
        max_val = max(valid_values)

        for bar in bars:
            height = bar.get_height()
            offset = max_val * 0.01

            if show_all_labels or height >= np.sort(valid_values)[-5:][0]:
                ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, rotation=45)

        ax.set_ylim(0, max_val * 1.20)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sources', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        if len(valid_names) <= 8:
            rotation, fontsize = 45, 10
        elif len(valid_names) <= 15:
            rotation, fontsize = 60, 9
        else:
            rotation, fontsize = 75, 8

        ax.set_xticks(range(len(valid_names)))
        ax.set_xticklabels(valid_names, rotation=rotation, ha='right', fontsize=fontsize)

        filename = os.path.join(output_dir, f"{metric.replace(' ', '_')}_chart.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Individual metric charts saved in '{output_dir}' directory")


def plot_rating_distribution(ground_truth_path, items_path, output_dir="rating_charts"):
    """
    Create a bar chart of rating distribution from ground truth data.
    """
    gt = pd.read_csv(ground_truth_path)
    items = pd.read_csv(items_path, engine='python', on_bad_lines='skip')
    items['genres'] = items['genres'].fillna('Unknown')

    rating_counts = gt['rating'].value_counts().sort_index()
    rating_percentages = (rating_counts / len(gt) * 100).round(1)

    num_users = gt["userId"].nunique()
    item_col = "itemId" if "itemId" in gt.columns else "movieId"
    num_items = gt[item_col].nunique()
    total_ratings = len(gt)
    rating_sparsity = (total_ratings / (num_users * num_items)) * 100

    if 'genres_list' not in items.columns:
        items['genres_list'] = items['genres'].str.split('|')
    all_genres = set(g for genres in items['genres_list'] if isinstance(genres, list)
                     for g in genres if g and g.strip().lower() != 'unknown')
    num_genres = len(all_genres)
    items_with_genres = (items['genres'] != 'Unknown').sum()
    genre_coverage = (items_with_genres / len(items)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
                  color=plt.cm.Set3(np.linspace(0, 1, len(rating_counts))))

    for bar, percentage in zip(bars, rating_percentages.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(rating_counts.values) * 0.01,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    dataset_name = "Books" if "book" in items_path.lower() else "Movies"
    ax.set_title(f'Rating Distribution - {dataset_name} Ground Truth', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

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


# NEW FUNCTION: Plot metrics vs K
# NEW FUNCTION: Plot metrics vs K
def plot_metrics_vs_k_from_directory(results_dir, metrics_to_plot=['NDCG', 'HitRate', 'ILD_Cosine'],
                                     output_subdir="k_comparison_charts"):
    """
    Create line plots showing how metrics vary with K (1-10) from a directory of Excel files.
    """
    # Create output directory
    output_dir = os.path.join(results_dir, output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all Excel files with the consistent naming pattern
    excel_files = sorted(glob.glob(os.path.join(results_dir, "*comparison_results*.xlsx")))

    if not excel_files:
        print(f"❌ No Excel files found in {results_dir}")
        return

    print(f"\n📊 Found {len(excel_files)} Excel files for K-comparison")

    # Extract K values and load data
    k_values = []
    all_results = {}

    for file_path in excel_files:
        filename = os.path.basename(file_path)
        k_match = re.search(r'top(\d+)', filename, re.IGNORECASE)

        if not k_match:
            print(f"⚠ Could not extract K from: {filename}")
            continue

        k = int(k_match.group(1))

        try:
            df = pd.read_excel(file_path, sheet_name=0, index_col=0)

            # Skip if DataFrame is empty
            if df.empty:
                print(f"⚠ Empty DataFrame in {filename}, skipping")
                continue

            # Store only if we haven't seen this K yet (avoid duplicates)
            if k not in k_values:
                k_values.append(k)
                all_results[k] = df
                print(f"✅ Loaded K={k}: {filename}")
            else:
                print(f"⚠ K={k} already loaded, skipping duplicate: {filename}")

        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    if not all_results:
        print("❌ No valid data loaded")
        return

    k_values = sorted(k_values)
    print(f"📊 Successfully loaded K values: {k_values}")

    # Get models from the first file (use intersection of all files for safety)
    model_sets = [set(df.index) for df in all_results.values()]
    models = list(set.intersection(*model_sets))

    if not models:
        print("❌ No common models found across all K files")
        return

    print(f"📈 Found {len(models)} common models: {models}")

    # Plot each metric
    for metric_base in metrics_to_plot:
        metric_data = {model: [] for model in models}

        for k in k_values:
            df = all_results[k]

            # Construct metric name
            if metric_base == 'ILD_Cosine':
                full_metric_name = f'ILD@{k}_Cosine'
            else:
                full_metric_name = f'{metric_base}@{k}'

            if full_metric_name not in df.columns:
                print(f"⚠ '{full_metric_name}' not found in K={k} data")
                for model in models:
                    metric_data[model].append(np.nan)
                continue

            for model in models:
                if model in df.index:
                    value = df.loc[model, full_metric_name]
                    metric_data[model].append(value)
                else:
                    print(f"⚠ Model '{model}' not found in K={k} data")
                    metric_data[model].append(np.nan)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in models:
            values = metric_data[model]
            if not np.all(np.isnan(values)):
                ax.plot(k_values, values, marker='o', label=model, linewidth=2, markersize=6)

        ax.set_title(f'{metric_base} vs K (Top-K Recommendations)', fontsize=14, fontweight='bold')
        ax.set_xlabel('K (Number of recommendations)', fontsize=12)
        ax.set_ylabel(f'{metric_base} Value', fontsize=12)
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        filename = os.path.join(output_dir, f'{metric_base}_vs_K.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved {metric_base} plot")

    print(f"\n📊 All K-comparison plots saved to {output_dir}/")