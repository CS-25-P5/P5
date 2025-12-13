import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import pandas as pd

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
    ax.set_title(f'Rating Distribution - 100K MovieLens Ground Truth', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Statistics box (LEFT-SIDED with separate lines for genres)
    # stats_text = (f'Total ratings: {total_ratings:,}\n'
    #               f'Users: {num_users:,} | Items: {num_items:,}\n'
    #               f'Rating sparsity: {rating_sparsity:.3f}%\n'
    #               f'Genres: {num_genres}\n'
    #               f'Genre coverage: {genre_coverage:.1f}%')
    # ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,  # Changed to 0.02 for left side
    #         fontsize=10, verticalalignment='top', horizontalalignment='left',  # Changed to left align
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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