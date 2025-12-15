import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Dataset configuration
DATASETS = {
    "MovieLens 100K": {
        "ground_truth_path": r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_movies_test.csv",
        "user_col": "userId",
        "item_col": "movieId",
        "rating_col": "rating",
        "expected_scale": (1, 5)
    },
    "GoodBooks 100K": {
        "ground_truth_path": r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_goodbooks_test.csv",
        "user_col": "user_id",
        "item_col": "itemId",
        "rating_col": "rating",
        "expected_scale": (1, 5)
    },
    "MovieLens 1M": {
        "ground_truth_path": r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_1M_movies_test.csv",
        "user_col": "userId",
        "item_col": "movieId",
        "rating_col": "rating",
        "expected_scale": (1, 5)
    }
}

plt.style.use("seaborn-v0_8-whitegrid")
FIGURE_SIZE = (10, 6)
OUTPUT_DIR = Path("./rating_distribution_plots")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_ground_truth_data(base_path, user_col, item_col, rating_col):
    """Load CSV data with flexible path handling."""
    path = Path(base_path)

    if path.is_file() and path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        possible_files = ["interactions.csv", "ground_truth.csv", "test.csv", "ratings.csv"]
        df = None
        for file_name in possible_files:
            file_path = path / file_name
            if file_path.exists():
                df = pd.read_csv(file_path)
                break
        if df is None:
            csv_files = list(path.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError(f"No CSV files found in {path}")

    required_cols = [user_col, item_col, rating_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}. Available: {df.columns.tolist()}")

    df = df[[user_col, item_col, rating_col]].copy()
    df.rename(columns={
        user_col: "user_id_normalized",
        item_col: "item_id_normalized",
        rating_col: "rating"
    }, inplace=True)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    return df


def create_rating_distribution_plot(df, dataset_name, scale=(1, 5), bar_width_factor=0.6):
    """
    Single plot showing rating frequency with adjustable column spacing.
    """
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    ratings = df["rating"]
    rating_counts = ratings.value_counts().sort_index()
    rating_percentages = (rating_counts / len(ratings) * 100).round(1)

    # Calculate bar width based on minimum spacing
    rating_values = sorted(rating_counts.index)
    rating_diffs = np.diff(rating_values)
    min_diff = min(rating_diffs) if len(rating_diffs) > 0 else 1.0
    bar_width = min_diff * bar_width_factor

    # Use distinct colors for each bar
    colors = plt.cm.Set3(np.linspace(0, 1, len(rating_counts)))

    # Create bars
    bars = ax.bar(rating_counts.index, rating_counts.values,
                  color=colors, edgecolor="black", alpha=0.8,
                  width=bar_width, align="center")

    # INCREASED FONT SIZE for percentage labels (was 10, now 12)
    for bar, pct in zip(bars, rating_percentages.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(rating_counts.values) * 0.01,
                f"{pct}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Set labels and title
    ax.set_xlabel("rating", fontsize=12, fontweight="bold")
    ax.set_ylabel("count", fontsize=12, fontweight="bold")
    ax.set_title(f"Rating Distribution: {dataset_name}\n({len(ratings):,} total ratings)",
                 fontsize=14, fontweight="bold")

    # Center x-axis labels
    ax.set_xticks(rating_values)
    ax.set_xticklabels([f"{x:.1f}" for x in rating_values], ha="center")

    # Add padding on both sides
    padding = min_diff * 0.8
    ax.set_xlim(min(rating_values) - padding, max(rating_values) + padding)

    # Add grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("RATING DISTRIBUTION ANALYSIS")
    print("=" * 60)

    all_data = {}

    for dataset_name, config in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")
        try:
            df = load_ground_truth_data(
                config["ground_truth_path"],
                config["user_col"],
                config["item_col"],
                config["rating_col"]
            )

            # Calculate density and sparsity
            total_ratings = len(df)
            unique_users = df["user_id_normalized"].nunique()
            unique_items = df["item_id_normalized"].nunique()
            total_possible = unique_users * unique_items
            density = (total_ratings / total_possible * 100) if total_possible > 0 else 0
            sparsity = 100 - density

            # Validate rating range
            min_rating, max_rating = config["expected_scale"]
            out_of_range = df[(df["rating"] < min_rating) | (df["rating"] > max_rating)]
            if not out_of_range.empty:
                print(f"⚠️  {len(out_of_range)} ratings outside range [{min_rating}, {max_rating}]")

            ratings = df["rating"]
            all_data[dataset_name] = {
                "ratings": ratings,
                "df": df,
                "density": density,
                "sparsity": sparsity
            }

            print(f"✓ Loaded {len(df):,} total ratings")
            print(f"  Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
            print(f"  Mean rating: {ratings.mean():.3f}")
            print(f"  Unique rating values: {sorted(ratings.unique())}")
            print(f"  Dataset density: {density:.4f}%")
            print(f"  Dataset sparsity: {sparsity:.2f}%")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue

    if not all_data:
        print("\nNo datasets loaded. Exiting.")
        return

    print("\n" + "=" * 60)
    print("Generating plots...")

    # Generate individual plots
    for dataset_name, data in all_data.items():
        print(f"  {dataset_name}...")

        # Apply reduced spacing (1/3 less gap) for MovieLens 100K
        bar_width_factor = 0.73 if dataset_name == "MovieLens 100K" else 0.6

        fig = create_rating_distribution_plot(
            data["df"],
            dataset_name,
            bar_width_factor=bar_width_factor
        )

        safe_name = dataset_name.replace(" ", "_").replace("-", "_")
        plt.savefig(OUTPUT_DIR / f"rating_distribution_{safe_name}.svg",
                    format="svg", dpi=300, bbox_inches="tight")
        plt.savefig(OUTPUT_DIR / f"rating_distribution_{safe_name}.png",
                    format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Generate comparative plot
    print("\n  Creating comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Rating Distribution Comparison", fontsize=16, fontweight="bold")

    for idx, (dataset_name, data) in enumerate(all_data.items()):
        ax = axes[idx]
        ratings = data["ratings"]

        rating_counts = ratings.value_counts().sort_index()
        rating_percentages = (rating_counts / len(ratings) * 100).round(1)

        # Calculate bar width for this dataset
        rating_values = sorted(rating_counts.index)
        rating_diffs = np.diff(rating_values)
        min_diff = min(rating_diffs) if len(rating_diffs) > 0 else 1.0

        # Apply reduced spacing for MovieLens 100K in comparison too
        bar_width_factor = 0.73 if dataset_name == "MovieLens 100K" else 0.6
        bar_width = min_diff * bar_width_factor

        colors = plt.cm.Set3(np.linspace(0, 1, len(rating_counts)))
        bars = ax.bar(rating_counts.index, rating_counts.values,
                      color=colors, edgecolor="black", alpha=0.8,
                      width=bar_width, align="center")

        # INCREASED FONT SIZE for comparison plot labels (was 9, now 11)
        for bar, pct in zip(bars, rating_percentages.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(rating_counts.values) * 0.01,
                    f"{pct}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(f"{dataset_name}\n({len(ratings):,} ratings)", fontsize=11)
        ax.set_xlabel("rating", fontsize=10)
        if idx == 0:
            ax.set_ylabel("count", fontsize=10)

        ax.set_xticks(rating_values)
        ax.set_xticklabels([f"{x:.1f}" for x in rating_values], ha="center", rotation=45)

        padding = min_diff * 0.8
        ax.set_xlim(min(rating_values) - padding, max(rating_values) + padding)

        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rating_distribution_comparison.svg",
                format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "rating_distribution_comparison.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved comparison plot.")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    stats_df = pd.DataFrame({
        dataset_name: {
            "Total Ratings": len(data["ratings"]),
            "Unique Users": data["df"]["user_id_normalized"].nunique(),
            "Unique Items": data["df"]["item_id_normalized"].nunique(),
            "Mean Rating": f"{data['ratings'].mean():.3f}",
            "Median": f"{data['ratings'].median():.3f}",
            "Std Dev": f"{data['ratings'].std():.3f}",
            "Density (%)": f"{data['density']:.4f}",
            "Sparsity (%)": f"{data['sparsity']:.2f}",
            "Rating Values": str(sorted([f"{x:.1f}" for x in data['ratings'].unique()]))
        }
        for dataset_name, data in all_data.items()
    }).T

    print(stats_df.to_string())

    # Save to Excel
    excel_path = OUTPUT_DIR / "rating_statistics.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Statistics")

        for dataset_name, data in all_data.items():
            safe_name = dataset_name.replace(" ", "_").replace("-", "_")
            distribution = data["ratings"].value_counts().sort_index()
            distribution_pct = (distribution / len(data["ratings"]) * 100).round(1)

            pd.DataFrame({
                "Rating": distribution.index,
                "Count": distribution.values,
                "Percentage": distribution_pct.values
            }).to_excel(writer, sheet_name=f"{safe_name}_dist", index=False)

    print(f"\nDetailed statistics saved to: {excel_path}")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()