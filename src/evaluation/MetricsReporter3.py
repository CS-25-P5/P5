import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import calc_metrics
from rectools.metrics import (
    MAP, MRR, NDCG, Precision, Recall,
    CatalogCoverage, IntraListDiversity
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator


class DataHandler:
    """Data handler for RecTools format"""

    def __init__(self, ground_truth_path, predictions_path):
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.predictions = pd.read_csv(predictions_path)

        # Prepare RecTools format
        self._prepare_rectools_format()

    def _prepare_rectools_format(self):
        """Convert to RecTools standard format"""
        # Ground truth interactions
        self.interactions_test = self.ground_truth.rename(columns={
            'userId': Columns.User,
            'title': Columns.Item,
            'rating': Columns.Weight
        })

        # Predictions (recommendations)
        self.recommendations = self.predictions.rename(columns={
            'userId': Columns.User,
            'title': Columns.Item,
            'rating': Columns.Rank  # RecTools uses rank for sorting
        })

        # Sort by user and rating (descending)
        self.recommendations = self.recommendations.sort_values(
            [Columns.User, Columns.Rank],
            ascending=[True, False]
        )

    def get_merged_for_accuracy(self):
        """Get data for RMSE/MAE"""
        merged = pd.merge(
            self.predictions,
            self.ground_truth,
            on=['userId', 'title'],
            how='inner',
            suffixes=('_pred', '_gt')
        )
        return merged['rating_gt'].values, merged['rating_pred'].values

    def get_catalog(self):
        """Get all unique items in catalog"""
        return self.ground_truth['title'].unique()


class MetricsReporter:
    """Metrics reporter using RecTools"""

    def __init__(self, data_handler, k=5, threshold=4.0, item_features=None):
        self.data_handler = data_handler
        self.k = k
        self.threshold = threshold
        self.item_features = item_features

    def calculate_all_metrics(self):
        """Calculate all metrics using RecTools"""
        results = {}

        # === ACCURACY METRICS (RMSE, MAE) ===
        print("Calculating RMSE and MAE...")
        y_true, y_pred = self.data_handler.get_merged_for_accuracy()
        results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        results['MAE'] = mean_absolute_error(y_true, y_pred)

        # === RECTOOLS RANKING METRICS ===
        print(f"Calculating ranking metrics @{self.k}...")

        # Prepare interactions (test set with relevance threshold)
        interactions = self.data_handler.interactions_test.copy()
        interactions = interactions[interactions[Columns.Weight] >= self.threshold]

        # Get recommendations
        recommendations = self.data_handler.recommendations.copy()

        # Define metrics
        metrics = {
            f'Precision@{self.k}': Precision(k=self.k),
            f'Recall@{self.k}': Recall(k=self.k),
            f'NDCG@{self.k}': NDCG(k=self.k),
            f'MAP@{self.k}': MAP(k=self.k),
            f'MRR@{self.k}': MRR(k=self.k),
            f'Coverage@{self.k}': CatalogCoverage(k=self.k),
        }

        # Add ILD if item features are available
        if self.item_features is not None:
            try:
                # Prepare features in RecTools format
                features_df = self.item_features.reset_index()
                features_df = features_df.rename(columns={'title': Columns.Item})
                features_indexed = features_df.set_index(Columns.Item)

                # Create distance calculator with item features
                distance_calculator = PairwiseHammingDistanceCalculator(features_indexed)

                # Add ILD metric with the distance calculator
                metrics[f'ILD@{self.k}'] = IntraListDiversity(
                    k=self.k,
                    distance_calculator=distance_calculator
                )
            except Exception as e:
                print(f"Warning: Could not add ILD metric: {e}")

        # Calculate all metrics at once
        rectools_results = calc_metrics(
            metrics=metrics,
            reco=recommendations,
            interactions=interactions,
            catalog=self.data_handler.get_catalog()
        )

        # Extract results
        for metric_name, value in rectools_results.items():
            results[metric_name] = value

        # Calculate F1 manually (not in RecTools by default)
        p = results.get(f'Precision@{self.k}', 0)
        r = results.get(f'Recall@{self.k}', 0)
        results[f'F1@{self.k}'] = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # === CUSTOM METRICS ===
        print("Calculating diversity metrics...")
        results['Overall Coverage'] = self._calculate_overall_coverage()
        results['Reverse Gini'] = self._calculate_reverse_gini()

        return results

    def _calculate_overall_coverage(self):
        """Overall catalog coverage"""
        unique_recommended = self.data_handler.predictions['title'].nunique()
        total_items = len(self.data_handler.get_catalog())
        return (unique_recommended / total_items) * 100

    def _calculate_reverse_gini(self):
        """Reverse Gini coefficient (1 - Gini)"""
        item_counts = self.data_handler.predictions['title'].value_counts()
        sorted_counts = np.sort(item_counts.values)
        n = len(sorted_counts)

        if n == 0 or sorted_counts.sum() == 0:
            return 0.0

        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((n - np.arange(1, n + 1) + 0.5) * sorted_counts)) / (n * cumsum[-1]) - 1
        return 1 - gini

    def display_results(self, results):
        """Display formatted results"""
        print("\n" + "=" * 80)
        print("RECOMMENDATION SYSTEM EVALUATION RESULTS")
        print("=" * 80)

        overall_metrics = ['RMSE', 'MAE', 'Overall Coverage', 'Reverse Gini']
        print("\nOVERALL METRICS:")
        for metric in overall_metrics:
            if metric in results:
                print(f"  {metric:20s}: {results[metric]:10.4f}")

        print(f"\nTOP-{self.k} METRICS:")
        topk_metrics = [
            f'Precision@{self.k}', f'Recall@{self.k}', f'F1@{self.k}',
            f'NDCG@{self.k}', f'MAP@{self.k}', f'MRR@{self.k}',
            f'Coverage@{self.k}', f'ILD@{self.k}'
        ]
        for metric in topk_metrics:
            if metric in results:
                print(f"  {metric:20s}: {results[metric]:10.4f}")

        print("=" * 80)

        return pd.DataFrame([results])


def run_model_comparison(ground_truth_path, sources, threshold=4.0, k=5,
                         item_features=None, output_prefix="comparison"):
    """Run metrics for multiple models and generate comparison outputs"""
    all_results_df = pd.DataFrame()

    print("\n" + "=" * 60)
    print("RUNNING COMPARISON ACROSS MULTIPLE MODELS")
    print("=" * 60)

    for predictions_path, source_name in sources:
        print(f"\n📊 Processing '{source_name}'...")

        try:
            # Initialize
            data_handler = DataHandler(
                ground_truth_path=ground_truth_path,
                predictions_path=predictions_path
            )

            # Calculate metrics
            reporter = MetricsReporter(
                data_handler,
                k=k,
                threshold=threshold,
                item_features=item_features
            )
            results = reporter.calculate_all_metrics()

            # Display and collect
            df_results = reporter.display_results(results)
            df_results.index = [source_name]
            all_results_df = pd.concat([all_results_df, df_results])

        except Exception as e:
            print(f"❌ Error processing {source_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results_df.empty:
        print("\n❌ No models were processed successfully!")
        return None

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    csv_file = f"{output_prefix}_results.csv"
    all_results_df.to_csv(csv_file)
    print(f"✓ Results saved to {csv_file}")

    return all_results_df


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Configuration
    ground_truth_path = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

    sources = [
        (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "MMR"),
        # Add more models here
    ]

    # Optional: Load item features for ILD calculation
    item_features = pd.DataFrame({
        'title': ['The Matrix', 'Toy Story', 'Inception', 'Joker', 'Interstellar'],
        'Sci-Fi': [1, 0, 1, 0, 1],
        'Animation': [0, 1, 0, 0, 0],
        'Drama': [0, 0, 0, 1, 0],
        'Action': [1, 0, 1, 0, 1],
    }).set_index('title')

    # Run comparison
    results_df = run_model_comparison(
        ground_truth_path=ground_truth_path,
        sources=sources,
        threshold=4.0,
        k=5,
        item_features=item_features,
        output_prefix="model_comparison"
    )

    print("\n✓ Metrics calculation complete!")