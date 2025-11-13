import pandas as pd
import numpy as np
from DataHandler import DataHandler


def calculate_gini_index(predictions_df: pd.DataFrame) -> float:
    """
    Gini Index: Measures inequality in item recommendation frequency.
    Range: 0.0 (perfect equality) to 1.0 (one item dominates)
    """
    item_counts = predictions_df['title'].value_counts().sort_values()
    n = len(item_counts)

    if n < 2:
        return 0.0

    # Gini formula: G = Σ(2i - n - 1) × x_i / (n × Σx_i)
    index = np.arange(1, n + 1)
    gini = np.sum((2 * index - n - 1) * item_counts.values) / (n * item_counts.sum())

    return gini

# XXXXXXXXXXXXXXXXX
# Test

# Define parameters (k for context, though Gini is system-wide)
k = 5

# Initialize DataHandler
data_handler = DataHandler()

# Get all predictions
all_predictions = data_handler.predictions

# Calculate Gini Index
print("Gini Index - System Diversity Metric")
print("=" * 60)

gini_score = calculate_gini_index(all_predictions)

# Show most recommended items for context
print(f"\nTop {min(10, len(all_predictions['title'].unique()))} Most Recommended Items:")
item_counts = all_predictions['title'].value_counts().head(10)
print(item_counts.to_string())

print(f"\nCatalog Statistics:")
print(f"- Total unique items in catalog: {len(all_predictions['title'].unique())}")
print(f"- Total recommendations made: {len(all_predictions)}")

print(f"\nGini Index Results:")
print(f"- Gini Index: {gini_score:.4f}")

#print(f"\nInterpretation:")
#print(f"- Range: 0.0 (perfect equality) to 1.0 (one item dominates)")
#print(f"- Low (<0.30): High diversity, low popularity bias ✅")
#print(f"- Medium (0.30-0.60): Moderate concentration ⚠️")
#print(f"- High (>0.60): Strong filter bubble risk ❌")
#print(f"- Your system: {gini_score:.4f} ({'Low' if gini_score < 0.3 else 'Medium' if gini_score < 0.6 else 'High'} concentration)")