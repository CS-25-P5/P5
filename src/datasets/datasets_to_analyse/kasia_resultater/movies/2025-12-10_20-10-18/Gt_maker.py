import pandas as pd

# Read the CSV file
df = pd.read_csv('mf_test_100000_predictions.csv')

# Select only the columns you want
columns_to_keep = ['userId', 'itemId', 'true_rating']
filtered_df = df[columns_to_keep]

filtered_df = filtered_df.rename(columns={'true_rating': 'rating'})

# Save to new CSV file
filtered_df.to_csv('gt_ratings.csv', index=False)

print("Successfully extracted columns to filtered_ratings.csv")