import pandas as pd

# Load your files
gt = pd.read_csv(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movies_ratings_100000_test.csv")
pred = pd.read_csv(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\mf_test_predictions.csv")

# Check for common user-item pairs
common_users = set(gt['userId'].astype(str)) & set(pred['userId'].astype(str))
common_items = set(gt['itemId'].astype(str)) & set(pred['title'].astype(str))

print(f"Common users: {len(common_users)}")
print(f"Common items: {len(common_items)}")

# See if there are any exact matches
gt_tuples = set(zip(gt['userId'].astype(str), gt['itemId'].astype(str)))
pred_tuples = set(zip(pred['userId'].astype(str), pred['title'].astype(str)))
matches = gt_tuples & pred_tuples
print(f"Matching user-item pairs: {len(matches)}")