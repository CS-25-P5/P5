import pandas as pd
import os

def check_overlap(ground_truth_file, mf_test_file, movies_file, user_col='userId', item_col='itemId'):
    # Load datasets
    ground_truth = pd.read_csv(ground_truth_file)
    mf_test = pd.read_csv(mf_test_file)
    movies = pd.read_csv(movies_file)
    
    # Map titles to itemId in mf_test
    title_to_id = dict(zip(movies['title'], movies['itemId']))
    mf_test['itemId'] = mf_test['title'].map(title_to_id)
    
    # Drop rows where mapping failed
    mf_test = mf_test.dropna(subset=['itemId'])
    mf_test['itemId'] = mf_test['itemId'].astype(int)
    
    # Merge to find overlap
    overlap = pd.merge(ground_truth, mf_test, on=[user_col, item_col])
    
    # Compute summary
    summary = {
        "total_ground_truth": len(ground_truth),
        "total_mf_test": len(mf_test),
        "overlap_count": len(overlap),
        "overlap_percentage": len(overlap) / len(ground_truth) * 100
    }
    
    # Print summary
    print(f"Total ground-truth entries: {summary['total_ground_truth']}")
    print(f"Total MF test entries: {summary['total_mf_test']}")
    print(f"Number of overlapping entries: {summary['overlap_count']}")
    print(f"Percentage of ground-truth entries present in MF test: {summary['overlap_percentage']:.2f}%")
    
    return summary, overlap

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
ground_truth = os.path.join(base_dir, "../datasets/mmr_data", "books_ratings_10000_train.csv")
mf_test = os.path.join(base_dir, "../datasets/mmr_data/books/2025-12-06_11-50-24/mf_test_10000_predictions.csv")
# mf_test = os.path.join(base_dir, "../evaluation/movie/mf_test_100000_predictions.csv")
movies_file = os.path.join(base_dir, "../datasets/GoodBooks/books.csv")

summary, overlap_df = check_overlap(ground_truth, mf_test, movies_file)
print(overlap_df.head())
