import pandas as pd
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_path = os.path.join(base_path, "datasets", "ratings_test_titles.csv")
mf_path = os.path.join(base_path, "datasets", "mf_test_predictions.csv")

test = pd.read_csv(test_path)  # has 'title'
mf = pd.read_csv(mf_path)      # has 'movie'

def clean_title(s):
    return s.strip().lower()

test_titles = set(test['title'].apply(clean_title))
mf_titles = set(mf['movie'].apply(clean_title))

# Titles that exist in both
common_titles = test_titles & mf_titles
print(f"Number of common titles: {len(common_titles)}")

# Titles in MF predictions not in test set
missing_in_test = mf_titles - test_titles
print(f"MF titles not in test set ({len(missing_in_test)}):", list(missing_in_test)[:10])

# Titles in test set not in MF predictions
missing_in_mf = test_titles - mf_titles
print(f"Test titles not in MF predictions ({len(missing_in_mf)}):", list(missing_in_mf)[:10])