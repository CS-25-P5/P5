import pandas as pd
import numpy as np
import os

def load_and_prepare_matrix(
        ratings_file_path,
        item_file_path,
        user_col="userId",
        item_col="itemId",
        rating_col="rating",
        title_col="title",
        feature_col="genre",
        feature_separator="|",
        nrows_items=None
):
    """
    Loads rating + item metadata files and constructs:
      - user-item rating matrix
      - feature map (e.g., genres/categories/ingredients/tags)
      - list of all unique features

    Preserves the original MovieLens workflow but works for ANY dataset.
    """

    # Check if files exist before loading
    if not os.path.exists(ratings_file_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file_path}")

    if not os.path.exists(item_file_path):
        raise FileNotFoundError(f"Item file not found: {item_file_path}")


    # Load the files
    ratings = pd.read_csv(ratings_file_path)
    items = pd.read_csv(item_file_path, nrows=nrows_items)


    #  Merge
    if item_col not in ratings.columns or item_col not in items.columns:
        raise KeyError(f"'{item_col}' must exist in both files.")

    data = pd.merge(ratings, items, on=item_col, how="inner")


    # drop timestamp
    if "timestamp" in data.columns:
        data = data.drop(columns=["timestamp"], errors="ignore")


    # Title
    if title_col not in data.columns:
        # fallback - treats the itemId as title
        title_col = item_col
        data[title_col] = data[item_col].astype(str)

    # drop rows with missing titles
    data = data.dropna(subset=[title_col])


    #  Pivot into USER-ITEM MATRIX
    user_item_matrix = (
        data.pivot(index=user_col, columns=title_col, values=rating_col)
        .fillna(0)
        .sort_index()
    )

    # Ensure items files also has title_col
    if title_col not in items.columns:
        items[title_col] = items[item_col].astype(str)


    #  Feature Map/Genre Map(genre/category/tag/ingredients)
    feature_map = {}

    if feature_col in items.columns:
        for _, row in items.iterrows():
            title = row[title_col]

            raw_features = row.get(feature_col, None)

            if isinstance(raw_features, str):
                feature_set = set(map(str.strip, raw_features.split(feature_separator)))
            else:
                feature_set = set()

            feature_map[title] = feature_set
    else:
        # If dataset has no feature column -> assign empty sets
            feature_map = {
                row[title_col]: set()
                for _, row in items.iterrows()
            }

    # build all unique genre list
    all_features = set()
    for features in feature_map.values():
        all_features.update(features)
    all_features = sorted(all_features)

    return user_item_matrix, feature_map, all_features