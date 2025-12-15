import pandas

final_dataset= pandas.read_csv("data/INPUT_datasets/ratingsandgenres_100K_movies_DUPE.csv")

final_dataset[final_dataset.duplicated(keep=False)].head(10)

dups = final_dataset[
    final_dataset.duplicated(subset=["userId", "movieId"], keep=False)
]

print(dups.sort_values(["userId", "movieId"]).head(10))
