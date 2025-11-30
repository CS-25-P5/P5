from sklearn.model_selection import train_test_split
import os
import pandas as pd


def standardize_csv(
    input_csv: str,
    output_csv: str,
    col_mapping: dict = None,
    drop_columns: list = None,
    nrows: int = None,
    map_to_dense : bool = False

):

  df = pd.read_csv(input_csv, nrows=nrows)

  # Strip whitespace from columns to avoid mismatches
  df.columns = df.columns.str.strip()

  existing_mapping = {}

  # rename according to mapping
  for old,new in col_mapping.items():
    if old in df.columns:
      existing_mapping[old] = new

  df.rename(columns=existing_mapping, inplace=True)


  # drop unwanted columns
  if drop_columns:
    for col in drop_columns:
      if col in df.columns:
        df.drop(columns=col, inplace=True)


  # Map userId and itemId to consecutive dense IDS

  if map_to_dense:
    for col in ["userId", "itemId"]:
      if col in df.columns:
        #get the unique values for column
        unique_ids = df[col].unique()

        # Build a mapping from orginal ID to new dense index
        id_to_idx = {original_id: idx for idx, original_id in enumerate(unique_ids)}

        # Apply mapping to dataframe
        df[col] = df[col].map(id_to_idx)



  # save standardized CSV
  os.makedirs(os.path.dirname(output_csv), exist_ok=True)
  df.to_csv(output_csv, index=False)
  print(f"Standardized CSV saved: {output_csv}")


  return df



def split_ratings(
    ratings_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    chunksize: int = None,
):

  os.makedirs(output_dir, exist_ok=True)

  # Split for each user
  train_list = []
  test_list = []
  val_list = []

  min_train_ratings = 1


  for user_id, group in ratings_df.groupby('userId'):
    # Skip users with too few ratings
    n_ratings = len(group)
    if n_ratings < (min_train_ratings + 2):
      continue

    #Shuffle user's ratings
    group = group.sample(frac=1, random_state=random_state)

    # Reserve minimum items for training
    train_ratings = group.iloc[:min_train_ratings]
    # The rest are saved in ramaining for test and validation
    remaining = group.iloc[min_train_ratings:]  


    if len(remaining) > 1:
      test_count = max(1, int(test_size * len(remaining)))
      val_count = max(1, int(val_size * len(remaining)))

      test_ratings = remaining.iloc[:test_count]
      val_ratings = remaining.iloc[test_count:test_count + val_count]
      train_ratings = pd.concat([train_ratings, remaining.iloc[test_count + val_count:]])
    else: # if only one rating left
      train_ratings = pd.concat([train_ratings, remaining])
      val_ratings = pd.DataFrame(columns=group.columns)
      test_ratings = pd.DataFrame(columns=group.columns)

    

    # split into train and test
    train_ratings, test_ratings = train_test_split(group, test_size=test_size, random_state=random_state)

    # split intro train and validation
    train_ratings, val_ratings = train_test_split(train_ratings, test_size=val_size, random_state=random_state)

    train_list.append(train_ratings)
    val_list.append(val_ratings)
    test_list.append(test_ratings)

  # Combine splits
  train_df = pd.concat(train_list).reset_index(drop=True)
  val_df = pd.concat(val_list).reset_index(drop=True)
  test_df = pd.concat(test_list).reset_index(drop=True)

  # Build output paths
  train_file = os.path.join(output_dir, f"{dataset_name}_ratings_{chunksize}_train.csv")
  val_file = os.path.join(output_dir, f"{dataset_name}_ratings_{chunksize}_val.csv")
  test_file = os.path.join(output_dir, f"{dataset_name}_ratings_{chunksize}_test.csv")

  #save
  train_df.to_csv(train_file, index=False)
  val_df.to_csv(val_file, index=False)
  test_df.to_csv(test_file, index=False)


  print(
      f"Train/Test dataset generated.\n"
      f"Rows: {len(ratings_df)} (train={len(train_df)},val={len(val_df)}, test={len(test_df)})\n"
      f"Saved to: {output_dir}"
  )



# Parameters
CHUNKSIZE = 100000
TEST_SIZE = 0.20


movies_cols = [
    "movieId", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


# Load ratings and movies
ratings = pd.read_csv(
  "src/datasets/MovieLens/u.data",
  sep="\t",
  names=["userId", "movieId", "rating", "timestamp"]
)




movies = pd.read_csv(
  "src/datasets/MovieLens/u.item",
  sep="|",
  names=movies_cols,
  encoding="latin-1"
)

#convert the binary genre flages into a genre list
genre_cols = movies_cols[5:]

movies["genres"] = movies[genre_cols].apply(
    lambda row: "|".join([genre for genre, flag in row.items() if flag == 1]),
    axis=1
)

# Keep only the three columns needed to match MovieLens 20M / 32M format
movies = movies[["movieId", "title", "genres"]]



#load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
input_rating_csv = os.path.join(base_dir, "datasets/MovieLens", "ratings.csv")
input_movies_csv = os.path.join(base_dir, "datasets/MovieLens", "movies.csv")

ratings.to_csv(input_rating_csv, index=False)
movies.to_csv(input_movies_csv, index=False)



output_dir = os.path.join(base_dir, "datasets/mmr_data")
output_dir_rating = os.path.join(base_dir, "datasets/MovieLens")




# #Prepare MOVie dataset
ratings_df = standardize_csv(
    input_csv=input_rating_csv,
    output_csv=os.path.join(output_dir_rating, f"ratings_{CHUNKSIZE}_.csv"),
    col_mapping={"userId": "userId", "movieId": "itemId", "rating": "rating"},
    drop_columns=["timestamp"],
    nrows = CHUNKSIZE,
)
# Reload the CSV to ensure standardized column names
ratings_df = pd.read_csv(os.path.join(output_dir_rating, f"ratings_{CHUNKSIZE}_.csv"))
ratings_df.columns = ratings_df.columns.str.strip()
print("Columns after strip:", ratings_df.columns)


standardize_csv(
    input_csv=input_movies_csv,
    output_csv=os.path.join(output_dir_rating, f"movies.csv"),
    col_mapping={"movieId": "itemId", "title": "title", "genres": "genres"},
)


split_ratings(
    ratings_df,
    output_dir=output_dir,
    dataset_name="movies",
    test_size=0.2,
    val_size=0.2,
    chunksize = CHUNKSIZE,
)



# Prepare book dataset

input_rating_csv = os.path.join(base_dir, "datasets/GoodBooks", "ratings.csv")
input_movies_csv = os.path.join(base_dir, "datasets/GoodBooks", "books.csv")

output_dir = os.path.join(base_dir, "datasets/mmr_data")
output_dir_rating = os.path.join(base_dir, "datasets/GoodBooks")


ratings_df = standardize_csv(
    input_csv=input_rating_csv,
    output_csv=os.path.join(output_dir_rating, f"ratings_{CHUNKSIZE}_.csv"),
    col_mapping={"user_id": "userId", "book_id": "itemId", "rating": "rating"},
    nrows = CHUNKSIZE,
    map_to_dense = True
)

# Reload the CSV to ensure standardized column names
ratings_df = pd.read_csv(os.path.join(output_dir_rating, f"ratings_{CHUNKSIZE}_.csv"))
ratings_df.columns = ratings_df.columns.str.strip()
print("Columns after strip:", ratings_df.columns)


standardize_csv(
    input_csv=input_movies_csv,
    output_csv=os.path.join(output_dir_rating, f"books.csv"),
    col_mapping={"book_id": "itemId", "book_title": "title", "genres": "genres"},
    drop_columns=["title_ex","book_series", "book_authors", "book_score", "book_rating", 
                  "book_rating_obj","book_rating_count", "book_review_count", 
                  "book_desc", "tags", "FE_text", "book_desc_tags_FE", "ratings_1",
                  "ratings_2","ratings_3","ratings_4","ratings_5","book_edition",
                  "book_format","original_publication_year","language_code", "book_pages",
                  "book_pages_obj","books_count","books_count_obj","goodreads_book_id","book_isbn",
                  "isbn","isbn13","image_url_x","image_url_y","small_image_url"]
)


split_ratings(
    ratings_df,
    output_dir=output_dir,
    dataset_name="books",
    test_size=0.2,
    val_size=0.2,
    chunksize = CHUNKSIZE,
)



