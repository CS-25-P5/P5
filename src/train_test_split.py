from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import ast


# Function to convert genres column
def convert_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)  # convert string to list of dicts
        genre_names = [g['name'] for g in genres_list]
        return "|".join(genre_names)
    except:
        return ""  # empty if parsing fails


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
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    chunksize: int = None,
):
    np.random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)

    # Split for each user
    train_list = []
    test_list = []
    val_list = []


    for user_id, user_ratings in ratings_df.groupby('userId'):
        #Ensure enough ratings
        if len(user_ratings) < 3:
            continue  #skip users with very few ratings
        
        # shuffle
        user_ratings = user_ratings.sample(frac=1, random_state=random_state)

        n = len(user_ratings)
        n_test = max(1, int(n * test_size))
        n_val  = max(1, int(n * val_size))


        # slices
        test = user_ratings.iloc[:n_test]
        val  = user_ratings.iloc[n_test:n_test + n_val]
        train = user_ratings.iloc[n_test + n_val:]

        # ensure train has at least 1 rating
        if len(train) < 1:
            # move 1 item from val into train
            train = pd.concat([train, val.iloc[:1]])
            val = val.iloc[1:]



        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    # Final dataframes
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
CHUNKSIZE = 10000
TEST_SIZE = 0.20


#load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
input_rating_csv = os.path.join(base_dir, "datasets/MovieLens", "ratings.csv")
input_movies_csv = os.path.join(base_dir, "datasets/MovieLens", "movies.csv")

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




# movies_df = standardize_csv(
#     input_csv=input_movies_csv,
#     output_csv=os.path.join(output_dir_rating, f"movies.csv"),
#     col_mapping={"id": "itemId", "title": "title", "genres": "genres"},
#     drop_columns=["adult","belongs_to_collection","budget","homepage","imdb_id","original_language",
#                   "original_title","overview","popularity","poster_path","production_companies",
#                   "production_countries","release_date","revenue","runtime","spoken_languages",
#                   "status","tagline","video","vote_average","vote_count"]
# )

# movies_df['genres'] = movies_df['genres'].apply(convert_genres)
# movies_df = movies_df[['itemId', 'title', 'genres']]
# movies_df = movies_df.sort_values(by="itemId").reset_index(drop=True)
# movies_df.to_csv(os.path.join(output_dir_rating, "movies.csv"), index=False)

split_ratings(
    ratings_df,
    output_dir=output_dir,
    dataset_name="movies",
    test_size=0.1,
    val_size=0.1,
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




# standardize_csv(
#     input_csv=input_movies_csv,
#     output_csv=os.path.join(output_dir_rating, f"books.csv"),
#     col_mapping={"book_id": "itemId", "book_title": "title", "genres": "genres"},
#     drop_columns=["title_ex","book_series", "book_authors", "book_score", "book_rating", 
#                   "book_rating_obj","book_rating_count", "book_review_count", 
#                   "book_desc", "tags", "FE_text", "book_desc_tags_FE", "ratings_1",
#                   "ratings_2","ratings_3","ratings_4","ratings_5","book_edition",
#                   "book_format","original_publication_year","language_code", "book_pages",
#                   "book_pages_obj","books_count","books_count_obj","goodreads_book_id","book_isbn",
#                   "isbn","isbn13","image_url_x","image_url_y","small_image_url"]
# )


split_ratings(
    ratings_df,
    output_dir=output_dir,
    dataset_name="books",
    test_size=0.1,
    val_size=0.1,
    chunksize = CHUNKSIZE,
)



