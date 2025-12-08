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
    

def remove_duplicate_genres(df, genre_column='genres'):

    def removal(genres_str):
        if pd.isna(genres_str) or genres_str == "":
            return ""
        
        # Split, clean, remove duplicates
        genres = str(genres_str).split("|")
        # preserves order, removes duplicates
        unique_genres = list(dict.fromkeys(genres))

        return "|".join(unique_genres)
    
    df[genre_column] = df[genre_column].apply(removal)

    return df


def fix_movie_ratings_mapping_with_links(ratings_df, movies_df, links_df):    
    # Make copies
    ratings = ratings_df.copy()
    movies = movies_df.copy()
    links = links_df.copy()
    
    # Clean links file
    links = links.dropna(subset=['tmdbId', 'movieId'])
    links['tmdbId'] = links['tmdbId'].astype(int).astype(str)
    links['movieId'] = links['movieId'].astype(int)
    
    # Add tmdbId to ratings
    ratings_with_tmdb = ratings.merge(
        links[['movieId', 'tmdbId']],
        left_on='movieId',
        right_on='movieId',
        how='inner'
    )
    
    # Prepare movies data
    movies['id'] = movies['id'].astype(str)
    
    # Step 4: Merge with movies metadata
    final_ratings = ratings_with_tmdb.merge(
        movies,
        left_on='tmdbId',
        right_on='id',
        how='inner'
    )

    
    # Create consistent itemId (use tmdbId)
    final_ratings['itemId'] = final_ratings['tmdbId']
    movies['itemId'] = movies['id']
    
    # Keep only necessary columns
    final_ratings = final_ratings[['userId', 'itemId', 'rating']].copy()
    movies = movies[['itemId', 'title', 'genres']].copy()
    
    print(f"FINAL DATASET:")
    print(f"   Ratings: {len(final_ratings):,} rows")
    print(f"   Movies:  {len(movies):,} unique movies")
    print(f"   Users:   {final_ratings['userId'].nunique():,}")
    
    return final_ratings, movies



def standardize_csv(
    input_csv: str,
    output_csv: str,
    col_mapping: dict = None,
    drop_columns: list = None,
    nrows: int = None,
    #map_to_dense : bool = False

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


    # Remove duplicate user-item pairs
    if 'userId' in df.columns and 'itemId' in df.columns:
        duplicate_count = df.duplicated(subset=['userId', 'itemId']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Removing {duplicate_count} duplicate user-item ratings")
            df = df.drop_duplicates(subset=['userId', 'itemId'], keep='first')

    # Map userId and itemId to consecutive dense IDS
    # if map_to_dense:
    #     for col in ["userId", "itemId"]:
    #         if col in df.columns:
    #             get the unique values for column
    #             unique_ids = df[col].unique()

    #             Build a mapping from orginal ID to new dense index
    #             id_to_idx = {original_id: idx for idx, original_id in enumerate(unique_ids)}

    #             Apply mapping to dataframe
    #             df[col] = df[col].map(id_to_idx)

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
    min_user_ratings: int = 5,
):

    np.random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter users
    user_counts = ratings_df.groupby('userId').size()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    filtered_df = ratings_df[ratings_df['userId'].isin(valid_users)].copy()
    
    print(f"Users: {filtered_df['userId'].nunique()}, Items: {filtered_df['itemId'].nunique()}")
    
    train_list, val_list, test_list = [], [], []
    
    for user_id, user_ratings in filtered_df.groupby('userId'):
        user_ratings = user_ratings.sample(frac=1, random_state=random_state)
        n = len(user_ratings)

        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        
        test = user_ratings.iloc[:n_test]
        val = user_ratings.iloc[n_test:n_test + n_val]
        train = user_ratings.iloc[n_test + n_val:]
        
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
    
    # 4. Combine
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    

    
    # Find items that appear in ALL THREE splits
    train_items = set(train_df['itemId'].unique())
    val_items = set(val_df['itemId'].unique())
    test_items = set(test_df['itemId'].unique())
    
    common_items = train_items.intersection(val_items).intersection(test_items)
    print(f"\nItems in ALL splits: {len(common_items)}")


    print(f"\nFinal splits:")
    print(f"Train: {len(train_df)} ratings, {train_df['itemId'].nunique()} items")
    print(f"Val:   {len(val_df)} ratings, {val_df['itemId'].nunique()} items")
    print(f"Test:  {len(test_df)} ratings, {test_df['itemId'].nunique()} items")
    

    # 8. Save
    train_file = os.path.join(output_dir, f"{dataset_name}_ratings_{chunksize}_train.csv")
    val_file = os.path.join(output_dir, f"{dataset_name}_ratings_{chunksize}_val.csv")
    test_file = os.path.join(output_dir, f"{dataset_name}_ratings_{chunksize}_test.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\nFiles saved to:")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")
    


# Parameters
CHUNKSIZE = 100000
TEST_SIZE = 0.20


#Prepare movie dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
input_rating_csv = os.path.join(base_dir, "datasets/MovieLens", "ratings_small.csv")
input_movies_csv = os.path.join(base_dir, "datasets/MovieLens", "movies_metadata.csv")

output_dir = os.path.join(base_dir, "datasets/mmr_data")
output_dir_rating = os.path.join(base_dir, "datasets/MovieLens")


movies_df = standardize_csv(
    input_csv=input_movies_csv,
    output_csv=os.path.join(output_dir_rating, f"movies.csv"),
    col_mapping={"id": "id", "title": "title", "genres": "genres"},
    drop_columns=["adult","belongs_to_collection","budget","homepage","imdb_id","original_language",
                "original_title","overview","popularity","poster_path","production_companies",
                "production_countries","release_date","revenue","runtime","spoken_languages",
                "status","tagline","video","vote_average","vote_count"]
)

movies_df['genres'] = movies_df['genres'].apply(convert_genres)
movies_df = movies_df.sort_values(by="id").reset_index(drop=True)
movies_df.to_csv(os.path.join(output_dir_rating, "movies.csv"), index=False)


ratings_raw = pd.read_csv(input_rating_csv)
movies_raw = pd.read_csv(os.path.join(base_dir, "datasets/MovieLens", "movies.csv"))
links_raw = pd.read_csv(os.path.join(base_dir, "datasets/MovieLens", "links.csv"))

# Apply the CORRECT fix
ratings_fixed, movies_fixed = fix_movie_ratings_mapping_with_links(
    ratings_df=ratings_raw,
    movies_df=movies_raw,
    links_df=links_raw
)

# Save the fixed versions temporarily
temp_ratings_path = os.path.join(output_dir_rating, "ratings.csv")
final_movies_path = os.path.join(output_dir_rating, "movies.csv")


ratings_fixed.to_csv(temp_ratings_path, index=False)
movies_fixed.to_csv(final_movies_path, index=False)

# #Prepare MOVie dataset
ratings_df = standardize_csv(
    input_csv=temp_ratings_path,
    output_csv=os.path.join(output_dir_rating, f"ratings_{CHUNKSIZE}_.csv"),
    col_mapping={"userId": "userId", "movieId": "itemId", "rating": "rating"},
    drop_columns=["timestamp"],
    nrows = CHUNKSIZE,
)


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
input_books_csv = os.path.join(base_dir, "datasets/GoodBooks", "books.csv")

output_dir = os.path.join(base_dir, "datasets/mmr_data")
output_dir_rating = os.path.join(base_dir, "datasets/GoodBooks")


books_df = standardize_csv(
    input_csv=input_books_csv,
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


books_df = remove_duplicate_genres(books_df)
books_df.to_csv(os.path.join(output_dir_rating, "books.csv"), index=False)

ratings_df = standardize_csv(
    input_csv=input_rating_csv,
    output_csv=os.path.join(output_dir_rating, f"ratings_{CHUNKSIZE}_.csv"),
    col_mapping={"user_id": "userId", "book_id": "itemId", "rating": "rating"},
    nrows = CHUNKSIZE,
)


split_ratings(
    ratings_df,
    output_dir=output_dir,
    dataset_name="books",
    test_size=0.1,
    val_size=0.1,
    chunksize = CHUNKSIZE,
)



