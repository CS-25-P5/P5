from sklearn.model_selection import train_test_split
import os
import pandas as pd


def split_ratings_dataset(
    input_csv: str,
    output_dir: str,
    chunksize: int = None,
    test_size:  float = 0.2,
    random_state: int = 42
):
  # ensure output folder exsits 
  os.makedirs(output_dir, exist_ok=True)

  # Load dataset
  df = pd.read_csv(input_csv, nrows=chunksize)

  # Split for each user
  train_list = []
  test_list = []

  for user_id, group in df.groupby('userId'):
    train_ratings, test_ratings = train_test_split(group, test_size=test_size, random_state=random_state)
    train_list.append(train_ratings)
    test_list.append(test_ratings)

    # Combine splits
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)  

    # Build output paths
    train_file = os.path.join(output_dir, f"ratings_{chunksize}_train.csv")
    test_file = os.path.join(output_dir, f"ratings_{chunksize}_test.csv")

    #save
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)


  print(
      f"Train/Test dataset generated.\n"
      f"Rows: {len(df)} (train={len(train_df)}, test={len(test_df)})\n"
      f"Saved to: {output_dir}"
  )


# Parameters 
CHUNKSIZE = 10000
TEST_SIZE = 0.20


#load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(base_dir, "datasets", "ratings.csv")
output_dir = os.path.join(base_dir, "datasets/mmr_data")

split_ratings_dataset(
  input_csv=input_csv,
  output_dir=output_dir,
  chunksize=CHUNKSIZE,
  test_size= TEST_SIZE
)




# ratings = pd.read_csv(ratings_file_path, nrows=chunksize)


# train_list = []
# test_list = []


# for user_id, group in ratings.groupby('userId'):
#   train_ratings, test_ratings = train_test_split(group, test_size=test_size, random_state=42)
#   train_list.append(train_ratings)
#   test_list.append(test_ratings)



# train_df = pd.concat(train_list).reset_index(drop=True)
# test_df = pd.concat(test_list).reset_index(drop=True)  

# # build full file path
# base_dir = os.path.dirname(os.path.abspath(__file__))
# train_file_path = os.path.join(base_dir,"datasets", "ratings_train.csv")
# test_file_path = os.path.join(base_dir,"datasets", "ratings_test.csv" )


# train_df.to_csv(train_file_path, index=False)
# test_df.to_csv(test_file_path, index=False)

# print(f"Test and Train dataset genreated, with {chunksize} number of rows")
