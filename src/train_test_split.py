from sklearn.model_selection import train_test_split
import os
import pandas as pd


chunksize = 10000
test_size = 0.20


#load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file_path = os.path.join(base_dir, "datasets", "ratings.csv")

ratings = pd.read_csv(ratings_file_path, nrows=chunksize)


train_list = []
test_list = []


for user_id, group in ratings.groupby('userId'):
  train_ratings, test_ratings = train_test_split(group, test_size=test_size, random_state=42)
  train_list.append(train_ratings)
  test_list.append(test_ratings)



train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)  

# build full file path
base_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(base_dir,"datasets", "ratings_train.csv")
test_file_path = os.path.join(base_dir,"datasets", "ratings_test.csv" )


train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)

print(f"Test and Train dataset genreated, with {chunksize} number of rows")

