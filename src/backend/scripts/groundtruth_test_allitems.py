
import time
import os
import torch
from torch import nn
import copy
import ast
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pandas
import torch.nn.functional as F
import numpy as np
import random
from sklearn.model_selection import train_test_split

''' Previous way of splitting up dataset to get ground truth
input1 = "data/Output_Predictions_test_1M_movies(MLPwithBPR)/TEST_GROUNDTRUTH.csv"
input2 = "data/Output_Predictions_test_1M_movies(MLPwithGenres)/TEST_GROUNDTRUTH.csv"
input5 = "data/Output_Predictions_test_100K_movies(MLPwithBPR)/TEST_GROUNDTRUTH.csv"
input6 = "data/Output_Predictions_test_100K_movies(MLPwithGenres)/TEST_GROUNDTRUTH.csv"


input3 = "data/Output_Predictions_test_100K_goodbooks(MLPwithBPR)/TEST_GROUNDTRUTH.csv"
input4 = "data/Output_Predictions_test_100K_goodbooks(MLPwithGenres)/TEST_GROUNDTRUTH.csv"

def rewrite(input):
    inputpath = input
    df = pandas.read_csv(inputpath)

    df = df.drop(columns=["timestamp"])
    
    users_unique = df["userId"].unique()
    movies_unique = df["movieId"].unique()

    #Full user and moie column
    full_movies = (pandas.MultiIndex.from_product([users_unique, movies_unique], names = ["userId", "movieId"])
            .to_frame(index=False))

    final = full_movies.merge(df, on=["userId", "movieId"], how="left")
    final = final.sort_values(["userId", "movieId"])

    #Create output filename 
    input_folder = "data/TEST_RECOMMEND_inputfile"
    output_name = "GROUNDTRUTH_alluserandmovies.csv"
    output_path = os.path.join(input_folder, output_name)

    #Save
    final.to_csv(output_path, index=False)


def rewritebook(input):
    inputpath = input
    df = pandas.read_csv(inputpath)

    users_unique = df["userId"].unique()
    books_unique = df["itemId"].unique()

    full_books = (pandas.MultiIndex.from_product([users_unique, books_unique], names = ["userId", "itemId"])
            .to_frame(index=False))
    
    final = full_books.merge(df, on=["userId", "itemId"], how="left")
    final = final.sort_values(["userId", "itemId"])

    #Create output filename 
    input_folder = "data/TEST_RECOMMEND_inputfile"
    output_name = "GROUNDTRUTH_alluserandbooks.csv"
    output_path = os.path.join(input_folder, output_name)

    #Save
    final.to_csv(output_path, index=False)


#rewrite(input2)
#rewrite(input1)
#rewrite(input5)
#rewrite(input6)

#rewritebook(input3)
#rewritebook(input4)

'''

#STEP 0) Chop off the books dataset from 5M to 100 K 

goodbooks_largefile = pandas.read_csv("data/IMPORTANTdatasets/ratings_5M_goodbooks.csv", nrows=100_000)
goodbooks_largefile.to_csv("data/IMPORTANTdatasets/ratings_100K_goodbooks.csv", index=False)

#STEP1) Make goodbooks and movies genre file

ratings_book = pandas.read_csv("data/IMPORTANTdatasets/ratings_100K_goodbooks.csv") #We have userid, bookid, rating
books_metadata = pandas.read_csv("data/Input_goodbooks_dataset_100K/goodbooks_10k_rating_and_description.csv") #bookid + genres
books_metadata = books_metadata[["book_id", "genres"]]
ratings_book = ratings_book.rename(columns={"itemId": "book_id"})

merged_book = ratings_book.merge(books_metadata, on="book_id", how="left")

#Genres needed as a list
def turn_genres_into_list(gen):
    if not isinstance(gen, str) or gen.strip() == "":
        return ""
    types = [g.strip() for g in gen.split("|") if g.strip()]
    return "|".join(types)
    

merged_book["genres"] = merged_book["genres"].apply(turn_genres_into_list)
merged_book.to_csv("data/IMPORTANTdatasets/ratingsandgenres_100K_goodbooks.csv", index=False)

merged_book = merged_book.rename(columns={"book_id": "itemId"}) #CHANGE IT BACK because we all worked with itemID up until now
merged_book.to_csv("data/IMPORTANTdatasets/ratingsandgenres_100K_goodbooks.csv", index=False)

######################Movies


ratings_data = pandas.read_csv("data/IMPORTANTdatasets/ratings_100K_movies.csv", low_memory=False)
links_data = pandas.read_csv("data/Input_movies_dataset_100K/links.csv", low_memory=False)
movie_data = pandas.read_csv("data/Input_movies_dataset_100K/movies_metadata.csv", low_memory=False)


movie_genres = movie_data[["id", "genres"]].copy()
movie_genres["id"] = pandas.to_numeric(movie_genres["id"], errors="coerce")
movie_genres = movie_genres.dropna(subset=["id"])
movie_genres["id"] = movie_genres["id"].astype(int)
movie_genres = movie_genres.drop_duplicates(subset=["id"], keep="first")



links_data["tmdbId"] = pandas.to_numeric(links_data["tmdbId"], errors="coerce")

if "timestamp" in ratings_data.columns:
    ratings_data = ratings_data.drop(columns="timestamp")


rating_links_merged = ratings_data.merge(
    links_data[["movieId",  "tmdbId"]],
    on="movieId",
    how="left" 
)


final_dataset = rating_links_merged.merge(
    movie_genres,
    left_on="tmdbId",
    right_on="id",
    how="left" 
)

final_dataset = final_dataset.drop(columns=["id"])


def parse_genres(gen):
    if not isinstance(gen, str) or gen.strip() == "":
        return ""
    try:
        parsed = ast.literal_eval(gen)  
    except (ValueError, SyntaxError):
        return ""
    
    if isinstance(parsed, list):
        names = [d["name"] for d in parsed if isinstance(d, dict) and "name" in d]
        return "|".join(names)
    return ""

final_dataset["genres"] = final_dataset["genres"].apply(parse_genres)
final_dataset = final_dataset[["userId", "movieId", "rating", "genres"]]
final_dataset.to_csv("data/IMPORTANTdatasets/ratingsandgenres_100K_movies.csv", index=False)



#SPIT DATA 80% - 10% - 10% => Make sure users and movies in test appear in training! Add 10% Groundtruth to each filev

input1= pandas.read_csv("data/IMPORTANTdatasets/ratings_1M_movies.csv")
input2= pandas.read_csv("data/IMPORTANTdatasets/ratingsandgenres_1M_movies.csv")
input3= pandas.read_csv("data/IMPORTANTdatasets/ratings_100K_goodbooks.csv")
input4= pandas.read_csv("data/IMPORTANTdatasets/ratingsandgenres_100K_goodbooks.csv")
input5= pandas.read_csv("data/IMPORTANTdatasets/ratings_100K_movies.csv")


input6= pandas.read_csv("data/IMPORTANTdatasets/ratingsandgenres_100K_movies.csv")

def split_train_val_test(input, user_column_name, item_column_name, outputfortrain=None, outputfortest=None, outputforval=None, base_name=None):
    train_data, temp_data = train_test_split(input, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


    training_users = set(train_data[user_column_name]) #All unique userIds in the training set
    train_items = set(train_data[item_column_name]) # ALL unique itemIds in the training set
    
    #FIND in val and test all the unknown users and items that do not appear in train
    is_val_unknown = ((~val_data[user_column_name].isin(training_users)) |
        (~val_data[item_column_name].isin(train_items)))
    is_test_unknown = (
        (~test_data[user_column_name].isin(training_users)) |
        (~test_data[item_column_name].isin(train_items))
    )
    #Whcih ones
    val_unknown = val_data[is_val_unknown]
    test_unknown = test_data[is_test_unknown]

    #MOve the unknown in test and val to train!
    train_data = pandas.concat([train_data, val_unknown, test_unknown], ignore_index=True)

    #removie from val and test the unknown ones
    val_data = val_data[~is_val_unknown]
    test_data = test_data[~is_test_unknown]

    #Upated users in the training set
    train_users = set(train_data[user_column_name]) 
    train_items = set(train_data[item_column_name])

    val_data = val_data[
        val_data[user_column_name].isin(train_users) &
        val_data[item_column_name].isin(train_items)
    ]
    test_data = test_data[
        test_data[user_column_name].isin(train_users) &
        test_data[item_column_name].isin(train_items)
    ]

    #building back index from 0 ..number of rating
    train_data = train_data.reset_index(drop=True)
    val_data   = val_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)
    if outputfortrain is not None and outputfortest is not None and outputforval is not None and base_name is not None:
        train_path = f"{outputfortrain}/{base_name}_train.csv"
        val_path   = f"{outputforval}/{base_name}_val.csv"
        test_path  = f"{outputfortest}/{base_name}_test.csv"

        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)

    return train_data, val_data, test_data



#train1, val1, test1 = split_train_val_test(input1, "userId", "movieId", "data/TRAIN_GROUNDTRUTH", "data/TEST_GROUNDTRUTH", "data/VAL_GROUNDTRUTH", "ratings_1M_movies")
#train2, val2, test2 = split_train_val_test(input2, "userId", "movieId", "data/TRAIN_GROUNDTRUTH", "data/TEST_GROUNDTRUTH", "data/VAL_GROUNDTRUTH", "ratingsandgenres_1M_movies")
#train3, val3, test3 = split_train_val_test(input3, "user_id", "itemId", "data/TRAIN_GROUNDTRUTH", "data/TEST_GROUNDTRUTH", "data/VAL_GROUNDTRUTH", "ratings_100K_goodbooks")
#train4, val4, test4 = split_train_val_test(input4, "user_id", "itemId", "data/TRAIN_GROUNDTRUTH", "data/TEST_GROUNDTRUTH", "data/VAL_GROUNDTRUTH", "ratingsandgenres_100K_goodbooks")
#train5, val5, test5 = split_train_val_test(input5, "userId", "movieId", "data/TRAIN_GROUNDTRUTH", "data/TEST_GROUNDTRUTH", "data/VAL_GROUNDTRUTH", "ratings_100K_movies")

train6, val6, test6 = split_train_val_test(input6, "userId", "movieId", "data/TRAIN_GROUNDTRUTH", "data/TEST_GROUNDTRUTH", "data/VAL_GROUNDTRUTH", "ratingsandgenres_100K_movies")


#Pair all users with all items in TEST_groundtruth!

movie1 = "data/TEST_GROUNDTRUTH/ratings_1M_movies_test.csv"
movie2 = "data/TEST_GROUNDTRUTH/ratings_100K_movies_test.csv"

movie3 = "data/TEST_GROUNDTRUTH/ratingsandgenres_1M_movies_test.csv"
movie4 = "data/TEST_GROUNDTRUTH/ratingsandgenres_100K_movies_test.csv"

book1 = "data/TEST_GROUNDTRUTH/ratings_100K_goodbooks_test.csv"
book2 = "data/TEST_GROUNDTRUTH/ratingsandgenres_100K_goodbooks_test.csv"

total1 = "data/IMPORTANTdatasets/ratings_100K_movies.csv"
total2 = "data/IMPORTANTdatasets/ratings_100K_goodbooks.csv"
total3 = "data/IMPORTANTdatasets/ratings_1M_movies.csv"


def rewrite_rating(input, outputname):
    inputpath = input
    df = pandas.read_csv(inputpath)

    df = df.drop(columns=["timestamp"])
    
    users_unique = df["userId"].unique()
    movies_unique = df["movieId"].unique()

    #Full user and moie column
    full_movies = (pandas.MultiIndex.from_product([users_unique, movies_unique], names = ["userId", "movieId"])
            .to_frame(index=False))

    final = full_movies.merge(df, on=["userId", "movieId"], how="left")
    final = final.sort_values(["userId", "movieId"])

    #Create output filename 
    input_folder = "data/TEST_RECOMMEND_inputfile"
    output_name = outputname
    output_path = os.path.join(input_folder, output_name)

    #Save
    final.to_csv(output_path, index=False)

#rewrite_rating(movie1, "ratings_1M_movies.csv")
#rewrite_rating(movie2, "ratings_100K_movies.csv")


def rewrite_rating_and_genres(input, outputname):
    inputpath = input
    df = pandas.read_csv(inputpath)

    users_unique = df["userId"].unique()
    movies_unique = df["movieId"].unique()

    #Full user and moie column
    full_movies = (pandas.MultiIndex.from_product([users_unique, movies_unique], names = ["userId", "movieId"])
            .to_frame(index=False))

    final = full_movies.merge(df, on=["userId", "movieId"], how="left")
    final = final.sort_values(["userId", "movieId"])

    #Create output filename 
    input_folder = "data/TEST_RECOMMEND_inputfile"
    output_name = outputname
    output_path = os.path.join(input_folder, output_name)

    #Save
    final.to_csv(output_path, index=False)

#rewrite_rating_and_genres(movie3, "ratingsandgenres_1M_movies.csv")
rewrite_rating_and_genres(movie4, "ratingsandgenres_100K_movies.csv")


def rewritebook(input, outputname):
    inputpath = input
    df = pandas.read_csv(inputpath)

    users_unique = df["user_id"].unique()
    books_unique = df["itemId"].unique()

    full_books = (pandas.MultiIndex.from_product([users_unique, books_unique], names = ["user_id", "itemId"])
            .to_frame(index=False))
    
    final = full_books.merge(df, on=["user_id", "itemId"], how="left")
    final = final.sort_values(["user_id", "itemId"])

    #Create output filename 
    input_folder = "data/TEST_RECOMMEND_inputfile"
    output_name = outputname
    output_path = os.path.join(input_folder, output_name)

    #Save
    final.to_csv(output_path, index=False)
'''
#rewritebook(book1, "ratingsbooks_100K.csv")
#rewritebook(book2, "ratingsandgenresgoodbook_100K.csv")
'''
##### Do entire dataset

#rewrite_rating(total1, "ratings_100K_movies_TOTAL.csv")
#rewritebook(total2, "ratings_100K_books_TOTAL.csv")
#rewrite_rating(total3, "ratings_1M_movies_TOTAL.csv")
