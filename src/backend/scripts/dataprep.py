
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
import json
from sklearn.preprocessing import MultiLabelBinarizer


'''
############################################################################# SETTING UP THE DATABASE AND PREPOCESSING - 100 K movies
#STEP 1: reading the credits and movies file

movie_data = pandas.read_csv("data\\Movies_dataset\\movies_metadata.csv", low_memory = False) #rows in the data are weird, so low_memo tries to read whole file before deiciding on datatype 
keywords_data = pandas.read_csv("data\\Movies_dataset\\keywords.csv", low_memory = False)
links_data = pandas.read_csv("data\\Movies_dataset\\links_small.csv", low_memory = False)
ratings_data = pandas.read_csv("data\\Movies_dataset\\ratings_small.csv", low_memory = False)

# STEP 2: Choose the attributes wished in the NN db => make a whole new table from these 
# from movies_data: id, budget, genres, orig_lang, popul, prod_comp, runtime, title, vote_avg, vote_count (10)
# from keywords: id and keywords
# from ratings: userId,movieId,rating, 

movie_data_columns = ["id", "budget", "genres", "original_language", "popularity", "production_companies", "runtime", "title", "vote_average", "vote_count"]
movies_shrunk = movie_data[movie_data_columns].copy()

####Part 2.1 Merge movies_metadata + keywords based on id

movies_shrunk["id"] = pandas.to_numeric(movies_shrunk["id"], errors = "coerce") # error='oerce' means that invalid parsing will be set as NaN. Make sure ids are numbers (in some places they are dates?)
movies_shrunk = movies_shrunk.dropna(subset=["id"]) #drop rows with missing values from id column
movies_shrunk["id"] = movies_shrunk["id"].astype(int)

keywords_data["id"] = pandas.to_numeric(keywords_data["id"], errors= "coerce")
keywords_data = keywords_data.dropna(subset=["id"])
keywords_data["id"] = keywords_data["id"].astype(int)

key_movie_merged = movies_shrunk.merge(
    keywords_data[["id", "keywords"]],
    on="id",
    how="left" #movies should be kept even if no keyword! 
)
#print(key_movie_merged.head(5).to_string())

####Part 2.2 Merge links with ratings.  
# Ratings.csv uses MovieLens movieId whitch connects through links.csv to the movie_info id (TMDB)
links_data["tmdbId"] = pandas.to_numeric(links_data["tmdbId"], errors = "coerce")
links_data = links_data.dropna(subset=["tmdbId"])
links_data['tmdbId'] = links_data["tmdbId"].astype(int)

if "timestamp" in ratings_data.columns: 
    ratings_data = ratings_data.drop(columns="timestamp")

rating_links_merged = ratings_data.merge(
    links_data[["movieId", "tmdbId"]],
    on = "movieId",
    how="inner" #Keep only ratings whose movieId exists in the links => we need them for user recommend.
)
#print(rating_links_merged.head(5).to_string())



#Part 2.3 Merge Part 2.1 and Part 2.2 on tmdbId and id - This will be the input of the NN

final_dataset = rating_links_merged.merge(
    key_movie_merged,
    left_on="tmdbId", 
    right_on="id",
    how="inner" # keep only movies that exist in key_movie_merged (we want keywords!)
)

#We dont need id and tmdbID, keep only one. 
# MovieID and tdbId is not the same! movieId is from MovieLens and tmdbId is coming from the Movies Dataset from Kaggle. 
# MovieId is the link to ratings and users. TmdbId is the link to metadata about movies

final_dataset = final_dataset.drop(columns=["id"])
final_dataset = final_dataset[["userId", "movieId", "rating", "tmdbId", "budget", "genres", 
                               "original_language", "popularity", "production_companies", "runtime", 
                               "title", "vote_average", "vote_count","keywords",]]

#print(final_dataset.head(5).to_string())

#SAVE the dataset
#final_dataset.to_csv("data\\input_data_til_MLP_genres_100K", index=False)


'''

############################################################################# SETTING UP THE DATABASE AND PREPOCESSING - 1M movies


#Convert dat to csv

'''
input1 = "data/Movies_dataset_1M/ratings.dat"
input2 = "data/Movies_dataset_1M/movies.dat"
input3 = "data/Movies_dataset_1M/users.dat"

new_data = pandas.read_csv(input2, sep = "::", engine = "python", names = ["movieId", "title", "genres"], encoding="latin-1",)
output_file = input2.replace(".dat", "1M.csv")
new_data.to_csv(output_file, index = False)
'''

#Create one file with userId as foreign.k.
'''
movie_rating_1M = pandas.read_csv("data/Movies_dataset_1M/ratings1M.csv")
movie_metadata_1M = pandas.read_csv("data/Movies_dataset_1M/movies1M.csv")


wished_movie_data_columns_1M = ["userId", "movieId", "rating"]
new_movies_1M = movie_rating_1M[wished_movie_data_columns_1M].copy()


new_merged_dataset_1M = new_movies_1M.merge(
    movie_metadata_1M[["movieId", "genres"]],
    on="movieId",
    how="left" 
)

#SAVE the dataset
new_merged_dataset_1M.to_csv("data/input_data_til_MLP_genres_1M.csv", index=False)
'''
############################################################################# SAVE GOUNDTRUTH

'''

def getgroundtruth(input, savingplacetest, savingplaceval):
     
    dataset = pandas.read_csv(input) 
    
    train_df, temporary_df = train_test_split(dataset, test_size=0.2, random_state=42) 
    validation_df, test_df = train_test_split(temporary_df, test_size=0.5, random_state=42)
    validation_groundtruth = validation_df.to_csv(savingplaceval + "/VAL_GROUNDTRUTH.csv", index = False)
    test_groundtruth = test_df.to_csv(savingplacetest + "/TEST_GROUNDTRUTH.csv", index = False)


in1 = "data/Input_movies_dataset_100K/ratings_100K.csv"
savingplacetest1 = "data/Output_Predictions_test_100K_movies(MLPwithBPR)"
savingplaceval1 = "data/Output_Predictions_val_100K_movies(MLPwithBPR)"


in2 = "data/Input_movies_dataset_1M/ratings_1M.csv"
savingplacetest2 = "data/Output_Predictions_test_1M_movies(MLPwithBPR)"
savingplaceval2 = "data/Output_Predictions_val_1M_movies(MLPwithBPR)"

in3 = "data/Input_goodbooks_dataset_100K/ratings_100K.csv"
savingplacetest3 = "data/Output_Predictions_test_100K_goodbooks(MLPwithBPR)"
savingplaceval3 = "data/Output_Predictions_val_100K_goodbooks(MLPwithBPR)"


getgroundtruth(in1, savingplacetest1, savingplaceval1)
getgroundtruth(in2, savingplacetest2, savingplaceval2)
getgroundtruth(in3, savingplacetest3, savingplaceval3)

getgroundtruth(in1, "data/Output_Predictions_test_100K_movies(MLPwithGenres)", "data/Output_Predictions_val_100K_movies(MLPwithGenres)" )
getgroundtruth(in2, "data/Output_Predictions_test_1M_movies(MLPwithGenres)", "data/Output_Predictions_val_1M_movies(MLPwithGenres)")
getgroundtruth(in3, "data/Output_Predictions_test_100K_goodbooks(MLPwithGenres)", "data/Output_Predictions_val_100K_goodbooks(MLPwithGenres)")

'''

############################################################################# SETTING UP THE DATABASE AND PREPOCESSING - 100K GOOdbooks


#Create one file with userId, itemId, rating, genres

'''
books_100K = pandas.read_csv("data/Input_goodbooks_dataset_100K/ratings_100K.csv")
books_genres = pandas.read_csv("data/Input_goodbooks_dataset_100K/books.csv")

new_merged_dataset_100K = books_100K.merge(
    books_genres[["itemId", "genres"]],
    on="itemId",
    how="left" 
)

#SAVE the dataset
new_merged_dataset_100K.to_csv("data/input_data_til_MLP_genres_100K_books.csv", index=False)
'''



########## Make datasets for recommendation test for 100Kbooks and 100K movies


###inputs
books_test_dataset = pandas.read_csv("data/Output_Predictions_test_100K_goodbooks(MLPwithGenres)/GROUNDTRUTH_alluserandbooks.csv")
books_metadata = pandas.read_csv("data/Input_goodbooks_dataset_100K/books.csv")

movies_test_dataset = pandas.read_csv("data/Output_Predictions_test_100K_movies(MLPwithGenres)/GROUNDTRUTH_alluserandmovies.csv")
movies_links = pandas.read_csv("data/Input_movies_dataset_100K/links.csv")
movies_meta = pandas.read_csv("data/Input_movies_dataset_100K/movies_metadata.csv", low_memory=False)

'''
#Books
books_test_dataset = books_test_dataset.head(200000)
merged_books = books_test_dataset.merge(books_metadata[["itemId", "genres"]], on="itemId", how="left")

def processing_missing_genres_forbook(input):
    if pandas.isna(input) or input == "":
        return []              # if no genre => make empty list []
    if isinstance(input, str):
        return input.split("|")    #Split it into a lits
    return input                   
    
merged_books["genres"] = merged_books["genres"].apply(processing_missing_genres_forbook)
merged_books.to_csv("data/Recommend_test_100K_goodbooks(MLPwithGenres)/Final_input.csv", index=False)

'''



#Movies => We have to link ratings to links first, and then the merged file to movie metadata
movies_test_dataset = movies_test_dataset.head(200000)
merged_on_links = movies_test_dataset.merge(movies_links[["movieId", "tmdbId"]], on="movieId", how="left")

merged_on_links["tmdbId"] = pandas.to_numeric(merged_on_links["tmdbId"], errors="coerce")
movies_meta["id"] = pandas.to_numeric(movies_meta["id"], errors="coerce")

merge_all = merged_on_links.merge(movies_meta[["id", "genres"]], left_on="tmdbId", right_on="id", how="left")
#print(merge_all.head(5))
#print(merge_all.columns)
merge_all=merge_all.drop(columns=["tmdbId", "id"])
#print(merge_all.columns)


def change_genre_layout(inputmovie):
    if pandas.isna(inputmovie) or inputmovie == "":
        return[]
    try:
        parsedgenres = ast.literal_eval(inputmovie)
    except (ValueError, SyntaxError):
        return []
    
    if isinstance(parsedgenres, list):
        genre_list = []
        for entry in parsedgenres:
            if isinstance(entry, dict) and "name" in entry:  #Get the name of the dictionary list and append to list
                genre_list.append(entry["name"])
        return genre_list


    #For any other weird input we just return a lsit
    return []

merge_all["genres"] = merge_all["genres"].apply(change_genre_layout) #Apply the function to the dataset

final_dataset = merge_all[["userId", "movieId", "rating", "genres"]]
final_dataset.to_csv("data/Recommend_test_100K_movies(MLPwithGenres)/Final_input_movies.csv", index=False)


