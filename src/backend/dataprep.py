import os
import pandas as pandas
import numpy as np

############################################################################# SETTING UP THE DATABASE AND PREPOCESSING
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
#final_dataset.to_csv("data\\input_dataset_for_NN.csv", index=False)
