
import time
import os
import torch
from torch import nn
import copy

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pandas
import torch.nn.functional as F
import numpy as np
import random


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
    input_folder = os.path.dirname(inputpath)
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
    input_folder = os.path.dirname(inputpath)
    output_name = "GROUNDTRUTH_alluserandbooks.csv"
    output_path = os.path.join(input_folder, output_name)

    #Save
    final.to_csv(output_path, index=False)


#rewrite(input2)
#rewrite(input1)
#rewrite(input5)
#rewrite(input6)

rewritebook(input3)
rewritebook(input4)