import os
import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pandas
import torch.nn.functional as F
import numpy as np
import ast
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

### STEP 1 - load the data and set up the NN
original_df = pandas.read_csv("data/input_data_til_MLP_genres_100K.csv")
original_df = original_df[["userId","movieId","rating","genres"]].copy()

#STEP 1.1. : Split into train 80%, validation 10%, test 10% => SAVE
train_df, temporary_df = train_test_split(original_df, test_size=0.2, random_state=42) 
validation_df, test_df = train_test_split(temporary_df, test_size=0.5, random_state=42)

#STEP2) REDO GENRES LIST because weird

#genres has a format like "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]". 
# Need to be changed to ["Animation", "Comedy"]

def change_list(s):
    if pandas.isna(s): #if the list is empty
        return []
    try: 
        data = ast.literal_eval(s) #ast.literal evalutates a string containng a Pityon literal (dictionary in this case)
        data = [d['name'] for d in data] #Get only the name value/field
        return data
    except Exception:
        return []

original_df["genres_list"] = original_df["genres"].apply(change_list) #row-by-row application
print(original_df.columns)


#STEP3) Generate one-hot encoding for all genres (binary)

mlb = MultiLabelBinarizer()

genres_onehot = mlb.fit_transform(original_df["genres_list"]) #tranfroam list column into one-hot

genre_columns = [f"genre_{g}" for g in mlb.classes_] #column names

genres_df = pandas.DataFrame(genres_onehot, columns = genre_columns) #Make dataframe for genre one-hot data

final_dataframe = pandas.concat(           #concat with userId, movieId, and ratings
    [original_df[["userId", "movieId", "rating"]].reset_index(drop=True),
     genres_df.reset_index(drop=True),
    ],
    axis=1
)

print(final_dataframe.head(5))


















class MLP_Model(nn.Module):
    def __init__(self, n_users, n_movies, embed_len, MLP_layers):
        super().__init__()
    
        self.user_embeds = nn.Embedding(n_users, embed_len)
        self.movie_embeds = nn.Embedding(n_movies, embed_len)

        layers = []
        input_size = embed_len * 2

        for output_size in MLP_layers:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  
            input_size = output_size

        layers.append(nn.Linear(MLP_layers[-1], out_features))
        self.fc = nn.Sequential(*layers)

    def forward(self, user, movie):
        user_embed = self.user_embeds(user)
        movie_embed = self.movie_embeds(movie)
        x = torch.cat([user_embed, movie_embed], dim=1)
        return self.fc(x)
