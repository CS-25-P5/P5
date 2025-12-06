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

#STEP4) Building the feature vector for input for the NN +  Mapping user and movie to 0 .. n-1

unique_users = final_dataframe["userId"].unique()
unique_movies = final_dataframe["movieId"].unique()
user_to_index = {user: i for i, user in enumerate(unique_users)} 
movie_to_index = {movie: i for i, movie in enumerate(unique_movies)}

final_dataframe["user_index"] = final_dataframe["userId"].map(user_to_index)
final_dataframe["movie_index"] = final_dataframe["movieId"].map(movie_to_index)

genre_columns = [c for c in final_dataframe.columns if c.startswith("genre_")]

def build_genre_vector(row):
    return row[genre_columns].to_numpy(dtype=np.float32)

final_dataframe["united_feature_vector"] = final_dataframe.apply(build_genre_vector, axis = 1 )



### STEP5) - Pytorch friendly dataset
class TorchDataset(Dataset):
    def __init__(self, dataframe): #dataframe
        
        self.users = torch.tensor(dataframe['user_index'].values, dtype=torch.long)
        self.movies = torch.tensor(dataframe['movie_index'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float)
        
        #We have to joing multiple numpy feature vectors (or arrays). One per row. They are turned into a list of lists, where each list is a row vector [[],[],[]]
        self.features = torch.tensor(np.stack(dataframe['united_feature_vector'].values), dtype = torch.float)

    def __len__(self):
        return len(self.ratings) 
    
    def __getitem__(self, item): 
        return (
            self.users[item],
            self.movies[item],
            self.lang[item],
            self.features[item],
            self.ratings[item],
            
        )
    


# STEP6) Create training, test and validation loaders
training_dataset = TorchDataset(dataframe = train_df)
train_loader = DataLoader(training_dataset, batch_size = 15, shuffle=True)

test_dataset = TorchDataset(dataframe = test_df)
test_loader = DataLoader(test_dataset, batch_size = 15, shuffle=False)


validation_dataset = TorchDataset(dataframe = validation_df)
validation_loader = DataLoader(validation_dataset, batch_size = 15, shuffle=False)



#STEP7) Create the MLPmodel
class MLP_Model(nn.Module):
    def __init__(self, n_users, n_movies, embed_len, output, MLP_layers=None):
        super(MLP_Model, self).__init__()
    
        self.user_embeds = nn.Embedding(n_users, embed_len)
        self.movie_embeds = nn.Embedding(n_movies, embed_len)
        
        if hidden_layers is None: 
            hidden_layers = [] 

            self.user_embeds = nn.Embedding(n_users, embed_len)
            self.movie_embeds = nn.Embedding(n_movies, embed_len)

    #STEP 7.1 Connect the layers with one another
            input_dimension = embed_len
            layers = [] 
            previous_input = input_dimension 
            
            for hidden_dim in hidden_layers: 
                layers.append(nn.Linear(previous_input, hidden_dim)) 
                layers.append(nn.ReLU())
                previous_input = hidden_dim 
        
            layers.append(nn.Linear(previous_input, output))
            self.perceptron = nn.Sequential(*layers) 

#STEP 7.2 Define the forward propagation
    def forward(self, uservect, movievect, langvect, movie_features): #Shape is (batch_size,1)
        user_embed = self.user_embeddings(uservect) #Shape is (batch_size, emb_dimension)
        movie_embed = self.movie_embeddings(movievect)

        concat_vector= torch.cat([user_embed, movie_embed, language_embed, movie_features], dim=1) #(Batch, 2* embedding + feature)


