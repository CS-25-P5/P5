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


### STEP 1 - load the data and set up the NN

df = pandas.read_csv("data\\input_dataset_for_NN.csv")

df = df[["userId", "movieId", "rating", "budget", "popularity", "runtime", "vote_average", "vote_count", "original_language", "genres"]].copy() #missing "keywords"  and ""production_companies"


#Replace missing numerical values in the numerical columns with 0 and standartize features so that they are on similar ranges
numerical_columns = ["budget", "popularity", "runtime", "vote_average", "vote_count"]

df[numerical_columns] = df[numerical_columns].fillna(0.0) #Replacing NULL VALUES WITH 0
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


#Embedding for languages
all_languages = sorted(df["original_language"].dropna().unique()) #Count the languages
languages_to_index = {lingo: i for i, lingo in enumerate(all_languages)} #Give each language and index
number_of_lingos = len(all_languages) 

df["language_index"] = df["original_language"].map(languages_to_index) #Map them to their indecciss 
df["language_index"] = df["language_index"].fillna(0).astype(int) ##Fillna replaces null/NaN values with 0


#genres has a format like "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]". Need to be changed to ["Animation", "Comedy"]
def change_list(s):
    if pandas.isna(s): #if the list is empty
        return []
    try: 
        data = ast.literal_eval(s) #ast.literal evalutates a string containng a Pityon literal (dictionary in this case)
        data = [d['name'] for d in data] #Get only the name value/field
        return data
    except Exception:
        return []

df["genres_list"] = df["genres"].apply(change_list) #row-by-row application
#print(df[["genres", "genres_list"]].head())

#CREATING VECTORS FOR genres and making them distinct => Multi-label classification for whether movie has a given genre or not
all_genres = sorted({g for genlist in df["genres_list"] for g in genlist}) #Go through all movies, get the genres and make them into a unique list of genres
genres_to_index = {genre : i for i, genre in enumerate(all_genres)} #Give each genre an index
number_of_genres = len(all_genres)

def genre_to_vec(gl):
    vector = np.zeros(number_of_genres, dtype=float)
    for g in gl:
        index = genres_to_index.get(g)
        if index is not None:
            vector[index] = 1.0
    return vector

df["genres_vector"] =  df["genres_list"].apply(genre_to_vec)
#print(df["genres_vector"].iloc[0], df["genres_vector"].iloc[0].shape) #Be aware : sometimes all 0s are returned becuase of missing value in dataset, of mistake in dataset.
#df = df[:250]


#Building the feature vector for numercial features + genres + eventually production companies and keywords

def final_feature_vec(row):
    numeric_vector = row[numerical_columns].values.astype(float) #5 entry points #We iterate over values here
    genre_vector = row["genres_vector"].astype(float) #20 genres
    return np.concatenate([numeric_vector, genre_vector])


df["united_feature_vector"] = df.apply(final_feature_vec, axis=1)
feature_dimension = len(df["united_feature_vector"].iloc[0])
print(df["united_feature_vector"].iloc[0], df["genres_vector"].iloc[0].shape)


### STEP 4 - Mapping user and movie to 0 .. n-1. Nn.Embeddings is a lookuptable that needs indices. current dataset for userId goes 1, 55, 105, 255, 6023.. We turn this into the amount of users
# Pytorch is not working with raw IDs, so we map each userId and movieId to a common new index

unique_users = df["userId"].unique()
unique_movies = df["movieId"].unique()

user_to_index = {user: i for i, user in enumerate(unique_users)} #dictionary comprehension for remapping user and movie to index. So user 1: 0, 55:2, 105:2 
movie_to_index = {movie: i for i, movie in enumerate(unique_movies)}

df["user_index"] = df["userId"].map(user_to_index)
df["movie_index"] = df["movieId"].map(movie_to_index)

### STEP 5 - Split data into train and test
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) #random state ctonrols shuffling applied to the data before applying split. Pass an int for reproducible output across multiple function calls


### STEP 6 - Pytorch friendly dataset
class TorchDataset(Dataset):
    def __init__(self, dataframe): #dataframe
        
        self.users = torch.tensor(dataframe['user_index'].values, dtype=torch.long) #Turning the 3 columns into 3 rows/tensors with defined type from the dataframe imported
        self.movies = torch.tensor(dataframe['movie_index'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float)
        self.lang = torch.tensor(dataframe['language_index'].values, dtype = torch.long)
        
        #We have to joing multiple numpy feature vectors (or arrays). One per row. They are turned into a list of lists, where each list is a row vector [[],[],[]]
        self.features = torch.tensor(np.stack(dataframe['united_feature_vector'].values), dtype = torch.float)

    def __len__(self):
        return len(self.ratings) #triple number (user, movie, rating) should be the same as number of ratings (ca. 100.000)
    
    def __getitem__(self, item): #Given index 1, return  (user_1, movie_1, rating_1)
        return (
            self.users[item],
            self.movies[item],
            self.lang[item],
            self.features[item],
            self.ratings[item],
            
        )

training_dataset = TorchDataset(train_df) #making the training dataset from TorchDataset using the dataframe parameter (mimicking the train_df and val_df) needs to be given to the Trochdataset class
test_dataset = TorchDataset(val_df)

train_loader = DataLoader(training_dataset, batch_size = 50, shuffle=True) #Loading data in minibatches, so returning 50 tensofr of user vectors, 50 tensors of movie, and 50 rating tensors
validation_loader = DataLoader(test_dataset, batch_size = 50, shuffle = False) #Dont shuffle here


### STEP 6 - Defining the Neural network class

class NeuralNetwork(nn.Module):
    def __init__(self, n_users, n_movies, n_language, feature_dimension, language_embedding=8, embed_len=64, h1=128, h2=64, h3=32, output=1):
        super(NeuralNetwork, self).__init__()
        self.user_embeddings = nn.Embedding(n_users, embed_len) #Creating n_users amount of user ID vectors that will identify users
        self.movie_embeddings = nn.Embedding(n_movies, embed_len)
        self.lang_embeddings = nn.Embedding(n_language, language_embedding)
        
        input_dimension = embed_len * 2 + feature_dimension + language_embedding

        #STEP 2.1 Connect the layers with one another
        self.fc1 = nn.Linear(input_dimension, h1) 
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output)

    #STEP 6.2 Define the forward propagation
    def forward(self, uservect, movievect, langvect, movie_features): #Shape is (batch_size,1)
        user_embed = self.user_embeddings(uservect) #Shape is (batch_size, emb_dimension)
        movie_embed = self.movie_embeddings(movievect)
        language_embed = self.lang_embeddings(langvect)

        concat_vector= torch.cat([user_embed, movie_embed, language_embed, movie_features], dim=1) #(Batch, 2* embedding + feature)

        #activation functions for layers
        concat_vector = F.relu(self.fc1(concat_vector))
        concat_vector = F.relu(self.fc2(concat_vector))
        concat_vector = F.relu(self.fc3(concat_vector))

        concat_vector = self.out(concat_vector)
        return concat_vector
    

### STEP 7 - Setting up MSE
numberof_users = len(user_to_index)
numberof_movies = len(movie_to_index)
#number_of_lingos = len(all_languages) #from earilier
#feature_dimension = len(df["united_feature_vector"].iloc[0])  #from earilier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #NN goes much fast on GPUs according to doc. Moving tensor to device here.

model = NeuralNetwork(n_users=numberof_users, n_movies=numberof_movies, n_language=number_of_lingos, feature_dimension=feature_dimension).to(device) #start the nn with two embeddings, 3 layers and an output layer. All params are moved to GPU memory or CPU memory

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #Adam is the optimizer, optimizing the model.parameters with learning rate 0.01

criterion = nn.MSELoss() #Defining the MSE divided by bathsize

### STEP 8 - Training loop (amount of epochs)

num_epoch = 20

for e in range(num_epoch):

    model.train()
    total_loss = 0.0 #initial loss

    for batch_i, (users, movies, languages, movie_meta, ratings) in enumerate(train_loader): #Trainloader gives batches, where a batch contains three tensors. In batch_0 we load 50 users, 50 movies and 50 ratings
        #move tensor to GPU since model is on device, and data must be on the same device as model
        users = users.to(device)
        movies = movies.to(device)
        languages = languages.to(device)
        features = movie_meta.to(device)
        ratings = ratings.to(device)

        #forward pass
        y_preds= model(users, movies, languages, features) #shape (batchsize, 1). Finds users and movies embeds, concats and passes through the MLP, and outputs a tensor. Tensor looks like tensor([4], [3.5], ..)
        ratings = ratings.view(-1, 1) #ratings must have same form as output (50, 1)  Ratings now looks like tensor(4, 3.5, ...) and we need it to look like ex. on prev line to match the output shape


        loss = criterion(y_preds, ratings) #Compute loss => criterion =nn.MSELoss()
        #UPDATE
        optimizer.zero_grad() #Delete old gradients from prev. batch
        loss.backward() #compute current/acutal grads of loss with all params
        optimizer.step() #update weights and apply changes with learning rate 0.01

        total_loss = total_loss + loss.item() * ratings.size(0) #avergae loss per batch times batch_size (50) = sum of squared errors in batch


    avg_loss = total_loss/len(training_dataset) #Average loss per barch equals to the total loss computed divided with the size of the training data. So I divide the sum of losses over all batches with the total number of rows in the train_data
    print(f"Epoch {e + 1} has an average loss of:  {avg_loss: .5f}")
