import os
import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pandas
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
#import torchvision
#import tqdm

############################################################################# SETTING UP THE DATABASE AND PREPOCESSING
#STEP 1: reading the credits and movies file

movie_data = pandas.read_csv("src\\backend\\Movies_dataset\\movies_metadata.csv", low_memory = False) #rows in the data are weird, so low_memo tries to read whole file before deiciding on datatype 
keywords_data = pandas.read_csv("src\\backend\\Movies_dataset\\keywords.csv", low_memory = False)
links_data = pandas.read_csv("src\\backend\\Movies_dataset\\links_small.csv", low_memory = False)
ratings_data = pandas.read_csv("src\\backend\\Movies_dataset\\ratings_small.csv", low_memory = False)


# STEP 2: Choose the attributes wished in the NN, make a new table and convert the values to numbers useful for the NN
# from movies_data: id, budget, genres, orig_lang, popul, prod_comp, runtime, title, vote_avg, vote_count (10)
# from keywords: id and keywords
# from ratings: userId,movieId,rating, 
# Make a whole new table from these 

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
    how="left" #movies should be kept even no keyword! 
)
#print(key_movie_merged.head(5))


####Part 2.2 Merge links with ratings.  Ratings.csv uses MovieLens movieId whitch connects through links.csv to the movie_info id (TMDB)
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
#print(rating_links_merged) #This dataset should now have userId, movieId, rating, tmdbId


#Part 2.3 Merge Part 2.1 and Part 2.2 on tmdbId and id - This will be the input of the NN

final_dataset = rating_links_merged.merge(
    key_movie_merged,
    left_on="tmdbId", 
    right_on="id",
    how="inner" # keep only movies that exist in key_movie_merged (we want keywords!)
)
#We dont need id and tmdbID, keep only one. MovieID and tdbId is not the same! movieId is from MovieLens and tmdbId is coming from the Movies Dataset from Kaggle. MovieId is the link to ratings and users. TmdbId is the link to metadata about movies

final_dataset = final_dataset.drop(columns=["id"])

final_dataset = final_dataset[["userId", "movieId", "rating", "tmdbId", "budget", "genres", "original_language", "popularity", "production_companies", "runtime", "title", "vote_average", "vote_count",
        "keywords",]]


#SAVE the dataset
#final_dataset.to_csv("input_dataset_for_NN.csv", index=False)


















############################################################################# STARTING THE NN


### STEP 3 - load the data and set up the NN

df = pandas.read_csv("src\\backend\\input_dataset_for_NN.csv")

df = df[["userId", "movieId", "rating", "budget", "popularity", "runtime", "vote_average", "vote_count", "original_language", "genres"]].copy() #missing "keywords"  and ""production_companies"


#Replace missing numerical values in the numerical columns with 0 and standartize features so that they are on similar ranges
numerical_columns = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
df[numerical_columns] = df[numerical_columns].fillna(0.0)
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])






#Embedding for languages
all_languages = sorted(df["original_language"].dropna().unique())
languages_to_index = {lingo: i for i, lingo in enumerate(all_languages)}
number_of_lingos = len(all_languages)

df["language_index"] = df["original_language"].map(languages_to_index)
df["language_index"] = df["language_index"].fillna(0).astype(int) ##Fillna replaces null/NaN values with 0



#keywords, genres and prod_comp has a format like "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]". Need to be changed to ["Animation", "Comedy"]
def change_list(s):
    if pd.isna(s): #if the list is empty
        return []
    try: 
        data = ast.literal_eval(s) #ast.literal evalutates a string containng a Pityon literal (dictionary in this case)
        data = [d['name'] for d in data]
        return data
    except Exception:
        return []

df["genres"] = change_list(df["genres"])

#CREATING VECTORS FOR genres
















#df = df[:250]

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
        
        self.users = torch.tensor(dataframe['user_index'].values, dtype=torch.long) #Turning the 3 columns into 3 rows/tensors with defined type
        self.movies = torch.tensor(dataframe['movie_index'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float)
        self.features = torch.tensor(dataframe['numerical_columns'].values, dtype = torch.float)

    def __len__(self):
        return len(self.ratings) #triple number (user, movie, rating) should be the same as number of ratings (ca. 100.000)
    
    def __getitem__(self, item): #Given index 1, return  (user_1, movie_1, rating_1)
        return (
            self.users[item],
            self.movies[item],
            self.ratings[item],
            self.features[item]
        )

training_dataset = TorchDataset(train_df) #making the training dataset from TorchDataset using the dataframe parameter (mimicking the train_df and val_df) needs to be given to the Trochdataset class
test_dataset = TorchDataset(val_df)

train_loader = DataLoader(training_dataset, batch_size = 50, shuffle=True) #Loading data in minibatches, so returning 50 tensofr of user vectors, 50 tensors of movie, and 50 rating tensors
validation_loader = DataLoader(test_dataset, batch_size = 50, shuffle = False) #Dont shuffle here


### STEP 6 - Defining the Neural network class

class NeuralNetwork(nn.Module):
    def __init__(self, n_users, n_movies, feature_dimension, embed_len=64, h1=32, h2=16, h3=8, output=1):
        super(NeuralNetwork, self).__init__()
        self.user_embeddings = nn.Embedding(n_users, embed_len) #Creating n_users amount of user ID vectors that will identify users
        self.movie_embeddings = nn.Embedding(n_movies, embed_len)
        
        input_dimension = embed_len * 2 + feature_dimension

        #STEP 2.1 Connect the layers with one another
        self.fc1 = nn.Linear(input_dimension, h1) 
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output)

    #STEP 6.2 Define the forward propagation
    def forward(self, uservect, movievect, movie_features): #Shape is (batch_size,1)
        user_embed = self.user_embeddings(uservect)
        movie_embed = self.movie_embeddings(movievect)
        concat_vector= torch.cat([user_embed, movie_embed, movie_features], dim=1) #(Batch, 2* embedding + feature)

        #activation functions for layers
        concat_vector = F.relu(self.fc1(concat_vector))
        concat_vector = F.relu(self.fc2(concat_vector))
        concat_vector = F.relu(self.fc3(concat_vector))

        concat_vector = self.out(concat_vector)
        return concat_vector
    

### STEP 7 - Setting up MSE
numberof_users = len(user_to_index)
numberof_movies = len(movie_to_index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #NN goes much fast on GPUs according to doc. Moving tensor to device here.

model = NeuralNetwork(unique_users, unique_movies, numerical_columns).to(device) #start the nn with two embeddings, 3 layers and an output layer. All params are moved to GPU memory or CPU memory

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #Adam is the optimizer, optimizing the model.parameters with learning rate 0.01

criterion = nn.MSELoss() #Defining the MSE divided by bathsize

### STEP 8 - Training loop (amount of epochs)

num_epoch = 20

for e in range(num_epoch):

    model.train()
    total_loss = 0.0 #initial loss

    for batch_i, (users, movies, ratings, movie_meta) in enumerate(train_loader): #Trainloader gives batches, where a batch contains three tensors. In batch_0 we load 50 users, 50 movies and 50 ratings
        #move tensor to GPU since model is on device, and data must be on the same device as model
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)
        features = movie_meta.to(device)
        #forward pass
        y_preds= model(users, movies) #shape (batchsize, 1). Finds users and movies embeds, concats and passes through the MLP, and outputs a tensor. Tensor looks like tensor([4], [3.5], ..)
        ratings = ratings.view(-1, 1) #ratings must have same form as output (50, 1)  Ratings now looks like tensor(4, 3.5, ...) and we need it to look like ex. on prev line to match the output shape


        loss = criterion(y_preds, ratings) #Compute loss => criterion =nn.MSELoss()
        #UPDATE
        optimizer.zero_grad() #Delete old gradients from prev. batch
        loss.backward() #compute current/acutal grads of loss with all params
        optimizer.step() #update weights and apply changes with learning rate 0.01

        total_loss = total_loss + loss.item() * ratings.size(0) #avergae loss per batch times batch_size (50) = sum of squared errors in batch


    avg_loss = total_loss/len(training_dataset) #Average loss per barch equals to the total loss computed divided with the size of the training data. So I divide the sum of losses over all batches with the total number of rows in the train_data
    print(f"Epoch {epoch_i + 1} has an average loss of:  {avg_loss}.5f")
