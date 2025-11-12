import os
import torch
from torch import nn
import torchvision
import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch.nn.functional as F
#import matplotlib.pyplot as plt


# Movielensdataset: 100,000 ratings (1-5) from 943 users on 1682 movies. 
# Each user has rated at least 20 movies. 
# Simple demographic info for the users (age, gender, occupation, zip)


#Read raw file
df = pd.read_csv("src\\backend\\u.data.txt", sep='\t', header = None, names = ['userId', 'movieId', 'rating', 'timestamp'])
df.to_csv("src\\backend\\ratings.csv", index = False)

#STEP1: Try to make a given list before neural network (top 10 movies per user to see before and after)

dataset = pd.read_csv("src\\backend\\ratings.csv")
dataset_sorted = dataset.sort_values(by=["userId", "rating", "timestamp"], ascending = [True, False, False])
toptenperuser = dataset_sorted.groupby("userId", group_keys=False).head(10)
toptenperuser.to_csv("src\\backend\\top10_per_user_long.csv", index=False)
print(toptenperuser)

positive_threshold = 4.0
positive_df = dataset[dataset["rating"] >= positive_threshold].copy()

#STEP2: Making the network

class NeuralNetwork(nn.Module):
    def __init__(self, n_users, n_movies, embed_len=32, h1=16, h2=8, h3=4, output=1):
        super(NeuralNetwork, self).__init__()
        self.user_embeddings = nn.Embedding(n_users, embed_len) #Creating n_users amount of user ID vectors
        self.movie_embeddings = nn.Embedding(n_movies, embed_len)

        #STEP 2.1 Connect the layers with one another
        self.fc1 = nn.Linear(embed_len*2, h1) 
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output)

    #STEP 2.2 Define the forward propagation
    def forward(self, uservect, movievect):
        user_embed = self.user_embeddings(uservect)
        movie_embed = self.movie_embeddings(movievect)
        concat_vector_pair= torch.cat([user_embed, movie_embed], dim=1)

        #activation functions for layers
        concat_vector_pair = F.relu(self.fc1(concat_vector_pair))
        concat_vector_pair = F.relu(self.fc2(concat_vector_pair))
        concat_vector_pair = F.relu(self.fc3(concat_vector_pair))

        concat_vector_pair = self.out(concat_vector_pair)
        return concat_vector_pair
        

#STEP3 - Movielens dataset needs to be made Pytorch friendly
class MovielensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long) #Turning the 3 columns into 3 rows/tensors with defined type
        self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings) #triple number (user, movie, rating) should be the same as number of ratings (ca. 100.000)
    
    def __getitem__(self, item): #the way we get the row 1 is by getting (user, movie, rating)
        return (
            self.users[item],
            self.movies[item],
            self.ratings[item]
        )

user_labels = LabelEncoder() #Remapping IDs into numbers from 0 .. N-1 because nn.Embedding is a lookuptable and needs inicides with no jumps
movie_labels = LabelEncoder()

dataset.userId = user_labels.fit_transform(dataset.userId.values) #Rewriting the userIds in the dataset from 0..N-1 and mapping it accordingly. We need because in csv items are having chaotic and not consec. numbers
dataset.movieId = movie_labels.fit_transform(dataset.movieId.values)





































#Step4 - Bayesian Personalized ranking  log loss function -  item is positive if rating >=4, negative otherwise



#Step5 - Split the data into test and train



