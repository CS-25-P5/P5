import os
import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pandas
import torch.nn.functional as F
import numpy as np
import random
#import matplotlib.pyplot as plt
#import torchvision
#import tqdm


#STEP 1 - Redo the database - now I need movies and ratings so that I can create triplets. 

dataset = pandas.read_csv("src\\backend\\Movies_dataset\\ratings_small.csv")

dataset = dataset[["userId", "movieId", "rating"]]

#Split dataset into likes and displake (ratings of 1 and below are negative)
positive_df = dataset[dataset["rating"] > 1].copy() 
negative_df = dataset[dataset["rating"] <=1 ].copy()


#STEP 2 - Mapping user and movie to 0 .. n-1. Nn.Embeddings is a lookuptable that needs indices. current dataset for userId goes 1, 55, 105, 255, 6023.. We turn this into the amount of users
# Pytorch is not working with raw IDs, so we map each userId and movieId to a common new index

unique_users = dataset["userId"].unique()
unique_movies = dataset["movieId"].unique()

user_to_index = {u: i for i,u in enumerate(unique_users)}
movie_to_index = {m:i for i,m in enumerate(unique_movies)}

index_to_user = {i: u for u, i in user_to_index.items()} #This is just a reverse mapping for possible decoding of indices. So user with wiht 22 is user with ID 154 in dataset. Did not use it yet.
index_to_movie = {i: m for m,i in movie_to_index.items()}

#Add the indicies for both positive and negative to the table, so that both use small numbers  instead of userId=545 likes movieId=8000 => userwithindex=0 likes movie with index 10.
positive_df["user_index"] = positive_df["userId"].map(user_to_index)
positive_df["positem_index"] = positive_df["movieId"].map(movie_to_index)

negative_df["user_index"] = negative_df["userId"].map(user_to_index)
negative_df["negitem_index"] = negative_df["movieId"].map(movie_to_index)




#Build the triplets

user_positive_item = (
    positive_df.groupby("user_index")["positem_index"].apply(set).to_dict()) #We group all the positive items rows by user, and for weach user we collect the set of movies they liked- We will get user_positiveitem[userindex] = {movie1, movie2, movie 3 ... }

user_negative_item = ( #Almost exact same as above, but here we will get a lsit inteas of a dict so basically => user_negativeitem[userindex] = [movie4, movie5, movie 10 ... ]
    negative_df.groupby("user_index")["negitem_index"].apply(list).to_dict())


numberofusers = len(user_to_index)
numberofitems = len(movie_to_index)

#print(f"number of users: {numberofusers} and number of items: {numberofitems}") 671 users and 9066 items




#STEP 3 - Pytorch dataset. i will be positive item and j will be negative item

class BPRdataset(Dataset):
    def __init__(self, user_pos_item, user_neg_item, num_item):

        self.user_pos_items = user_pos_item #This is a dict
        self.user_neg_items = user_neg_item #This is a lsit
        self.number_of_items = num_item

        self.user_positive_pair = [(u, positem) for u, items in user_pos_item.items() for positem in items] #Create tuple such as [(0,1) and (0, 2)] => i.e. user 0 likes movie 2 and 1


    def __len__(self):
        return len(self.user_positive_pair)
    
    def __getitem__(self, index):
        u, i = self.user_positive_pair[index] #How many positive samples we have

        negative_candidate = self.user_neg_items.get(u) #Does the given user has negative movies? so user1 : [movie2, movie15] (we get a list), user15 : [] (we get none)
        
        if negative_candidate: #Choose a random sample from the <=1 rated nivues frin tge kustm and if we dont have explicit negatives for a given user (he rated all 1<, then pick a random movie and make sure its not in the positives)
            j = random.choice(list(negative_candidate))
        else: #If user has empty list for non liked movies
            while True:
                j = random.randint(0, self.number_of_items-1) #Pick a random movie from whole movie collection
                if j not in self.user_pos_items[u]: #Make sure its not in the liked set! 
                    break
        #everytime with making a DB call we get a user u, pos item i, and neg item j
        return {
            "user": torch.tensor(u, dtype=torch.long),
            "positive": torch.tensor(i, dtype =torch.long),
            "negative": torch.tensor(j, dtype = torch.long)
        }
    

#Minibatches

dataset = BPRdataset(user_pos_item = user_positive_item, user_neg_item = user_negative_item, num_item = numberofitems)
dataloader = DataLoader(dataset, batch_size = 50, shuffle = True)


#STEP 4 - Neural network for this model

class NNforBPR(nn.Module):
    def __init__(self, number_users, number_items, emb_dim = 64):
        super(NNforBPR, self).__init__()
        self.user_emb = nn.Embedding(number_users, emb_dim)
        self.item_emb = nn.Embedding(number_items, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.03) #We start with random weights as a start
        nn.init.normal_(self.item_emb.weight, std=0.03)


    def forward(self, users, items):
        u = self.user_emb(users) #User vector is shape (Batch, 64)
        i= self.item_emb(items)
        score = (u * i).sum(dim=1) #We multiple element wise, matching each dimension and sum across all the 64 numbers in the embedding
        return score #A number for each user, movie pair determining how much the user likes this movie. OBS. This is not rating.

# STEP 5 - loss function

def bpr_loss(positive_score, negative_score):
    #Log loss f = - log(sigmoid(pos_score - neg_score))
    result = -torch.mean(torch.log(torch.sigmoid(positive_score-negative_score) + 1e-8)) #We punish the model if it gives a higher score to a negative item over a positive oe
    return result


#Inst. model
model = NNforBPR(number_users=numberofusers, number_items=numberofitems, emb_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 5

def training_with_brp(model, dataloader, optimizer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #NN goes much fast on GPUs according to doc. Moving tensor to device here.
    model.to(device)
    model.train()


    for e in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            users = batch["user"].to(device)
            pos = batch["positive"].to(device)
            neg = batch["negative"].to(device)

            positive_scor = model(users, pos)
            negative_scor = model(users, neg)
             
            loss_function = bpr_loss(positive_score=positive_scor, negative_score=negative_scor)

            optimizer.zero_grad() #Delete old gradients from prev. batch
            loss_function.backward() #compute current/acutal grads of loss with all params
            optimizer.step() #update weights and apply changes with learning rate 0.01


            total_loss = total_loss + loss_function.item() #How much loss over the total epoch

        print(f"Epoch {e+1}/{epochs} - Loss of {total_loss:.5f}") #Loss per epoch. This should go down with each epoch as we train the model§


# Call the training 
training_with_brp(model, dataloader, optimizer)