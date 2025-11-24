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

'''BPR is suited for datasets with implicit feedback. Currently we have a Movielens database with ratings 
from 0.5 - 5 (explicit feedback), and we will use a threshold for defining whether an item is positive or negative 
(rating above 3 is positive).'''


#STEP 1 - Redo the database - I need movies and ratings so that I can create triplets. 

dataset = pandas.read_csv("data\\Movies_dataset\\ratings_small.csv")
dataset = dataset[["userId", "movieId", "rating"]]

train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

#Split dataset into likes and dislakes (ratings of 3 and below are negative)
positive_df = train_df[train_df["rating"] > 3].copy() 
negative_df = train_df[train_df["rating"] <=3 ].copy()


#STEP 2 - Mapping user and movie to 0 .. n-1. Nn.Embeddings is a lookuptable that needs indices. 
# Current dataset for userId goes 1, 55, 105, 255, 6023.. We turn this into the amount of users
# Pytorch is not working with raw IDs, so we map each userId and movieId to a common new index

unique_users = dataset["userId"].unique()
unique_movies = dataset["movieId"].unique()

user_to_index = {u: i for i,u in enumerate(unique_users)}
movie_to_index = {m:i for i,m in enumerate(unique_movies)}


#Add the indicies for both positive and negative to the table, 
# so that both use small numbers  instead of userId=545 likes movieId=8000 => userwithindex=0 likes movie with index 10.
positive_df["user_index"] = positive_df["userId"].map(user_to_index)
positive_df["positem_index"] = positive_df["movieId"].map(movie_to_index)

negative_df["user_index"] = negative_df["userId"].map(user_to_index)
negative_df["negitem_index"] = negative_df["movieId"].map(movie_to_index)



# STEP 3 - Build the model for the  triplets

#We group all the positive items 
#rows by user, and for each user we collect the set of movies they liked- 
# We will get user_positiveitem[userindex] = {movie1, movie2, movie 3 ... }

user_positive_item = (
    positive_df.groupby("user_index")["positem_index"].apply(set).to_dict()) 

#Almost exact same as above,but  => user_negativeitem[userindex] = [movie4, movie5, movie 10 ... ]
user_negative_item = (
    negative_df.groupby("user_index")["negitem_index"].apply(list).to_dict())

numberofusers = len(user_to_index)
numberofitems = len(movie_to_index)

#print(f"number of users: {numberofusers} and number of items: {numberofitems}") 671 users and 9066 items

#STEP 3 - Pytorch dataset, where i will be positive item and j will be negative item

class BPRdataset(Dataset):
    def __init__(self, user_pos_item, user_neg_item, num_item, num_users):
        self.user_pos_items = user_pos_item #This is the dict
        self.user_neg_items = user_neg_item #This is hte lsit
        self.number_of_items = num_item
        self.numer_of_users = num_users
        #Create tuple such as [(0,1) and (0, 2)] => i.e. user 0 likes movie 2 and 1
        self.user_positive_pair = [(u, positem) for u, items in user_pos_item.items() for positem in items] 

    def __len__(self):
        return len(self.user_positive_pair) #Length should be the number users mapped to their positive items for every user
    
    def __getitem__(self, index):
        user, positem = self.user_positive_pair[index] #How many positive samples we have (for all users with all pos items)
        
        #Does the given user has negative movies? so user1 : [movie2, movie15] (we get a list), user15 : [] (we get none)
        negative_candidate = self.user_neg_items.get(user) 
        
        #Choose a random sample from the <=3 rated movies from the list and if we 
        # dont have explicit negatives for a given user (user rated all 3<, 
        # then pick a random movie and make sure its not in the positives)
        if negative_candidate: 
            negitem = random.choice(list(negative_candidate))
        else: #If user has empty list for non liked movies
            while True:
                negitem = random.randint(0, self.number_of_items - 1) #Pick a random movie from whole movie collection
                if negitem not in self.user_pos_items[user]: #Make sure its not in users liked set, adn stop if you have a candidate! 
                    break

        #everytime with making a DB call we get a user u, pos item i, and neg item j
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "positive": torch.tensor(positem, dtype =torch.long),
            "negative": torch.tensor(negitem, dtype = torch.long)
        }
    

#Minibatches for reading the data
bpr_dataset = BPRdataset(user_pos_item = user_positive_item, user_neg_item = user_negative_item, num_item = numberofitems, num_users=numberofusers)
bpr_dataloader = DataLoader(bpr_dataset, batch_size = 50, shuffle = True)


#STEP 4 - Neural network for this model
class NNforBPR(nn.Module):
    def __init__(self, number_users, number_items, emb_dim = 64):
        super(NNforBPR, self).__init__()

        self.user_emb = nn.Embedding(number_users, emb_dim) #Encoding the identifying vector
        self.item_emb = nn.Embedding(number_items, emb_dim)


        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, item_i, item_j):
        u = self.user_emb(users) #User vector is shape (Batchsize, 64)
        i = self.item_emb(item_i)
        j = self.item_emb(item_j)


        #We multiple element wise, matching each dimension and sum across all the 64 numbers in the embedding
        positive_score = (u * i).sum(dim=1)
        negative_score = (u * j).sum(dim=1)
        return positive_score, negative_score #A number for each user, movie pair determining how much the user likes this movie. OBS. This is not rating.

# STEP 5 - loss function
def bpr_loss(positive_score, negative_score):
    #Log loss f = - log(sigmoid(pos_score - neg_score))
    #We punish the model if it gives a higher score to a negative item over a positive oe
    result = -torch.sum(torch.log(torch.sigmoid(positive_score-negative_score) + 1e-8))
    return result

#Inst. model
model = NNforBPR(number_users=numberofusers, number_items=numberofitems, emb_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-5)
epochs = 100

def training_with_brp(model, dataloader, optimizer):
    #NN goes much fast on GPUs according to doc. Moving tensor to device here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    model.train()

    for e in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            users = batch["user"].to(device)
            pos = batch["positive"].to(device)
            neg = batch["negative"].to(device)

            positive_score, negative_score = model(users, pos, neg)
             
            loss_function = bpr_loss(positive_score=positive_score, negative_score=negative_score)

            optimizer.zero_grad() #Delete old gradients from prev. batch
            loss_function.backward() #compute current/acutal grads of loss with all params
            optimizer.step() #update weights and apply changes with learning rate 0.01
            total_loss = total_loss + loss_function.item() #How much loss over the total epoch

        print(f"Epoch {e+1}/{epochs} - Loss of {total_loss:.5f}") #Loss per epoch. This should go down with each epoch as we train the model§


# Call the training 
training_with_brp(model, bpr_dataloader, optimizer)


#Save the output from training to a file 

model.eval() #Turn off dropout layers, and turn of gradient computations as well - 
#just the trained weights need to be used to make predictions.

with torch.no_grad(): #no gradient computation

    #map userID to index again
    user_ix = dataset["userId"].map(user_to_index).values
    movie_ix = dataset["movieId"].map(movie_to_index).values

    user_tensor = torch.tensor(user_ix, dtype=torch.long) #Make rows from columns for user and corresponding movies
    movie_tensor = torch.tensor(movie_ix, dtype=torch.long)

    user_em = model.user_emb(user_tensor) #Creating a 64 wentry row for each entry in the user_tensor
    movie_em = model.item_emb(movie_tensor)

    predict_score = (user_em * movie_em).sum(dim=1).cpu().numpy() 
    #Take array by array and multiple userid vector with omvie, give one number for that calc.
    #|pandas needs data on CPU and so does nupmy + convert pytorch tensor to numpy, so that pandas can put it into a dataframe

prediction_dataset = dataset.copy()
prediction_dataset["rating"] = predict_score

prediction_dataset.to_csv("data\\predictionNNwithBPR.csv", index = False)

print(prediction_dataset.head(5).to_string())