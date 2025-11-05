import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #Splitting data into train and test
from sklearn.preprocessing import LabelEncoder #labeling data
from torch.utils.data import Dataset, DataLoader #elps with batching


df = pd.read_csv("src\\backend\\ratings.csv") # importing and reading data

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   # print(df.head(6000))
#print(df.tail())
#print(len(df.movieId.unique()))
#print(len(df.userId.unique()))
#print(sorted(df.rating.unique()))



class MLP_Model(nn.Module):
    def __init__(self, n_users, n_movies, embed_len=64, h1=64, h2=32, h3=16, out_features=1):
        super().__init__()
        self.user_embeds = nn.Embedding(n_users, embed_len) #ID vector (64 entries)/ embedding for user 
        self.movie_embeds = nn.Embedding(n_movies, embed_len) #ID vector (64 entries)/ embedding for movie

        self.fc1 = nn.Linear(embed_len * 2, h1) #NN layers with input and output. First layer has 128 inputs, and 64 outputs. Weight matrix of h1 * 128, and bias is h1 vector. Randomly gen.
        self.fc2 = nn.Linear(h1, h2) #second layer wth 64 inputs and 32 outputs
        self.fc3 = nn.Linear(h2, h3) #third layer with 32 inputs and 16 outputs
        self.out = nn.Linear(h3, out_features) #output layer taking 16 in and spitting one out

    def forward(self, user, movie):
        user_embed = self.user_embeds(user)     #size 64
        movie_embed = self.movie_embeds(movie)  #size 64
        x = torch.cat([user_embed, movie_embed], dim=1) #concat a user ID and a movie ID into one array

        x = F.relu(self.fc1(x))     #Applying activation function to different layers, and applying fc1 to input x which will compute x * W + b
        x = F.relu(self.fc2(x))     
        x = F.relu(self.fc3(x))     

        x = self.out(x) #Return output
        return x

class MovielensDataset(Dataset): #inheriting from PyT Dataset class having to def. init, len and getitem. Dataset gets access to one triple (user, movie, rating), while Dataload processes batches
  def __init__(self, df):                                                   #turning the 3 columns into 3 tensors or "rows" with defined type
      self.users = torch.tensor(df['userId'].values, dtype=torch.long) 
      self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
      self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

  def __len__(self):
    return len(self.ratings) #defining the amount of rows for the triples (user, movie, rating) which should be the same amount of rows as in the dataset. Dataloader uses this number to know when to stop an epoch

  def __getitem__(self, item): #defining how we get the item. If item is 1 then we return triple (user_1, movie_1_rating_1). Dataloder will call method for each index it needs when batching

    return (
        self.users[item],
        self.movies[item],
        self.ratings[item]
    )


user_labels = LabelEncoder() #Remapping IDs into numbers from 0 .. N-1 because nn.Embedding is a lookuptable and needs inicides
movie_labels = LabelEncoder()

df.userId = user_labels.fit_transform(df.userId.values) #user IDs will be 0 ...users-1, and fir_transform learns the mapping and applies it. We need because in csv items are often not consec. and embedding is a lookuptable
df.movieId = movie_labels.fit_transform(df.movieId.values) #movie IDs will be 0 ...movies-1



x_train, x_test = train_test_split(df, test_size=0.2, random_state=26, stratify=df.rating.values) # Divinding into 80% train and 20% test. Stratify keeps the distribution of rating roughly the same in test and train


train_dataset = MovielensDataset(x_train) #taking pandas Dataframes x_train and making it into Pytorch-like dataset where each column is a tensor row. Pytorch needs tensors
test_dataset = MovielensDataset(x_test)


train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) #Fetching 32 samples per batch, shuffled and we return mini-batches of tensors. Batches ensure efficiency and update their weights based on batch, not single exmaple
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)


n_users = df['userId'].nunique() #How many unique users and movies, to determine the size of the lookup table (embedding)
n_movies = df['movieId'].nunique()

model = MLP_Model(n_users, n_movies) #start the nn with two embeddings, 3 layers and an output layer

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #Adam is the optimizer, optimizing the model.parameters with learning rate 0.01

criterion = nn.MSELoss() #Defining the MSE divided by bathsize

print(model.user_embeds.weight) #Print out the embeddings for user and movies (tensors)
print(model.movie_embeds.weight)

epochs = 40 #We are running 40 times

for epoch_i in range(epochs):
    total_loss = 0
    for batch_i, train_data in enumerate(train_loader): #Trainloader gives batches, where a batch contains three tensors

        users = train_data[0]
        movies = train_data[1]
        ratings = train_data[2]

        y_preds= model(users, movies) #Forward pass

        ratings = ratings.view(-1, 1) #?? Match rating with y_pred shape?


        loss = criterion(y_preds, ratings) #Compare the differente
        total_loss = total_loss + loss.sum().item()
        optimizer.zero_grad() #Delete old gradients from prev. batch
        loss.backward() #compute acutal grads
        optimizer.step() #apply changes with learning rate 0.01


    avg_loss = total_loss/len(train_loader) #After all batches in 1 epoch we get the avg loss per epoch by total loss / batch (32)
    print(f"epoch {epoch_i + 1}: Avg loss is:  {avg_loss}")

print(model.user_embeds.weight) #We print the embeddings again
print(model.movie_embeds.weight)


with torch.no_grad(): #No looking at gradients
    correct = 0
    total = 0
    for test_data in test_loader: #Testing and unpacking a batch
        users = test_data[0]
        movies = test_data[1]
        ratings = test_data[2]

        y_pred = model(users, movies).squeeze() # Dont understand squeeze tho

        correct += ((y_pred - ratings).abs() < 1.0).sum() #Compare predictions and ground truth


        total += ratings.size(0)

    acc = correct / total
    print(f"accuracy (within rating value of +-1.0): {acc:.2f}")
    print(correct)
    print(total)