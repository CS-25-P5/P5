import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


df = pd.read_csv("src\\backend\\ratings.csv")


#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   # print(df.head(6000))
#print(df.tail())
#print(len(df.movieId.unique()))
#print(len(df.userId.unique()))
#print(sorted(df.rating.unique()))





class MLP_Model(nn.Module):
    def __init__(self, n_users, n_movies, embed_len=64, h1=64, h2=32, h3=16, out_features=1):
        super().__init__()
        self.user_embeds = nn.Embedding(n_users, embed_len)
        self.movie_embeds = nn.Embedding(n_movies, embed_len)

        self.fc1 = nn.Linear(embed_len * 2, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, user, movie):
        user_embed = self.user_embeds(user)
        movie_embed = self.movie_embeds(movie)
        x = torch.cat([user_embed, movie_embed], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.out(x)
        return x

class MovielensDataset(Dataset):
  def __init__(self, df):
      self.users = torch.tensor(df['userId'].values, dtype=torch.long)
      self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
      self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

  def __len__(self):
    return len(self.ratings)

  def __getitem__(self, item):

    return (
        self.users[item],
        self.movies[item],
        self.ratings[item]
    )


user_labels = LabelEncoder()
movie_labels = LabelEncoder()

df.userId = user_labels.fit_transform(df.userId.values)
df.movieId = movie_labels.fit_transform(df.movieId.values)



x_train, x_test = train_test_split(df, test_size=0.2, random_state=26, stratify=df.rating.values)


train_dataset = MovielensDataset(x_train)
test_dataset = MovielensDataset(x_test)


train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)


n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()

model = MLP_Model(n_users, n_movies)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = nn.MSELoss()

print(model.user_embeds.weight)
print(model.movie_embeds.weight)

epochs = 40

for epoch_i in range(epochs):
    total_loss = 0
    for batch_i, train_data in enumerate(train_loader):

        users = train_data[0]
        movies = train_data[1]
        ratings = train_data[2]

        y_preds= model(users, movies)

        ratings = ratings.view(-1, 1)


        loss = criterion(y_preds, ratings)
        total_loss = total_loss + loss.sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    avg_loss = total_loss/len(train_loader)
    print(f"epoch {epoch_i + 1}: Avg loss is:  {avg_loss}")

print(model.user_embeds.weight)
print(model.movie_embeds.weight)
with torch.no_grad():
    correct = 0
    total = 0
    for test_data in test_loader:
        users = test_data[0]
        movies = test_data[1]
        ratings = test_data[2]

        y_pred = model(users, movies).squeeze()

        correct += ((y_pred - ratings).abs() < 1.0).sum()


        total += ratings.size(0)

    acc = correct / total
    print(f"accuracy (within rating value of +-1.0): {acc:.2f}")
    print(correct)
    print(total)