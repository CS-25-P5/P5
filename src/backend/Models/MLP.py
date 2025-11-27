import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("src\backend\Datasets\MovieLens100kRatings.csv")


class MLP_Model(nn.Module):
    def __init__(self, n_users, n_movies, embed_len=64, MLP_layers=[128, 64, 32, 16], dropout=0.3, out_features=1):
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
    


class MovielensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)
  
    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, item):
        return self.users[item], self.movies[item], self.ratings[item]


def train_model(model, train_loader, lr=0.01, weight_decay=1e-5, epochs=15):
    model.to(device)  
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch_i in range(epochs):
        total_loss = 0

        for users, movies, ratings in train_loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device).unsqueeze(1)

            preds = model(users, movies)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(ratings)

        print(f"Epoch {epoch_i + 1}/{epochs} | Train Loss: {total_loss / len(train_loader.dataset):.4f}")

    return model


def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        preds, trues = [], []
        for users, movies, ratings in test_loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            y_pred = model(users, movies)
            preds.extend(y_pred.cpu())
            trues.extend(ratings.cpu())

    preds = torch.tensor(preds)
    trues = torch.tensor(trues)

    rmse = torch.sqrt(torch.mean((preds - trues) ** 2)).item()
    mae = torch.mean(torch.abs(preds - trues)).item()

    return rmse, mae

# Preprocessing

user_labels = LabelEncoder()
movie_labels = LabelEncoder()

df.userId = user_labels.fit_transform(df.userId.values)
df.movieId = movie_labels.fit_transform(df.movieId.values)

x_train, x_test = train_test_split(df, test_size=0.2, random_state=26, stratify=df.rating.values)

train_dataset = MovielensDataset(x_train)
test_dataset = MovielensDataset(x_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


# Initiliaze MLP Model, Train, & Evaluate

n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()

mlp = MLP_Model(
    n_users,
    n_movies,
    embed_len=64,
    MLP_layers=[128, 64, 32, 16],
    dropout=0.4 
).to(device)


train_model(mlp, train_loader, lr=0.01, weight_decay=1e-5, epochs=10)

mlp_rmse, mlp_mae = evaluate_model(mlp, test_loader)

print(f"\nMLP Results:\nRMSE = {mlp_rmse:.4f}, MAE = {mlp_mae:.4f}")
