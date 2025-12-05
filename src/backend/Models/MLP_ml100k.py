import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("src/backend/Datasets/ratings_small.csv")
print(max(df['movieId'].values))

class MLP_Model(nn.Module):
    def __init__(self, n_users, n_movies, embed_len=64, MLP_layers=[128, 64, 32, 16], dropout=0.2, out_features=1):
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


def train_model(model, train_loader, val_loader, lr=0.01, weight_decay=1e-5, epochs=100, patience=5):
    model.to(device)  
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # For early stopper
    counter = 0
    best_loss = float('inf')

    train_results = []


    for epoch_i in range(epochs):
        model.train()
        train_loss = 0
        
        # Training the model
        for users, movies, ratings in train_loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device).unsqueeze(1)

            preds = model(users, movies)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(ratings)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation & early stopping
        with torch.no_grad():
            model.eval()
            val_loss = 0   

            for users, movies, ratings in val_loader:
                users = users.to(device)
                movies = movies.to(device)
                ratings = ratings.to(device).unsqueeze(1)

                preds = model(users, movies)
                loss = criterion(preds, ratings)

                val_loss += loss.item() * len(ratings)

            val_loss = val_loss / len(val_loader.dataset)

            if (val_loss < best_loss):
                best_loss = val_loss
                counter = 0
                best_model_state = model.state_dict()
            else:
                counter += 1
         

        print(f"Epoch {epoch_i + 1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        train_results.append({
            "trainLoss": train_loss,
            "valLoss": val_loss
        })
                 
        if (counter >= patience):
            print(f"Early stopping triggered at epoch: {epoch_i}")
            print(f"Best validation loss: {best_loss:.4f}")
            break


    model.load_state_dict(best_model_state)
    return train_results


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

# Get the K most relevant movies from a particular user
def get_topK(model, user_id, df, k=1, gt_only=True):          
 
    if (user_id not in df['userId'].values):
        print("User ID could not be found")
        return
    
    # Setting gt_only to true only predicts the items the user has interacted with. Otherwise all movies from the dataset are predicted for the user.
    if (gt_only):           
        all_movies = df[df['userId'] == user_id]['movieId'].values
        all_movies = torch.tensor(all_movies, dtype=torch.long).to(device)
        k = len(all_movies)
    else: 
        all_movies = torch.tensor(df['movieId'].unique(), dtype=torch.long).to(device)

    user = torch.tensor(user_id, dtype=torch.long).repeat(all_movies.size(0)).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(user, all_movies).squeeze()
    
    topk_values, topk_indices = torch.topk(predictions, k)
    original_user_id = user_labels.inverse_transform([user_id])[0]
    topk_movies = movie_labels.inverse_transform(all_movies[topk_indices].cpu().numpy().reshape(-1))
    
    return original_user_id, topk_movies, topk_values.cpu().numpy().reshape(-1)

# Saves all predictions of the model to a csv file
def preds_to_csv(model, test_set, csv_fn):
    user_preds = []
    test_user_ids = test_set['userId'].unique()

    for user in test_user_ids:
        user_id, movie_ids, movie_scores = get_topK(model, user, x_test)

        for k in range(len(movie_ids)):
            user_preds.append({
                "userId": user_id,
                "movieId": movie_ids[k],
                "predictedRating": movie_scores[k]
            })

    topK_predictions = pd.DataFrame(user_preds)
    topK_predictions.to_csv(csv_fn, index=False)

def grid_search(h_params, file_paths, epochs, patience):
    
    cartesian_prod = product(*h_params.values())

    # Combinations will be stored in a list of dictionaries
    all_combs = []

    for values in cartesian_prod:
        pairs = zip(h_params.keys(), values)
        all_combs.append(dict(pairs))

    all_combs = sorted(all_combs, key=lambda x: x['batch_size'])

    # Creates dataloaders for all batch sizes specified in h_params
    train_loaders = {bs: DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True) for bs in h_params["batch_size"]}
    valid_loaders = {bs: DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False) for bs in h_params["batch_size"]}
    test_loaders = {bs: DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False) for bs in h_params["batch_size"]}
    print(all_combs)
    print(len(all_combs))

    # Trains models with all combinations of hyperparameters & saves train/val loss & prediction results to two separate files
    for params in all_combs:

        print(f"Training model with hyperparameters: {params}")
        
        train_loader = train_loaders[params["batch_size"]]
        valid_loader = valid_loaders[params["batch_size"]]
        test_loader = test_loaders[params["batch_size"]]
        
        model = MLP_Model(n_users, n_movies, embed_len=params["embed_len"], MLP_layers=params["layers"], dropout=0.1).to(device)
        start = time.time()
        
        results = train_model(
            model,
            train_loader,
            valid_loader,
            lr=params["lr"],
            weight_decay=params["wd"],
            epochs=epochs,
            patience=patience
        )
        end = time.time()
        training_time = end - start
    
        # Save loss & validation results
        with open(f"{file_paths[0]}_{len(params["layers"])}layers_embed{params["embed_len"]}_lr{params["lr"]}_batch{params["batch_size"]}.txt", "w") as f:
            f.write("trainingLoss, validationLoss\n")
            for r in results:
                row = f"{r['trainLoss']:.4f}, {r['valLoss']:.4f}\n"
                f.write(row)
            f.write(f"Training time: {training_time:.2f} seconds\n")

        # Save predictions
        preds_to_csv(model, x_test, f"{file_paths[1]}_{len(params["layers"])}layers_embed{params["embed_len"]}_lr{params["lr"]}_batch{params["batch_size"]}.csv")

# Preprocessing

user_labels = LabelEncoder()
movie_labels = LabelEncoder()

df.userId = user_labels.fit_transform(df.userId.values)
df.movieId = movie_labels.fit_transform(df.movieId.values)

x_train, x_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df.rating.values)
x_test, x_valid = train_test_split(x_temp, test_size=0.5, random_state=42, stratify=x_temp.rating.values)

train_dataset = MovielensDataset(x_train)
valid_dataset = MovielensDataset(x_valid)
test_dataset  = MovielensDataset(x_test)



userList = df['userId'].unique()
list = df['movieId'].unique()

movie_ids = df.movieId.values[:100]
movie_ids  = np.sort(movie_ids)


# Initialise MLP models & train

n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()

h_params = {
    "lr": [0.001, 0.0003],
    "embed_len": [32, 64], 
    "layers": [[32], [64, 32], [128, 64, 32]], 
    "batch_size": [64, 128],
    "wd": [1e-5]
}
epochs = 500
patience = 5
file_paths = ["src/backend/results/MLP/ml100k/loss_results/MLP", "src/backend/results/MLP/ml100k/predictions/MLP"]
grid_search(h_params, file_paths, epochs, patience)

# Save groundtruths to csv file
test_gt = x_test.copy()
test_gt["userId"]  = user_labels.inverse_transform(test_gt["userId"])
test_gt["movieId"] = movie_labels.inverse_transform(test_gt["movieId"])
test_gt.to_csv("src/backend/results/MLP/ml100k/predictions/ground_truth", index=False)
