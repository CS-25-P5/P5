
'''
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import pandas as pandas
import numpy as np
import ast
import copy
import time
import os



def run_program(optim,
                weightdecay,
                batchsize,
                hiddenlayers,
                learningrate,
                embedding_length,
                prediction_val_save,
                prediction_test_save):
    
    ### STEP 1 - load the data and set up the NN
    original_df = pandas.read_csv("data/input_data_til_MLP_genres_100K.csv")
    original_df = original_df[["userId","movieId","rating","genres"]].copy()

    
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



    final_dataframe = pandas.concat(           #concat with userId, movieId, and ratings for final dataframe
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

    numberof_users = len(user_to_index)
    numberof_movies = len(movie_to_index)

    final_dataframe["user_index"] = final_dataframe["userId"].map(user_to_index)
    final_dataframe["movie_index"] = final_dataframe["movieId"].map(movie_to_index)


    #create the genre columns
    genre_columns = [c for c in final_dataframe.columns if c.startswith("genre_")]

    def build_genre_vector(row):
        return row[genre_columns].to_numpy(dtype=np.float32) #just use numpy to get an array lijke: [0,0, 1, 0, 1]

    final_dataframe["united_genre_vector"] = final_dataframe.apply(build_genre_vector, axis = 1 )
    
    #STEP 4.1. : Split into train 80%, validation 10%, test 10% 
    train_df, temporary_df = train_test_split(final_dataframe, test_size=0.2, random_state=42) 
    validation_df, test_df = train_test_split(temporary_df, test_size=0.5, random_state=42)

    #STEP5) - Pytorch friendly dataset
    class TorchDataset(Dataset):
        def __init__(self, dataframe): #dataframe
            
            self.users = torch.tensor(dataframe['user_index'].values, dtype=torch.long)
            self.movies = torch.tensor(dataframe['movie_index'].values, dtype=torch.long)
            self.genres = torch.tensor(np.stack(dataframe["united_genre_vector"].values), dtype=torch.float)
            self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float)

        def __len__(self):
            return len(self.ratings)
        
        def __getitem__(self, item): 
            return (
                self.users[item],
                self.movies[item],
                self.genres[item],
                self.ratings[item],
            )
        

    # STEP6) Create training, test and validation loaders
    training_dataset = TorchDataset(dataframe = train_df)
    training_loader = DataLoader(training_dataset, batch_size = batchsize, shuffle=True)

    test_dataset = TorchDataset(dataframe = test_df)
    test_loader = DataLoader(test_dataset, batch_size = batchsize, shuffle=False)


    validation_dataset = TorchDataset(dataframe = validation_df)
    validation_loader = DataLoader(validation_dataset, batch_size = batchsize, shuffle=False)



    #STEP7) Create the MLPmodel
    class MLP_Model(nn.Module):
        def __init__(self, n_users, n_movies, embed_len, genre_length, hidden_layers=None, dropout = 0.5):

            super().__init__()
        
            self.user_embeds = nn.Embedding(n_users, embed_len)
            self.movie_embeds = nn.Embedding(n_movies, embed_len)

            
            if hidden_layers is None: 
                hidden_layers = [] 

            input_dimension = embed_len * 2 + genre_length
            layers = [] 


            for hidden_dim in hidden_layers: 
                layers.append(nn.Linear(input_dimension, hidden_dim)) 
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dimension = hidden_dim 

            #Output lay
            layers.append(nn.Linear(input_dimension, 1))
            self.perceptron = nn.Sequential(*layers) 

        #STEP 7.2 Define the forward propagation
        def forward(self, uservect, movievect, movie_features): #Shape is (batch_size,1)
            user_embed = self.user_embeds(uservect) #Shape is (batch_size, emb_dimension)
            movie_embed = self.movie_embeds(movievect)

            x = torch.cat([user_embed, movie_embed, movie_features], dim=1)  #[batch, embed_len * 2 + movie_feature length]
            output = self.perceptron(x) #batch, 1
            return output.squeeze(-1) #batch

            
    ### STEP8) - Instantiating the model #########################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = MLP_Model(n_users= numberof_users, n_movies=numberof_movies, 
                      embed_len = embedding_length, genre_length=len(genre_columns), 
                      hidden_layers=hiddenlayers, dropout = 0.5).to(device)
    optimizer = optim(model.parameters(), lr=learningrate, weight_decay=weightdecay) 
    criterion = nn.MSELoss() #Defining the MSE divided by bathsize




    #STEP9) - Add early stopping
    class EarlyStop:
        def __init__(self, patience, min_delta, restore_best_weight = True):
            self.patience = patience #How many unchanged epochs we tolerate
            self.min_delta = min_delta #minimum improvement
            self.restore_best_weight = restore_best_weight #load back the best model weights before we stop the nr of epochs
            self.best_m = None          #best model weights
            self.best_loss = None       #best loss (lowest seen so far)
            self.count = 0              #how many epochs in a row we see with no improvement => patience
            self.status_update = ""     #stat update msg

        def call(self, model, value_loss):    #calling once per epoch after having computed the loss
            # if its the first iteration 
            if self.best_loss is None:         
                self.best_loss = value_loss
                self.best_m = copy.deepcopy(model.state_dict()) #save model weights
                return False                                    #Stop training? : False
            
            #if improvement is at least min_delta big
            elif self.best_loss - value_loss >= self.min_delta:   #val_loss is the current epochs loss
                self.best_m = copy.deepcopy(model.state_dict())
                self.best_loss = value_loss
                self.count = 0
                self.status_update = f"There is an improvement, counter is reset to {self.count}."
                return False
            
            #if there is no improvement (or loss is even bigger than befor)
            else: 
                self.count +=1
                self.status_update = f"No more improvements in the last {self.count} epochs."
                if self.count >= self.patience:
                    self.status_update = f"Early stopping initialized after {self.count} epochs without improvement."
                    if self.restore_best_weight: #If this is true ni the input => we load the best weights during all epochs
                        model.load_state_dict(self.best_m)
                    return True #Stop the model here and now
                return False    #Dont stop if selv.count is < than patience



    # STEP 9 - two helper function to calculate loss, and predict ratings in different datasets
    def evaluate_loss(model, dataloader, device):
            "Compute MSE loss in dataloader for all 3 datasets"
            model.eval()
            total_loss = 0

            with torch.no_grad():
                for users, movies, genres, ratings in dataloader:
                    users = users.to(device)
                    movies = movies.to(device)
                    genres = genres.to(device)
                    ratings = ratings.to(device)

                    prediction = model(users, movies, genres)
                    loss = criterion(prediction, ratings)
                    total_loss = total_loss + loss.item()

            return total_loss / len(dataloader)


    def predict_ratings(model, dataset, device):
        "We will give rating for every user and its rated movie:  containing userId, movieId, predicted_rating and genres (as a list)"
        
        model.eval() #stop training, use the fixed wieghts
        
        with torch.no_grad():

            #Make rows from columns for user and corresponding movies
            user_tensor = torch.tensor(dataset["user_index"].values, dtype=torch.long).to(device) 
            movies_tensor = torch.tensor(dataset["movie_index"].values, dtype=torch.long).to(device) 
            genre_array = np.stack(dataset["united_genre_vector"].values)
            genre_tensor = torch.tensor(genre_array, dtype=torch.float).to(device) 

            predictions = model(user_tensor, movies_tensor, genre_tensor).cpu().numpy()
        return  predictions
            


    #STEP 10 - Train the model
    def training(model, trainloader, validationloader, optimizer, stopearly, epochs, device):
        #NN goes much fast on GPUs according to doc. Moving tensor to device here. 
        model.to(device)

        for epoch in range(1, epochs + 1):
            #Train the model
            model.train()
            total_training_loss = 0.0

            for users, movies, genres, ratings in trainloader:
                users = users.to(device)
                movies = movies.to(device)
                genres = genres.to(device)
                ratings = ratings.to(device)

                prediction = model(users, movies, genres)
                loss = criterion(prediction, ratings)

                optimizer.zero_grad() #Delete old gradients from prev. batch
                loss.backward() #compute current/acutal grads of loss with all params
                optimizer.step() #update weights and apply changes with learning rate 0.01
                
                total_training_loss = total_training_loss + loss.item() #How much loss over epoch
        
            average_train_loss = total_training_loss/len(trainloader)
            
            #Validate the modell
            average_validation_loss = evaluate_loss(model, validationloader, device)

            print(f"Epoch {epoch}/{epochs} : "
                f"Sum of batch train losses for one epoch {total_training_loss:.4f} |" 
                f"Avg training loss per batch for one epoch {average_train_loss:.4f} |" 
                f"Avg validation loss per batch for one epoch: {average_validation_loss:.4f}"
                )

            stop = stopearly.call(model, average_validation_loss)
        
            print(
            f"[EarlyStop] val_loss={average_validation_loss:.6f}, "
            f"best_loss={stopearly.best_loss:.6f}, "
            f"no_improve_count={stopearly.count}")

            if stop:
                print(stopearly.status_update)
                break


    #Call the early stop and train the model
    start_time = time.time() #We start timer here
    early_stop = EarlyStop(patience = 5, min_delta = 0, restore_best_weight = True)

    training(model = model, trainloader = training_loader, 
                    validationloader=validation_loader, 
                    optimizer=optimizer, 
                    stopearly = early_stop, 
                    epochs = 3000, 
                    device=device)




    #STEP 11 - Create predictions for validation set
    def validate():


        #Get preds in val dataset
        prediction = predict_ratings(model, validation_df, device)
        #Avr val loss pr batch
        average_val_loss_per_batch = evaluate_loss(model, validation_loader, device)
        #Measure how much time this whole thing takes
        end_time = time.time()
        elapsed_sec = end_time - start_time

        #Try to get max GPU usage for entire program
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024*1024)
        else:
            max_memory = None



        #Make a csv file
        prediction_val_dataset = validation_df.copy()
        prediction_val_dataset =  prediction_val_dataset[["userId", "movieId", "rating"]].copy()
        prediction_val_dataset["val_rating"] = prediction
        prediction_val_dataset.to_csv(prediction_val_save, index = False)
        
        #Add 2-3 lines about time and GPU usage:
        with open(prediction_val_save, "a") as file:
            file.write("\n")
            file.write(f"# Time spent on training and validation :  {elapsed_sec:.3f} seconds\n")
            file.write(f"# Average validation loss per batch for one epoch : {average_val_loss_per_batch:.4f}\n")
            if max_memory is not None: 
                file.write(f"# Maximum GPU allocated for the entire program : {max_memory:.2f} MB")
            else:
                file.write("# GPU not available for the program")

    validate()

    #STEP 12 - test the model
    def test(model, testdataloader, device):
        average_test_loss_per_batch = evaluate_loss(model, testdataloader, device)
        #predict stuff using the test_df 
        test_predict_score = predict_ratings(model, test_df, device)
        
        #Tildel prediction til test datasæt
        prediction_test_dataset = test_df.copy()
        prediction_test_dataset = prediction_test_dataset[["userId", "movieId", "rating"]].copy()
        prediction_test_dataset["test_rating"] = test_predict_score
        prediction_test_dataset.to_csv(prediction_test_save, index = False)

        with open(prediction_test_save, "a") as file:
            file.write("\n")
            file.write(f"# Average testing loss per batch for one epoch: {average_test_loss_per_batch:.4f}\n")

    #STEP 13 : TEST IT 
    test(model, test_loader, device)




a1 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr0001_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr0001_batch64.csv")
    
a2 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr0001_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr0001_batch64.csv")
    


a3 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr00003_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr00003_batch64.csv")
    
a4 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr00003_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr00003_batch64.csv")
    

a5 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr0001_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr0001_batch128.csv")


a6 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr0001_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr0001_batch128.csv")




a7 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr00003_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed64_lr00003_batch128.csv")




a8 =  run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr00003_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_OneLayer_embed32_lr00003_batch128.csv")












b1 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr0001_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr0001_batch64.csv")
    
b2 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr0001_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr0001_batch64.csv")
    


b3 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr00003_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr00003_batch64.csv")
    
b4 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr00003_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr00003_batch64.csv")
    

b5 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr0001_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr0001_batch128.csv")


b6 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr0001_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr0001_batch128.csv")




b7 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr00003_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed64_lr00003_batch128.csv")




b8 =  run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr00003_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_TwoLayers_embed32_lr00003_batch128.csv")




























c1 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr0001_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr0001_batch64.csv")
    
c2 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr0001_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr0001_batch64.csv")
    


c3 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr00003_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr00003_batch64.csv")
    
c4 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr00003_batch64.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr00003_batch64.csv")
    

c5 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr0001_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr0001_batch128.csv")


c6 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr0001_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr0001_batch128.csv")




c7 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr00003_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed64_lr00003_batch128.csv")




c8 =  run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/OUTPUT_datasets/NN/Output_Predictions_val_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr00003_batch128.csv",
                prediction_test_save = "data/OUTPUT_datasets/NN/Output_Predictions_test_100K_movies(MLPwithGenres)/NNattr_ThreeLayers_embed32_lr00003_batch128.csv")

'''