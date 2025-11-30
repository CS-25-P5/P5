import time
import os
import torch
from torch import nn
import copy

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pandas
import torch.nn.functional as F
import numpy as np
import random


'''BPR is suited for datasets with implicit feedback. Currently we have a Movielens database with ratings from 0.5 - 5 (explicit feedback), and we will use a threshold for defining whether an item is positive or negative (rating above 3 is positive).'''

starttime = time.time() 

#STEP 1 - Redo the database - I need movies and ratings so that I can create triplets. 

dataset = pandas.read_csv("data/Movies_dataset/ratings_small.csv") 
dataset = dataset[["userId", "movieId", "rating"]] 

#STEP 1.1. : Split into train 80%, validation 10%, test 10% 

train_df, temporary_df = train_test_split(dataset, test_size=0.2, random_state=42) 
validation_df, test_df = train_test_split(temporary_df, test_size=0.5, random_state=42)

#STEP 1.2 : Split dataset into likes and dislakes (ratings of 3 and below are negative). Do for train, val, test df

positive_training_df = train_df[train_df["rating"] > 3].copy()  #Train 
negative_training_df = train_df[train_df["rating"] <=3 ].copy() 

positive_validation_df = validation_df[validation_df["rating"] > 3].copy() #Validate 
negative_validation_df = validation_df[validation_df["rating"] <= 3].copy() 

positive_test_df = test_df[test_df["rating"] > 3].copy() #Test
negative_test_df = test_df[test_df["rating"] <= 3].copy()

#STEP 2 - Mapping user and movie to 0 .. n-1. Nn.Embeddings is a lookuptable that needs indices. # Current dataset for userId goes 1, 55, 105, 255, 6023.. We turn this into the amount of users # Pytorch is not working with raw IDs, so we map each userId and movieId to a common new index 

unique_users = dataset["userId"].unique() 
unique_movies = dataset["movieId"].unique() 
user_to_index = {u: i for i,u in enumerate(unique_users)} 
movie_to_index = {m:i for i,m in enumerate(unique_movies)} 

numberofusers = len(user_to_index) 
numberofitems = len(movie_to_index)


#STEP 3 :Add the indicies for both positive and negatives in all 3 datasets, so that all use small numbers instead of userId=545 likes movieId=8000 => userwithindex=0 likes movie with index 10. 


positive_training_df["user_index"] = positive_training_df["userId"].map(user_to_index) #For train 
positive_training_df["positem_index"] = positive_training_df["movieId"].map(movie_to_index) 
negative_training_df["user_index"] = negative_training_df["userId"].map(user_to_index) 
negative_training_df["negitem_index"] = negative_training_df["movieId"].map(movie_to_index)


positive_validation_df["user_index"] = positive_validation_df["userId"].map(user_to_index) #For validation
positive_validation_df["positem_index"] = positive_validation_df["movieId"].map(movie_to_index) 
negative_validation_df["user_index"] = negative_validation_df["userId"].map(user_to_index) 
negative_validation_df["negitem_index"] = negative_validation_df["movieId"].map(movie_to_index) 

positive_test_df["user_index"] = positive_test_df["userId"].map(user_to_index) #For test
positive_test_df["positem_index"] = positive_test_df["movieId"].map(movie_to_index)
negative_test_df["user_index"] = negative_test_df["userId"].map(user_to_index)
negative_test_df["negitem_index"] = negative_test_df["movieId"].map(movie_to_index)


# STEP 4 - Build the model triplets : 
# #We group all the positive items #rows by user, and for each user we collect the set of movies they liked
#We will get training_user_positiveitem[userindex] = {movie1, movie2, movie 3 ... } 
#We will get training_user_negativeitem[userindex] = [movie4, movie5, movie 10 ... ] 

training_user_positive_item = ( 
    positive_training_df.groupby("user_index")["positem_index"].apply(set).to_dict()) 
training_user_negative_item = ( 
    negative_training_df.groupby("user_index")["negitem_index"].apply(list).to_dict()) 

validation_user_positive_item = ( 
    positive_validation_df.groupby("user_index")["positem_index"].apply(set).to_dict()) 
validation_user_negative_item = ( 
    negative_validation_df.groupby("user_index")["negitem_index"].apply(list).to_dict()) 

test_user_positive_item = (
    positive_test_df.groupby("user_index")["positem_index"].apply(set).to_dict()
)
test_user_negative_item = (
    negative_test_df.groupby("user_index")["negitem_index"].apply(list).to_dict()
)


#STEP 5 - Pytorch dataset, where i = positive and j = negative. Pytorch needs specific dataset, cant work with pandas
class BPRdataset(Dataset):
    def __init__(self, user_pos_item, user_neg_item, num_item):
        self.user_pos_items = user_pos_item #This is the dict
        self.user_neg_items = user_neg_item #This is hte lsit
        self.number_of_items = num_item
        
        #Create tuple such as [(0,1) and (0, 2)] => i.e. user 0 likes movie 2 and 1
        self.user_positive_pair = [(u, positem) for u, items in user_pos_item.items() for positem in items] 

    def __len__(self):
        #Length should be the number users mapped to their positive items for every user
        return len(self.user_positive_pair) 
    
    def __getitem__(self, index):
        #How many positive samples we have (for all users with all pos items)
        user, positem = self.user_positive_pair[index] 
        
        #Does the given user has negative movies? so user1 : [movie2, movie15] (we get a list), user15 : [] (we get none)
        negative_candidate = self.user_neg_items.get(user) 
        
        #Choose a random sample from the <=3 rated movies from the list and if we dont have explicit negatives for a given user (user rated all 3 < , then pick a random movie and make sure its not in the positives)
        if negative_candidate: 
            negitem = random.choice(list(negative_candidate))
        else: #If user has empty list for non liked movies
            while True:
                negitem = random.randint(0, self.number_of_items - 1) #Pick a random movie from whole movie collection
                if negitem not in self.user_pos_items[user]: #Make sure its not in users_liked set, adn stop if candidate! 
                    break

        #When we call in batches we get a user u, pos item i, and neg item j
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "positive": torch.tensor(positem, dtype =torch.long),
            "negative": torch.tensor(negitem, dtype = torch.long)
        }
    

#STEP 5.1 minibatches for reading the data
train_bpr_dataset = BPRdataset(
    user_pos_item = training_user_positive_item, 
    user_neg_item = training_user_negative_item, 
    num_item = numberofitems)

train_bpr_dataloader = DataLoader(train_bpr_dataset, batch_size = 50, shuffle = True)



validate_bpr_dataset = BPRdataset(
    user_pos_item = validation_user_positive_item, 
    user_neg_item = validation_user_negative_item, 
    num_item = numberofitems)

validate_bpr_dataloader = DataLoader(validate_bpr_dataset, batch_size = 50, shuffle = False)




test_bpr_dataset = BPRdataset(
    user_pos_item = test_user_positive_item, 
    user_neg_item = test_user_negative_item, 
    num_item = numberofitems)

test_bpr_dataloader = DataLoader(test_bpr_dataset, batch_size = 50, shuffle = False)


#STEP 6 - NN model
class NNforBPR(nn.Module):
    def __init__(self, number_users, number_items, emb_dim = 32, hidden_layers = None, output = 1):
        super(NNforBPR, self).__init__() 

        if hidden_layers is None: #if NN is linear and no hidden layers
            hidden_layers = [] #number of elements are the nr. hidden layers, and the element numbers are the number of neurons

        self.user_emb = nn.Embedding(number_users, emb_dim) #Encoding the matrix [num_users x 64] => each row is a vector for id. a user
        self.item_emb = nn.Embedding(number_items, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01) #initializing weights with some random val before train
        nn.init.normal_(self.item_emb.weight, std=0.01)

#STEP 4.1 Connect the layers with one another
        input_dimension = emb_dim #User-item int. vector
        layers = [] #A list of layers such as 128 => 64 => 32 or just 64
        previous_input = input_dimension #size of input for next layer
        
        for hidden_dim in hidden_layers: #loop throgh hidden layers like [128,64,32]
            layers.append(nn.Linear(previous_input, hidden_dim)) #first layer has previous_input as its input, hidden_layer output
            layers.append(nn.ReLU()) #apply Relu act. func. for layer
            previous_input = hidden_dim #Update: for the next hidden layer, we have the output dimension from prev. layer as input
    
        layers.append(nn.Linear(previous_input, output)) #Final output scorore for movie append to list of layers in a list. Adding Linear(64,1) to layers
        self.perceptron = nn.Sequential(*layers) #Keep the order! Linear(64, 128) => ReLU() => Linear(128, 64) => ReLU(). Dont apply ReLU to last hidden layer!

    def forward(self, users, item_i, item_j):
        u = self.user_emb(users) #User vector is shape (Batchsize, 64). Aka the tensors look like 50 lists after one another, each size 64
        i = self.item_emb(item_i)
        j = self.item_emb(item_j)

        #We multiple element-wise, matching each dimension and sum across all the 64 numbers in the embedding. Vectors : [Batch, emb_dim]
        positive_score = u * i
        negative_score = u * j
      
        output_positive_score = self.perceptron(positive_score).squeeze(-1) #Drop the emb_dim - keep only vector of 50 inputs
        output_negative_score = self.perceptron(negative_score).squeeze(-1)

        #Input it [50, 64] (50 rows of user and movie embeddings each of size 64). We then multiply these and get [50, 1] : fiftly lists containing exactly one element (rating). Squeeze removes the dimension size, so output_positive score is of shape [50] => one score per (user, positive,item) pair
        return output_positive_score, output_negative_score  
        #A number for each user, movie pair determining how much the user likes this movie. OBS. This is not rating.


# STEP 7 - BPR loss function
def bpr_loss(positive_score, negative_score):
    result = -torch.sum(torch.log(torch.sigmoid(positive_score-negative_score) + 1e-8))
    return result


#STEP 8 - Add early stopping
class EarlyStop:
    def __init__(self, patience = 5, min_delta = 0.0001, restore_best_weight = True):
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
                self.status_update = f"Early stopping initialized after {self.count} epochs."
                if self.restore_best_weight: #If this is true ni the input => we load the best weights during all epochs
                    model.load_state_dict(self.best_m)
                return True #Stop the model here and now
            return False    #Dont stop if selv.count is < than patience


#Instantiating the network with model, optimizer with lr and wd, device
model = NNforBPR(number_users=numberofusers, number_items=numberofitems, emb_dim=32, hidden_layers=[64])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.003, weight_decay=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# STEP 9 - two helper function to calculate loss, and predict ratings in different datasets
def evaluate_loss(model, dataloader, device):
    "Compute BPR loss in dataloader for all 3"
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            users = batch["user"].to(device)
            pos = batch["positive"].to(device)
            neg = batch["negative"].to(device)
            
            positive_score, negative_score = model(users, pos, neg)
            loss = bpr_loss(positive_score, negative_score)
            total = total_loss + loss.item()

    return total / len(dataloader)


def predict_ratings(model, dataset, device):
    "We will comåute the scores for user and movie pairs in train, val, test dataframe"
    model.eval() #stop training, use the fixed wieghts
    with torch.no_grad():
        #map userID to index again
        user_id = dataset["userId"].map(user_to_index).values
        movie_id = dataset["movieId"].map(movie_to_index).values

        #Make rows from columns for user and corresponding movies
        user_tensor = torch.tensor(user_id, dtype=torch.long).to(device) 
        movie_tensor = torch.tensor(movie_id, dtype=torch.long).to(device)

        user_em = model.user_emb(user_tensor) #Creating a 64 wentry row for each entry in the user_tensor
        movie_em = model.item_emb(movie_tensor)

        interaction = user_em * movie_em 

        scores = model.perceptron(interaction).squeeze(-1).cpu().numpy() 
        #Take array by array and multiple userid vector with omvie, give one number for that calc.
        #|pandas needs data on CPU and so does nupmy + convert pytorch tensor to numpy, so that pandas can put it into a dataframe

    return scores
        


#STEP 10 - Train the model
def training_with_brp(model, trainloader, validationloader, optimizer, stopearly, epochs, device):
    #NN goes much fast on GPUs according to doc. Moving tensor to device here. 
    model.to(device)

    for epoch in range(1, epochs + 1):
        #Train the model
        model.train()
        total_training_loss = 0.0

        for batch in trainloader:
            users = batch["user"].to(device)
            pos = batch["positive"].to(device)
            neg = batch["negative"].to(device)

            positive_score, negative_score = model(users, pos, neg)
            loss_function = bpr_loss(positive_score=positive_score, negative_score=negative_score)

            optimizer.zero_grad() #Delete old gradients from prev. batch
            loss_function.backward() #compute current/acutal grads of loss with all params
            optimizer.step() #update weights and apply changes with learning rate 0.01
            
            total_training_loss = total_training_loss + loss_function.item() #How much loss over epoch
        
        average_train_loss = total_training_loss/len(trainloader)
        

        #Validate the modell
        average_validation_loss = evaluate_loss(model, validationloader, device)

        print(f"Epoch {epoch}/{epochs} with average train loss: {average_train_loss:.4f} and average validation loss: {average_validation_loss:.4f}")

        #Stop early!
        if stopearly.call(model, average_validation_loss):
            print(stopearly.status_update)
            break
    print("The model has now finished training!")


#Call the early stop and train the model
early_stop = EarlyStop(patience = 5, min_delta = 0.0001, restore_best_weight = True)
training_with_brp(model = model, trainloader = train_bpr_dataloader, 
                  validationloader=validate_bpr_dataloader, 
                  optimizer=optimizer, 
                  stopearly = early_stop, 
                  epochs = 3000, 
                  device=device)



#STEP 11 - Create predictions for validation set

def validate_bpr():
    predicted_score = predict_ratings(model, validation_df, device)

    calculated_loss = evaluate_loss(model, validate_bpr_dataloader, device)

    #Make a csv file
    prediction_val_dataset = validation_df.copy()
    prediction_val_dataset["val_predicted_rating"] = predicted_score
    prediction_val_dataset.to_csv("data//Predictions_val//BPRnn_OneLayer_embed32_lr0003_optimizeradam.csv", index = False)

    endtime =  time.time()

    print(f"\nTraining and validation time :  {endtime - starttime:.2f} seconds\n"
          f"\nAverage loss for validation is : {calculated_loss}\n")
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024*1024)
        print(f"GPUS allocated for train and validate : {max_memory:.2f} MB")

validate_BPR = validate_bpr()


#STEP 12 - test the model
def test_bpr(model, testdataloader, device):
    total_loss = evaluate_loss (model, testdataloader, device)
    #predict stuff using the test_df 
    test_predict_score = predict_ratings(model, test_df, device)
    
    #Tildel prediction til test datasæt
    prediction_test_dataset = test_df.copy()
    prediction_test_dataset["test_predicted_rating"] = test_predict_score

    prediction_test_dataset.to_csv("data//Predictions_test//BPRnn_OneLayer_embed32_lr0003_optimizeradam.csv", index = False)
    return total_loss 

#STEP 13 : TEST IT 

test_loss = test_bpr(model, test_bpr_dataloader, device)
print(f"Final test loss: {test_loss}")