
import time
import os
import torch
from torch import nn
import copy

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pandas

import numpy as np
import random


'''BPR is suited for datasets with implicit feedback. 
Currently we have a Movielens database with ratings from 0.5 - 5 (explicit feedback), and we will use a threshold 
for defining whether an item is positive or negative (rating above 3 is positive).'''



def run_program(optim,
                weightdecay,
                batchsize,
                hiddenlayers,
                learningrate,
                embedding_length,
                prediction_val_save,
                prediction_test_save):
    
    #STEP 1 - Redo the database - I need books and ratings so that I can create triplets. 
    dataset = pandas.read_csv("data/Input_goodbooks_dataset_100K/ratings_100K.csv") 
    dataset = dataset[["userId", "itemId", "rating"]] 

    #STEP 1.1. : Split into train 80%, validation 10%, test 10% => SAVE
    train_df, temporary_df = train_test_split(dataset, test_size=0.2, random_state=42) 
    validation_df, test_df = train_test_split(temporary_df, test_size=0.5, random_state=42)

    #STEP 1.2 : Split dataset into likes and dislakes (ratings of 3 and below are negative). Do for train, val, test df

    positive_training_df = train_df[train_df["rating"] > 3].copy()  #Train 
    negative_training_df = train_df[train_df["rating"] <=3 ].copy() 

    positive_validation_df = validation_df[validation_df["rating"] > 3].copy() #Validate 
    negative_validation_df = validation_df[validation_df["rating"] <= 3].copy() 

    positive_test_df = test_df[test_df["rating"] > 3].copy() #Test
    negative_test_df = test_df[test_df["rating"] <= 3].copy()

    #STEP 2 - Mapping user and books to 0 .. n-1. Nn.Embeddings is a lookuptable that needs indices. # Current dataset for userId goes 1, 55, 105, 255, 6023.. We turn this into the amount of users # Pytorch is not working with raw IDs, so we map each userId and booksId to a common new index 

    unique_users = dataset["userId"].unique() 
    unique_books = dataset["itemId"].unique() 
    user_to_index = {u: i for i,u in enumerate(unique_users)} 
    book_to_index = {m:i for i,m in enumerate(unique_books)} 

    numberofusers = len(user_to_index) 
    numberofitems = len(book_to_index)


    #STEP 3 :Add the indicies for both positive and negatives in all 3 datasets, so that all use small numbers instead of userId=545 likes booksId=8000 => userwithindex=0 likes books with index 10. 


    positive_training_df["user_index"] = positive_training_df["userId"].map(user_to_index) #For train 
    positive_training_df["positem_index"] = positive_training_df["itemId"].map(book_to_index) 
    negative_training_df["user_index"] = negative_training_df["userId"].map(user_to_index) 
    negative_training_df["negitem_index"] = negative_training_df["itemId"].map(book_to_index)


    positive_validation_df["user_index"] = positive_validation_df["userId"].map(user_to_index) #For validation
    positive_validation_df["positem_index"] = positive_validation_df["itemId"].map(book_to_index) 
    negative_validation_df["user_index"] = negative_validation_df["userId"].map(user_to_index) 
    negative_validation_df["negitem_index"] = negative_validation_df["itemId"].map(book_to_index) 

    positive_test_df["user_index"] = positive_test_df["userId"].map(user_to_index) #For test
    positive_test_df["positem_index"] = positive_test_df["itemId"].map(book_to_index)
    negative_test_df["user_index"] = negative_test_df["userId"].map(user_to_index)
    negative_test_df["negitem_index"] = negative_test_df["itemId"].map(book_to_index)


    # STEP 4 - Build the model triplets : 
    # #We group all the positive items #rows by user, and for each user we collect the set of books they liked
    #We will get training_user_positiveitem[userindex] = {books1, books2, books 3 ... } 
    #We will get training_user_negativeitem[userindex] = [books4, books5, books 10 ... ] 

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
            
            #Create tuple such as [(0,1) and (0, 2)] => i.e. user 0 likes books 2 and 1
            self.user_positive_pair = [(u, positem) for u, items in user_pos_item.items() for positem in items] 

        def __len__(self):
            #Length should be the number users mapped to their positive items for every user
            return len(self.user_positive_pair) 
        
        def __getitem__(self, index):
            #How many positive samples we have (for all users with all pos items)
            user, positem = self.user_positive_pair[index] 
            
            #Does the given user has negative books? so user1 : [books2, books15] (we get a list), user15 : [] (we get none)
            negative_candidate = self.user_neg_items.get(user) 
            
            #Choose a random sample from the <=3 rated books from the list and if we dont have explicit negatives for a given user (user rated all 3 < , then pick a random books and make sure its not in the positives)
            if negative_candidate: 
                negitem = random.choice(list(negative_candidate))
            else: #If user has empty list for non liked books
                all_items = set(range(self.number_of_items)) #count all items
                remaining = list(all_items - self.user_pos_items[user])
                if len(remaining) == 0:
                    negitem = random.randint(0, self.number_of_items-1) #Pick a random books from whole books collection
                else:
                    negitem = random.choice(remaining)
                

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

    train_bpr_dataloader = DataLoader(train_bpr_dataset, batch_size = batchsize, shuffle = True)



    validate_bpr_dataset = BPRdataset(
        user_pos_item = validation_user_positive_item, 
        user_neg_item = validation_user_negative_item, 
        num_item = numberofitems)

    validate_bpr_dataloader = DataLoader(validate_bpr_dataset, batch_size = batchsize, shuffle = False)




    test_bpr_dataset = BPRdataset(
        user_pos_item = test_user_positive_item, 
        user_neg_item = test_user_negative_item, 
        num_item = numberofitems)

    test_bpr_dataloader = DataLoader(test_bpr_dataset, batch_size = batchsize, shuffle = False)



    #STEP 6 - NN model
    class NNforBPR(nn.Module):
        def __init__(self, number_users, number_items, emb_dim = embedding_length, hidden_layers = None, output = 1):
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
        
            layers.append(nn.Linear(previous_input, output)) #Final output scorore for books append to list of layers in a list. Adding Linear(64,1) to layers
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

            #Input it [50, 64] (50 rows of user and books embeddings each of size 64). We then multiply these and get [50, 1] : fiftly lists containing exactly one element (rating). Squeeze removes the dimension size, so output_positive score is of shape [50] => one score per (user, positive,item) pair
            return output_positive_score, output_negative_score  
            #A number for each user, books pair determining how much the user likes this books. OBS. This is not rating.




    # STEP 7 - BPR loss function
    def bpr_loss(positive_score, negative_score):
        result = -torch.sum(torch.log(torch.sigmoid(positive_score-negative_score) + 1e-8))
        return result





    #STEP 8 - Add early stopping
    class EarlyStop:
        def __init__(self, patience = 5, min_delta = 1e-5, restore_best_weight = True):
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


    #Instantiating the network with model, optimizer with lr and wd, device
    model = NNforBPR(number_users=numberofusers, number_items=numberofitems, emb_dim=embedding_length, hidden_layers=hiddenlayers)
    optimizer = optim(model.parameters(), lr = learningrate, weight_decay=weightdecay)
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
                total_loss = total_loss + loss.item()

        return total_loss / len(dataloader)




    def predict_ratings(model, dataset, device):
        "We will comåute the scores for user and book pairs in train, val, test dataframe"
        model.eval() #stop training, use the fixed wieghts
        with torch.no_grad():
            #map userID to index again
            user_id = dataset["userId"].map(user_to_index).values
            book_id = dataset["itemId"].map(book_to_index).values

            #Make rows from columns for user and corresponding books
            user_tensor = torch.tensor(user_id, dtype=torch.long).to(device) 
            book_tensor = torch.tensor(book_id, dtype=torch.long).to(device)

            user_em = model.user_emb(user_tensor) #Creating a 64 wentry row for each entry in the user_tensor
            book_em = model.item_emb(book_tensor)

            interaction = user_em * book_em 

            scores = model.perceptron(interaction).squeeze(-1).cpu().numpy() 
            #Take array by array and multiple userid vector with omvie, give one number for that calc.
            #|pandas needs data on CPU and so does nupmy + convert pytorch tensor to numpy, so that pandas can put it into a dataframe

        return scores
            



    #STEP 10 - Train the model
    def training_with_bpr(model, trainloader, validationloader, optimizer, stopearly, epochs, device):
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

    training_with_bpr(model = model, trainloader = train_bpr_dataloader, 
                    validationloader=validate_bpr_dataloader, 
                    optimizer=optimizer, 
                    stopearly = early_stop, 
                    epochs = 3000, 
                    device=device)


    #STEP 11 - Create predictions for validation set
    def validate_bpr():
        #Get preds in val dataset
        predicted_score = predict_ratings(model, validation_df, device)
        #Avr val loss pr batch
        average_val_loss_per_batch = evaluate_loss(model, validate_bpr_dataloader, device)
        #MEasure how much time this whole thing takes

        end_time = time.time()
        elapsed_sec = end_time - start_time

        #Try to get max GPU usage for entire program
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024*1024)
        else:
            max_memory = None

        #Make a csv file
        prediction_val_dataset = validation_df.copy()
        prediction_val_dataset["val_score"] = predicted_score
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

    validate_bpr()

    #STEP 12 - test the model
    def test_bpr(model, testdataloader, device):
        average_test_loss_per_batch = evaluate_loss (model, testdataloader, device)
        #predict stuff using the test_df 
        test_predict_score = predict_ratings(model, test_df, device)
        
        #Tildel prediction til test datasæt
        prediction_test_dataset = test_df.copy()
        prediction_test_dataset["test_score"] = test_predict_score
        prediction_test_dataset.to_csv(prediction_test_save, index = False)

        with open(prediction_test_save, "a") as file:
            file.write("\n")
            file.write(f"# Average testing loss per batch for one epoch: {average_test_loss_per_batch:.4f}\n")

    #STEP 13 : TEST IT 
    test_bpr(model, test_bpr_dataloader, device)


a1 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr0001_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr0001_batch64.csv")
    
a2 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr0001_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr0001_batch64.csv")
    


a3 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr00003_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr00003_batch64.csv")
    
a4 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr00003_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr00003_batch64.csv")
    

a5 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr0001_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr0001_batch128.csv")


a6 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr0001_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr0001_batch128.csv")




a7 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr00003_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed64_lr00003_batch128.csv")




a8 =  run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr00003_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_OneLayer_embed32_lr00003_batch128.csv")












b1 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr0001_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr0001_batch64.csv")
    
b2 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr0001_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr0001_batch64.csv")
    


b3 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr00003_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr00003_batch64.csv")
    
b4 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr00003_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr00003_batch64.csv")
    

b5 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr0001_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr0001_batch128.csv")


b6 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr0001_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr0001_batch128.csv")




b7 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr00003_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed64_lr00003_batch128.csv")




b8 =  run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr00003_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_TwoLayers_embed32_lr00003_batch128.csv")




























c1 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr0001_batch64.csv")
    
c2 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr0001_batch64.csv")
    


c3 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr00003_batch64.csv")
    
c4 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 64,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr00003_batch64.csv")
    

c5 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr0001_batch128.csv")


c6 = run_program(
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.001,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr0001_batch128.csv")




c7 = run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 64,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed64_lr00003_batch128.csv")




c8 =  run_program( 
                optim = torch.optim.Adam,
                weightdecay = 1e-5,
                batchsize = 128,
                hiddenlayers = [128, 64, 32],
                learningrate = 0.0003,
                embedding_length = 32,
                prediction_val_save = "data/Output_Predictions_val_100K_books(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
                prediction_test_save = "data/Output_Predictions_test_100K_mbooks(MLPwithBPR)/BPRnn_ThreeLayers_embed32_lr00003_batch128.csv")
