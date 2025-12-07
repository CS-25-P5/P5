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

input1 = "data/Output_Predictions_test_1M_movies(MLPwithBPR)"
input2 = "data/Output_Predictions_test_100K_goodbooks(MLPwithBPR)"
input3 = "data/Output_Predictions_test_100K_movies(MLPwithBPR)"

input4 = "data/Output_Predictions_test_100K_goodbooks(MLPwithGenres)"
input5 = "data/Output_Predictions_test_100K_movies(MLPwithGenres)"

def reorder(input_folder):

    for filename in os.listdir(input_folder):
        if filename.endswith("8.csv") or filename.endswith("4.csv"):
            input_path = os.path.join(input_folder, filename) #### GEt the the path for each file


            #Load me
            pred_dataset = pandas.read_csv(input_path, comment = "#") 

            #Sorty by userId first, then descend

            sort_by_id = pred_dataset.sort_values(
                by = ["userId", "test_rating"],
                ascending=[True, False]
            )

            #Create output filename 
            base_path, extension = os.path.splitext(filename)
            output_filename = f"{base_path}_ranked{extension}"
            output_path = os.path.join(input_folder, output_filename)

            #Save
            sort_by_id.to_csv(output_path, index = False)


'''
reorder(input1)
reorder(input2)
reorder(input3)

reorder(input4)
reorder(input5)
'''
