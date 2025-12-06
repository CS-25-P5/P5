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



def reorder(input_folder):

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename) #### GEt the the path for each file


            #Load me
            pred_dataset = pandas.read_csv(input_path) 

            #Sorty by userId first, then descend

            sort_by_id = pred_dataset.sort_values(
                by = ["userId", "test_score"],
                ascending=[True, False]
            )

            #Create output filename 
            base_path, extension = os.path.splitext(filename)
            output_filename = f"{base_path}_ranked{extension}"
            output_path = os.path.join(input_folder, output_filename)

            #Save
            sort_by_id.to_csv(output_path, index = False)

