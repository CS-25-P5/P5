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



def reorder(input = ""):
    pred_dataset = pandas.read_csv(input) 
    grouped_data = pred_dataset.groupby("userId")
    grouped_data.sort_values(by=["test_predicted_rating"], ascending = False)