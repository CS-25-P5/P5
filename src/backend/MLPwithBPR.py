import os
import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pandas
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
#import torchvision
#import tqdm


#STEP 1 - Redo the database - now I need movies even if they have no ratings



#STEP 2 - Create the mebedding vectors with the metadata ()



#STEP 3 - Forward propagate with model


# STEP 4 - calculate the loss function (BPR)



# STEP 5 - Backprop and epochs