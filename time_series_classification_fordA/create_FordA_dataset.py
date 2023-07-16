import pandas as pd
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import string
import itertools

Path_to_train_txt = ["FordA_TRAIN.txt", "FordA_TEST.txt"]
for item in Path_to_train_txt:
    f = open(item, "r")
    data = pd.DataFrame()
    for x in f:
        split_string = x.split()
        cast_string = np.float_(split_string)
        create_array = np.array(cast_string)
        create_array = np.expand_dims(create_array, axis=0)
        frame = pd.DataFrame(create_array)
        data = pd.concat([data, frame])
    print(data.shape)
    print(data.shape)
    data.to_excel(item.replace(".txt", ".xlsx"))







    
    
                
                
            
    

