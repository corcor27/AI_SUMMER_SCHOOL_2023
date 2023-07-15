import pandas as pd
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

Path_to_csv = "Iris/Iris.csv"

data = pd.read_csv(Path_to_csv)

unique_elements = list(set(list(data["Species"])))

target = []
virginica = []
versicolor = []
setosa = []

for row in range(0, data.shape[0]):
    Class = data["Species"].iloc[row]
    Position = unique_elements.index(Class)
    target.append(Position)
    if Position == 0:
        virginica.append(1)
        versicolor.append(0)
        setosa.append(0)
    elif Position == 1:
        virginica.append(0)
        versicolor.append(1)
        setosa.append(0)
    elif Position == 2:
        virginica.append(0)
        versicolor.append(0)
        setosa.append(1)

data["target"] = target
data["virginica"] = virginica
data["versicolor"] = versicolor
data["setosa"] = setosa




data.to_csv(Path_to_csv)

data.groupby('Species').size()

data.describe()
    
    
                
                
            
    

