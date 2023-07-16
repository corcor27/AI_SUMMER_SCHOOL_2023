import pandas as pd
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2


import string
import itertools

number_of_plot_sample = 3

Path_to_train = "FordA_TRAIN.xlsx"
training_data = pd.read_excel(Path_to_train)
backup_data = training_data.copy()
print(training_data.shape)
training_data.drop(columns= [training_data.columns[0], training_data.columns[1]], axis=1, inplace=True)
print(training_data.shape)
#training_data.plot()
for ii in range(0, number_of_plot_sample):
    sample = training_data.iloc[ii]
    sample.plot(label="sample_{}".format(ii))
plt.legend()








    
    
                
                
            
    

