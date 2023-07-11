import pandas as pd
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.utils import shuffle

Dataset_folder = "cats_dogs"



Dataset_allocation = os.listdir(Dataset_folder)

for option in Dataset_allocation:
    data = pd.DataFrame()
    File_names = []
    File_paths = []
    Cats = []
    Dogs = []
    Class_path = os.path.join(Dataset_folder, option)
    Class_options = os.listdir(Class_path)
    for Class in Class_options:
        Imgs_path = os.path.join(Class_path,  Class)
        Imgs_list = os.listdir(Imgs_path)
        for img in Imgs_list:
            Img_path = os.path.join(Imgs_path, img)
            File_names.append(img)
            File_paths.append(Img_path)
            if Class == "dogs":
                Dogs.append(1)
                Cats.append(0)
            else:
                Dogs.append(0)
                Cats.append(1)
    data["File_names"] = File_names
    data["File_paths"] = File_paths
    
    data["Dogs"] = Dogs
    data["Cats"] = Cats
    data = shuffle(data)
    data.to_excel("{}.xlsx".format(option))
    
                
    
                
                
            
    

