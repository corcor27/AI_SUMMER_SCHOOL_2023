import pandas as pd
import os 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from data_generator import DataGenerator
from tensorflow.keras import layers, Model

parser = argparse.ArgumentParser()

###dataset options
parser.add_argument('--mode',type=str, default="train", help='train, inference')
parser.add_argument('--train_excel',type=str, default = "train.xlsx", help='name of training excel')
parser.add_argument('--test_excel',type=str, default = "test.xlsx", help='name of test excel')
parser.add_argument('--image_paths_column',type=str, default = "File_paths", help='column name in excel that detail image names')
parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["Cats", "Dogs"])#, "Low"])
parser.add_argument('--validation_split', type=float, default=0.7, help='maximum epoch number to train')

### model parameters
parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,default=100, help='input patch size of network input')
parser.add_argument('--aug', default = False, help='do you want augmentation? true or false')
parser.add_argument('--img_colour', default = True, help='do you want to use colour')
parser.add_argument('--Starting_lr', type=float,  default=0.001,help='intial lr')

###output files

parser.add_argument('--weight_name', type=str, default="Cats_Dogs_model.h5", help='name of weights file')


args = parser.parse_args()
print(args)


### open excel sheets in pandas
training_data = pd.read_excel(args.train_excel)
test_data = pd.read_excel(args.test_excel)

# find data split value
split_value = int(round(training_data.shape[0] * args.validation_split, 0))

train_data, validation_data = training_data.iloc[:split_value], training_data.iloc[split_value:]

print(train_data.shape, validation_data.shape, test_data.shape)

#### load images into dataloader

train_set = DataGenerator(train_data, args, shuffle=True, augmentation = True)
validation_set = DataGenerator(validation_data, args)
test_set = DataGenerator(test_data, args)


####Creating model

def Simple_model(args, filters = 32):
    if args.img_colour:
        N_channels = 3
    else:
        N_channels = 1
    inputA = layers.Input(shape=(args.img_size, args.img_size, N_channels))
    
    x = layers.Conv2D(filters, (7, 7), strides=(2, 2), activation='relu')(inputA)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(filters, (3,3), activation='relu')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(filters, (3,3), activation='relu')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    dropout = layers.Dropout(0.3)(x)
    flattened = layers.Flatten()(dropout)
    fc = layers.Dense(len(args.prediction_classes), activation="softmax")(flattened)
    model = Model(inputs=inputA, outputs=fc)
    return model

model = Simple_model(args)

loss_fn = keras.losses.BinaryCrossentropy() ## binary loss function as there are two classes
optimizer = tf.keras.optimizers.Adam(learning_rate=args.Starting_lr) ##adam optimizer as it very robust
train_acc_metric = keras.metrics.BinaryAccuracy() ## two classes meaning binary accuracy, catigorical if more than 2

model.compile(loss=loss_fn, optimizer=optimizer, metrics=train_acc_metric)
if args.mode == "train":
    model.fit(train_set,validation_data=validation_set, use_multiprocessing=True, workers=6, epochs = args.max_epochs)
    model.save_weights(args.weight_name)
elif args.mode == "inference":
    model.load_weights(args.weight_name)
    model.evaluate(test_set)
elif args.features == "features":
    model.load_weights(args.weight_name)
    







    
          
    
                
                
            
    

