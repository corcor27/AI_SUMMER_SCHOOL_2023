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
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

parser = argparse.ArgumentParser()

###dataset options
parser.add_argument('--mode',type=str, default="train", help='train, inference')
parser.add_argument('--train_excel',type=str, default = "Iris/Iris.csv", help='name of training excel')

parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["virginica", "versicolor", "setosa"])
#
parser.add_argument('--validation_split', type=float, default=0.7, help='maximum epoch number to train')

### model parameters
parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')

parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--aug', default = False, help='do you want augmentation? true or false')
parser.add_argument('--img_colour', default = True, help='do you want to use colour')
parser.add_argument('--Starting_lr', type=float,  default=0.0001,help='intial lr')

###output files

parser.add_argument('--weight_name', type=str, default="Iris_model.h5", help='name of weights file')
parser.add_argument('--Tensorboard_name', type=str, default="Tensorboard_model", help='name of weights file')


args = parser.parse_args()
print(args)
args.features_columns = ["PetalLengthCm", "PetalWidthCm"]
print(args.features_columns)

### open excel sheets in pandas
training_data = pd.read_csv(args.train_excel)

# find data split value
split_value = int(round(training_data.shape[0] * args.validation_split, 0))

train_data, validation_data = training_data.iloc[:split_value], training_data.iloc[split_value:]

print(train_data.shape, validation_data.shape)

#### load images into dataloader

train_set = DataGenerator(train_data, args, shuffle=True)
validation_set = DataGenerator(validation_data, args)


####Creating model

def Simple_model(args, filters=16):
    inputA = layers.Input(shape=(len(args.features_columns)))
    x = layers.Dense(filters*2, activation='linear')(inputA)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(filters, activation='linear')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(int(round(filters/2,0)), activation='linear')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
    fc = layers.Dense(len(args.prediction_classes), activation="softmax")(x)
    model = Model(inputs=inputA, outputs=fc)
    return model

model = Simple_model(args)

loss_fn = keras.losses.CategoricalCrossentropy() ## binary loss function as there are two classes
optimizer = tf.keras.optimizers.Adam(learning_rate=args.Starting_lr) ##adam optimizer as it very robust
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()## three classes meaning binary accuracy, catigorical if more than 2

model.compile(loss=loss_fn, optimizer=optimizer, metrics=train_acc_metric)
tensorboard = TensorBoard(log_dir= args.Tensorboard_name)
checkpointer = ModelCheckpoint(filepath=args.weight_name, monitor='val_categorical_accuracy',mode='max', verbose = 1, save_best_only=True, save_weights_only = True)
callbacks = [tensorboard, checkpointer, ReduceLROnPlateau(monitor=('val_loss'), factor=0.8, patience=2, cooldown=0, min_lr=0.00000001, verbose=1)]
if args.mode == "train":
    model.fit(train_set,validation_data=validation_set, use_multiprocessing=True, workers=6, epochs = args.max_epochs, callbacks = callbacks)
    model.save_weights(args.weight_name)
elif args.mode == "inference":
    model.load_weights(args.weight_name)
    model.evaluate(validation_set)
elif args.features == "features":
    model.load_weights(args.weight_name)
    







    
          
    
                
                
            
    

