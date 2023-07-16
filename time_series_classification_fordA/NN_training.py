import pandas as pd
import os 
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from data_generator import DataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

###dataset options
parser.add_argument('--mode',type=str, default="train", help='train, inference')
parser.add_argument('--train_excel',type=str, default = "FordA_TRAIN.xlsx", help='name of training excel')
parser.add_argument('--test_excel',type=str, default = "FordA_TEST.xlsx", help='name of test excel')
parser.add_argument('--image_paths_column',type=str, default = "File_paths", help='column name in excel that detail image names')
parser.add_argument('-n' , '--prediction_classes', nargs='+', default =[0,1])#, "Low"])
parser.add_argument('--validation_split', type=float, default=0.7, help='maximum epoch number to train')

### model parameters
parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,default=100, help='input patch size of network input')
parser.add_argument('--aug', default = False, help='do you want augmentation? true or false')
parser.add_argument('--img_colour', default = True, help='do you want to use colour')
parser.add_argument('--Starting_lr', type=float,  default=0.001,help='intial lr')
parser.add_argument('--Tensorboard_name', type=str, default="Tensorboard_model", help='name of weights file')
###output files

parser.add_argument('--weight_name', type=str, default="FordA_model.h5", help='name of weights file')


args = parser.parse_args()
print(args)
args.features_columns = 501

### open excel sheets in pandas
training_data = pd.read_excel(args.train_excel)
test_data = pd.read_excel(args.test_excel)

# find data split value
split_value = int(round(training_data.shape[0] * args.validation_split, 0))

train_data, validation_data = training_data.iloc[:split_value], training_data.iloc[split_value:]

print(train_data.shape, validation_data.shape, test_data.shape)

#### load images into dataloader

train_set = DataGenerator(train_data, args, shuffle=True)
validation_set = DataGenerator(validation_data, args)
test_set = DataGenerator(test_data, args)


####Creating model

def Simple_model(args, filters = 64):
    
    input_layer = layers.Input(shape=(args.features_columns,1))

    conv1 = keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(len(args.prediction_classes), activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = Simple_model(args)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.Adam(learning_rate=args.Starting_lr) ##adam optimizer as it very robust

tensorboard = TensorBoard(log_dir= args.Tensorboard_name)
checkpointer = ModelCheckpoint(filepath=args.weight_name, monitor='sparse_categorical_accuracy',mode='max', verbose = 1, save_best_only=True, save_weights_only = True)
callbacks = [tensorboard, checkpointer, ReduceLROnPlateau(monitor=('val_loss'), factor=0.8, patience=2, cooldown=0, min_lr=0.00000001, verbose=1)]
model.compile(loss=loss_fn, optimizer=optimizer, metrics=acc_metric)
if args.mode == "train":
    model.fit(train_set,validation_data=validation_set, use_multiprocessing=True, workers=6, epochs = args.max_epochs, callbacks = callbacks)
    model.save_weights(args.weight_name)
elif args.mode == "inference":
    model.load_weights(args.weight_name)
    model.evaluate(test_set)
elif args.features == "features":
    model.load_weights(args.weight_name)
    







    
          
    
                
                
            
    

