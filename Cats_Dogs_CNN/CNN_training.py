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
from sklearn.decomposition import PCA
import umap

parser = argparse.ArgumentParser()

###dataset options
parser.add_argument('--mode',type=str, default="train", help='train, inference, features')
parser.add_argument('--train_excel',type=str, default = "train.xlsx", help='name of training excel')
parser.add_argument('--test_excel',type=str, default = "test.xlsx", help='name of test excel')
parser.add_argument('--image_paths_column',type=str, default = "File_paths", help='column name in excel that detail image names')
parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["Cats", "Dogs"])#, "Low"])
parser.add_argument('--validation_split', type=float, default=0.7, help='maximum epoch number to train')

### model parameters
parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,default=256, help='input patch size of network input')
parser.add_argument('--aug', default = False, help='do you want augmentation? true or false')
parser.add_argument('--img_colour', default = True, help='do you want to use colour')
parser.add_argument('--Starting_lr', type=float,  default=0.001,help='intial lr')
parser.add_argument('--Tensorboard_name', type=str, default="Tensorboard_model", help='name of weights file')
###output files

parser.add_argument('--weight_name', type=str, default="Cats_Dogs_model.h5", help='name of weights file')


args = parser.parse_args()
print(args)
if args.mode == "features":
    args.batch_size = 1

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
    fc1 = layers.Dense(args.img_size, activation="relu")(flattened)
    N = layers.BatchNormalization(momentum=0.1, epsilon=0.00001)(fc1)
    fc2 = layers.Dense(len(args.prediction_classes), activation="softmax")(N)
    model = Model(inputs=inputA, outputs=fc2)
    return model

model = Simple_model(args)

loss_fn = keras.losses.BinaryCrossentropy() ## binary loss function as there are two classes
optimizer = tf.keras.optimizers.Adam(learning_rate=args.Starting_lr) ##adam optimizer as it very robust
train_acc_metric = keras.metrics.BinaryAccuracy() ## two classes meaning binary accuracy, catigorical if more than 2
tensorboard = TensorBoard(log_dir= args.Tensorboard_name)
checkpointer = ModelCheckpoint(filepath=args.weight_name, monitor='val_binary_accuracy',mode='max', verbose = 1, save_best_only=True, save_weights_only = True)
callbacks = [tensorboard, checkpointer, ReduceLROnPlateau(monitor=('val_loss'), factor=0.8, patience=2, cooldown=0, min_lr=0.00000001, verbose=1)]
model.compile(loss=loss_fn, optimizer=optimizer, metrics=train_acc_metric)
if args.mode == "train":
    model.fit(train_set,validation_data=validation_set, use_multiprocessing=True, workers=6, epochs = args.max_epochs, callbacks = callbacks)
    model.save_weights(args.weight_name)
elif args.mode == "inference":
    model.load_weights(args.weight_name)
    model.evaluate(test_set)
elif args.mode == "features":
    model.load_weights(args.weight_name)
    features = []
    Extract = Model(model.inputs, model.layers[-3].output)
    data = pd.concat([training_data, test_data], axis=0)
    all_images = DataGenerator(data, args)
    for ii in range(0, data.shape[0]):
        image, label = all_images[ii]
        #image = np.squeeze(image)
        val_pred = Extract([image], training=False)
        features.append(val_pred)

    features = np.array(features)
    features = np.squeeze(features)
    print(features.shape)
    clusterable_embedding = umap.UMAP()
    cluster = clusterable_embedding.fit_transform(features)
    pca = PCA(n_components=2)
    cluster2 = pca.fit_transform(features)
    data["X_PCA_POSITIONS"] = cluster2[:, 0]
    data["Y_PCA_POSITIONS"] = cluster2[:, 1]
    data["X_UMAP_POSITIONS"] = cluster[:, 0]
    data["Y_UMAP_POSITIONS"] = cluster[:, 1]
    for item in args.prediction_classes:
        df2 = data[(data[item] == 1)]
        plt.scatter(df2["X_PCA_POSITIONS"],
                    df2["Y_PCA_POSITIONS"], label=item)
    plt.legend()
    plt.title("Pca")
    plt.savefig("Pca_features.png")
    plt.close()
    
    for item in args.prediction_classes:
        df2 = data[(data[item] == 1)]
        plt.scatter(df2["X_UMAP_POSITIONS"],
                    df2["Y_UMAP_POSITIONS"], label=item)
    plt.legend()
    plt.title("UMAP")
    plt.show()
    plt.savefig("UMAP_features.png")
    plt.close()
    
    
    data.to_excel("cats_dogs_features.xlsx")
    
    







    
          
    
                
                
            
    

