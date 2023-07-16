import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
import cv2
import imgaug.augmenters as iaa
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, args, shuffle=False):
        'Initialization'
        self.dataset = dataset
        self.args = args
        self.shuffle = shuffle
        self.n = len(self.dataset)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n / self.args.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        features = []
        labels = []
        # Generate indexes of the batch
        batches = self.dataset[index *self.args.batch_size:(index + 1) * self.args.batch_size]
        batch_labels = batches.copy()
        batch_labels = batch_labels[0]
        batches.drop(columns= batches.columns[0], axis=1, inplace=True)
        for ind in range(batches.shape[0]):
            feature = batches.iloc[ind]
            feature = np.expand_dims(feature, axis=1)
            features.append(feature)
            # Generate data
            if batch_labels.iloc[ind] == -1:
                labels.append([1])
            else:
                labels.append([0])
            
            
        features = np.array(features, np.float32)
        labels = np.array(labels, np.int64)
        labels = np.squeeze(labels)
        #labels = np.expand_dims(labels, axis=-1)
        return features, labels
    
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset = shuffle(self.dataset)
            self.n = len(self.dataset)

    
