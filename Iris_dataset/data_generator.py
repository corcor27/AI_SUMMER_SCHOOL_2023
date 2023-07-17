import numpy as np
import keras
from sklearn.utils import shuffle
import cv2
import imgaug.augmenters as iaa
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, args, shuffle=False):
        'Initialization'
        
        self.args = args
        for item in self.args.features_columns:

            dataset[self.args.features_columns] = dataset[self.args.features_columns]/np.max(dataset[self.args.features_columns])

        self.dataset = dataset
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
        for ind in range(batches.shape[0]):
            features.append([batches[self.args.features_columns].iloc[ind].values])
            # Generate data
            labels.append([batches[self.args.prediction_classes].iloc[ind].values])
        features = np.array(features, np.float32)
        features = np.squeeze(features)
        labels = np.array(labels, np.int64)
        labels = np.squeeze(labels)
        return features, labels
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset = shuffle(self.dataset)
            self.n = len(self.dataset)

    