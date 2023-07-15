import numpy as np
import keras
from sklearn.utils import shuffle
import cv2
import imgaug.augmenters as iaa
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, args, shuffle=False, augmentation=False):
        'Initialization'
        self.dataset = dataset
        self.args = args
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.n = len(self.dataset)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n / self.args.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        images = []
        labels = []
        # Generate indexes of the batch
        batches = self.dataset[index *self.args.batch_size:(index + 1) * self.args.batch_size]
        for ind in range(batches.shape[0]):
            img_path =  str(batches[self.args.image_paths_column].iloc[ind])
            image = self.__get_image(img_path)
            if self.augmentation:
                image = self.__apply_augmentation(image)
            images.append(image)
            # Generate data
            labels.append([batches[self.args.prediction_classes].iloc[ind].values])
        images = np.array(images, np.float32)
        labels = np.array(labels, np.int64)
        labels = np.squeeze(labels)
        return images, labels
    
    def __get_image(self, path):
        ## load image
        image = cv2.imread(path)
        
        image = cv2.resize(image, (self.args.img_size, self.args.img_size),interpolation=cv2.INTER_CUBIC)
        image = image/255.0
        return image
    
    def __apply_augmentation(self, image):
        random_number = random.randint(1, 5)
        if random_number == 1:
            image = self.__aug_flip_hr(image)
        elif random_number == 2:
            image = self.__aug_flip_vr(image)
        elif random_number == 3:
            image = self.__aug_rotation(image, 30)
        return image
        
        
        return image
    def __aug_flip_hr(self, img):
        hflip = iaa.Fliplr(p=1.0)
        img_hf = hflip.augment_image(img)

        return img_hf

    # Augmentation Vertical Flip
    def __aug_flip_vr(self, img):
        vflip = iaa.Flipud(p=1.0)
        img_vf = vflip.augment_image(img)

        return img_vf

    # Augmentation Rotation
    def __aug_rotation(self, img, rot_deg):
        rot1 = iaa.Affine(rotate=(-rot_deg, rot_deg))
        img_rot = rot1.augment_image(img)

        return img_rot
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset = shuffle(self.dataset)
            self.n = len(self.dataset)

    