import pandas as pd
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

Path_to_csv = "Iris/Iris.csv"

data = pd.read_csv(Path_to_csv)
data = shuffle(data, random_state=42)
X = data.drop(['target','Species'], axis=1)

# converting into numpy array and assigning petal length and petal width
X = X.to_numpy()[:, (2,3)]
y = data['target']
# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.6, random_state=42)


log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
training_prediction = log_reg.predict(X_train)
test_prediction = log_reg.predict(X_test)

print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recall scores
print(metrics.classification_report(y_train, training_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))

print("Precision, Recall, Confusion matrix, in testing\n")

# Precision Recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))




