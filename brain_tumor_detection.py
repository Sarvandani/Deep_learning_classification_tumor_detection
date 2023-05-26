#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##The data can be downloaded from the following link: 
##https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?datasetId=165566&sortBy=voteCount
@author: Sarvandani
"""
import numpy as np 
import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import img_to_array, load_img
from matplotlib.pyplot import imshow
plt.style.use('dark_background')
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

###############################
no_dir = '/volumes/DRIVE_2/DATA_SCI_TOMOR_DETECTION/brain_tumor_dataset/no/'
yes_dir = '/volumes/DRIVE_2/DATA_SCI_TOMOR_DETECTION/brain_tumor_dataset/yes/'
######################
###Histogram
img_to_array(load_img(os.path.join(no_dir , os.listdir(no_dir)[5]))).shape
dst = pd.DataFrame()
dst['INITIAL CLASSIFICATION'] = ['Yes_TUMOR'] * len(os.listdir(yes_dir)) + ['NO_TUMOR'] * len(os.listdir(no_dir))
sns.countplot(x='INITIAL CLASSIFICATION', data=dst)
####################
# image visualizations
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(load_img(os.path.join(yes_dir , os.listdir(yes_dir)[0])))
plt.title('yes')
plt.subplot(2, 2, 2)
plt.imshow(load_img(os.path.join(yes_dir , os.listdir(yes_dir)[1])))
plt.title('yes')
plt.subplot(2, 2, 3)
plt.imshow(load_img(os.path.join(no_dir , os.listdir(no_dir)[0])))
plt.title('no')
plt.subplot(2, 2, 4)
plt.imshow(load_img(os.path.join(no_dir , os.listdir(no_dir)[1])))
plt.title('no')

plt.show()
###########################################################

data = [] #creating a list for images
paths = [] #creating a list for paths
labels = [] #creating a list to put our 0 or 1 labels

#staring with the images that have tumors
for r, d, f in os.walk('/volumes/DRIVE_2/DATA_SCI_TOMOR_DETECTION/brain_tumor_dataset/yes/'):
    for file in f:
        if '.jpg' in file:
         
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        labels.append(1)


paths = []
for r, d, f in os.walk('/volumes/DRIVE_2/DATA_SCI_TOMOR_DETECTION/brain_tumor_dataset/no/'):
    for file in f:
        if '.jpg' in file:
        
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        labels.append(0)
        
data = np.array(data)
data.shape

labels = np.array(labels)
#labels = labels.reshape(139,1)
labels = labels.reshape(data.shape[0],1)
print('data shape is:', data.shape)
print('labels shape is:', labels.shape)

###################################
#reducing the data between 1 and 0
data = data / 255.00
#max data
print(np.max(data))
#getting the min of the array
print(np.min(data))
####################################
#splitting train and test
x_train,x_test,y_train,y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=7)
print("shape of our training data:",x_train.shape)
print("shape of our training labels:",y_train.shape)
print("shape of our test data:",x_test.shape)
print("shape of our test labels:",y_test.shape)
################################################################
## defining model
model = keras.Sequential([
    
    layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu", padding='same', input_shape=[128, 128, 3]),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.MaxPool2D(),
    
    layers.Flatten(),
    layers.Dropout(.25),
    layers.Dense(units=256, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
model.summary()
################################
model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01), loss='binary_crossentropy', metrics=['accuracy'])
#including early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)
######################################
history = model.fit(x = x_train,y = y_train, validation_data= (x_test,y_test),
    batch_size = 99,
    epochs=200,
    callbacks=[early_stopping],
    verbose=(2),)
##############################@
#plots
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss','val_loss']].plot();
history_frame.loc[:, ['accuracy','val_accuracy']].plot();
########################################################
img = Image.open('/volumes/DRIVE_2/DATA_SCI_TOMOR_DETECTION/brain_tumor_dataset/yes/Y11.jpg')
ax = np.array(img.resize((128,128)))
ax = ax.reshape(1,128,128,3)
res = model.predict_on_batch(ax)
imshow(img)
print('PREDICTION=',str(res[0]),'----close to 0 is NOT TUMOR, close to 1 is TUMOR')


































