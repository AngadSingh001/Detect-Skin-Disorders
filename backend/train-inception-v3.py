import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

metadata = pd.read_csv('CODE/DATA/HAM10000_metadata.csv')
metadata["filename"] = metadata["image_id"] + ".jpg"

input_shape = (224, 224, 3)

inception = InceptionV3(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([inception, Flatten(), Dense(512, activation='relu'),
                   BatchNormalization(), Dropout(0.5), Dense(7, activation='softmax')])

for layer in inception.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
target_size = (224, 224)

train_data = train_datagen.flow_from_dataframe(dataframe=metadata, directory='CODE/DATA/test', x_col='filename',
                                               y_col='dx', target_size=target_size, batch_size=32, class_mode='categorical', subset='training')

val_data = train_datagen.flow_from_dataframe(dataframe=metadata, directory='CODE/DATA/test', x_col='filename',
                                             y_col='dx', target_size=target_size, batch_size=32, class_mode='categorical', subset='validation')

test_data = test_datagen.flow_from_dataframe(dataframe=metadata, directory='CODE/DATA/test', x_col='filename',
                                             y_col='dx', target_size=target_size, batch_size=1, class_mode='categorical', shuffle=False)

history = model.fit(train_data, steps_per_epoch=len(
    train_data), epochs=1, validation_data=val_data, validation_steps=len(val_data))

model.save('CODE/PYTHON/MODELS/skin_cancer_inception.h5')
