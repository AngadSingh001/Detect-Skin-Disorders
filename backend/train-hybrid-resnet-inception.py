import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

input_shape = (224, 224, 3)
num_classes = 7

inception_model = InceptionV3(
    input_shape=input_shape, include_top=False, weights="imagenet")

resnet_model = ResNet50(input_shape=input_shape,
                        include_top=False, weights="imagenet")

for layer in inception_model.layers:
    layer.trainable = False
for layer in resnet_model.layers:
    layer.trainable = False

input_layer = Input(shape=input_shape)
x = inception_model(input_layer)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
inception_output = Dense(num_classes, activation='softmax')(x)

y = resnet_model(input_layer)
y = GlobalAveragePooling2D()(y)
y = Dense(1024, activation='relu')(y)
resnet_output = Dense(num_classes, activation='softmax')(y)

model = tf.keras.Model(inputs=input_layer, outputs=[
                       inception_output, resnet_output])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

batch_size = 32
epochs = 1

metadata = pd.read_csv('CODE/DATA/HAM10000_metadata.csv')
metadata["filename"] = metadata["image_id"] + ".jpg"

df = metadata

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

train_generator = datagen.flow_from_dataframe(dataframe=train_df, directory='CODE/DATA/test', x_col='filename',
                                              y_col='dx', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

validation_generator = datagen.flow_from_dataframe(dataframe=validation_df, directory='CODE/DATA/test',
                                                   x_col='filename', y_col='dx', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

model.save('CODE/PYTHON/MODELS/skin_cancer_hybrid.h5')

