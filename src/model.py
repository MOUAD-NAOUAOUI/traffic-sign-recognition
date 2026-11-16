import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras import layers, models
from src.dataset import create_data_augmentation, IMG_SIZE, NUM_CLASSES

def build_traffic_sign_cnn(input_shape=(IMG_SIZE[0],IMG_SIZE[1],3),
                           num_classes=NUM_CLASSES):
    inputs=layers.Input(shape=input_shape,name="image_input")
    x = create_data_augmentation()(inputs)
    x = layers.Conv2D(32,(3,3),padding='same',activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32,(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64,(3,3),padding='same',activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128,(3,3),padding='same',activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs =layers.Dense(num_classes,activation='softmax',name='classification')(x)
    model = models.Model(inputs=inputs,outputs=outputs,name='traffic_sign_cnn')
    return model

if __name__=="__main__":
    model = build_traffic_sign_cnn()
    model.summary()

