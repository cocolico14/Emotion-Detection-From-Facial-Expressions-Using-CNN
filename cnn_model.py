import pickle
import time
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D,  \
    Dropout, Dense, Input, Flatten, \
    BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def create_model(co, dr, lr):
    ''' create CNN model
    co -- layers configuration (num of node in each layer)
    dr -- dropout rate
    lr -- learning rate
    '''
  
    input_img = Input(shape=(48, 48, 1))
    
    x = Conv2D(co[0], (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
    
    x = Conv2D(co[1], (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
    
    x = Conv2D(co[2], (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
    if dr[0] != 0: x = Dropout(dr[0])(x)
    
    x = Flatten()(x)
    x = Dense(co[3], activation='relu')(x)
    x = BatchNormalization()(x)
    if dr[1] != 0: x = Dropout(dr[1])(x)
    x = Dense(co[4], activation='relu')(x)
    x = BatchNormalization()(x)
    if dr[2] != 0: x = Dropout(dr[2])(x)
    x = Dense(co[5], activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=x)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
        
    return model


DROPOUT = [(0.25, 0.25, 0.15), (0, 0.25, 0.15), (0.25, 0.25, 0), (0, 0.25, 0)]
CONFIG = [(16, 32, 64, 256, 128, 64), (32, 64, 128, 512, 256, 64), (64, 128, 256, 512, 256, 64)]
LR = [1e-4, 3e-4, 6e-4, 1e-3]
BATCH_SIZE = 64
DATAGEN = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False
)

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

MODEL_DIR = "./Models/"

X = tf.keras.utils.normalize(X, axis=1)

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=45)

for i, (train_index, test_index) in enumerate(skf.split(X, Y)):

    X_train, X_test = np.array([X[i] for i in train_index]), np.array([X[i] for i in test_index])
    Y_train, Y_test = np.array([Y[i] for i in train_index]), np.array([Y[i] for i in test_index])

    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=5)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=5)

    for n in CONFIG:
      for j in DROPOUT:
        for k in LR:
          model = create_model(n, j, k)

          filepath = MODEL_DIR + "fold_" + str(i+1) + ".hdf5"
          print(j, k, n)

          checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
          es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, mode='min')

          model.fit_generator(DATAGEN.flow(X_train, Y_train, batch_size=BATCH_SIZE), epochs=100,
            validation_data=(X_test, Y_test), steps_per_epoch=len(X_train)//BATCH_SIZE,
            callbacks=[checkpoint, es])