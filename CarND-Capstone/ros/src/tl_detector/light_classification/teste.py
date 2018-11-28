import csv
import cv2  
import numpy as np
import os.path

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import load_model
import numpy as np


model = Sequential()

model.add(Lambda(lambda x:x/255-0.5,input_shape=(800,600,3)))
model.add(Convolution2D(12, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (2, 2), activation="relu"))
model.add(Convolution2D(64, (2, 2), activation="relu"))
model.add(Convolution2D(64, (2, 2), activation="relu"))

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3, activation = 'softmax'))

model.load_weights('model_weights2.h5')


image = cv2.imread('Img_0.png')
image = cv2.resize(image,(600,800))
image = np.expand_dims(image, axis=0)
signal = model.predict(image)
signal_out = np.argmax(signal)
print(signal)
print(signal_out)