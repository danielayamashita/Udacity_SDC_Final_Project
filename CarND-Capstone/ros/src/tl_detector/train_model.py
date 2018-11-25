#!/usr/bin/env python

###########################################
##             Load images               ##
###########################################

import os
import cv2
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

img_root_folder = "./TLdataset02"
def load_images(folder, label):
    images = []
    image_folder = "{}/{}".format(img_root_folder, folder)
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        image = cv2.imread(image_folder + '/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        images.append(image)

    labels = [label for x in range(len(images))]
    return zip(images, labels)

labeled_data = []
green = load_images("green", 2)
labeled_data.extend(green)
red = load_images("red", 0)
labeled_data.extend(red)
yellow = load_images("yellow", 1)
labeled_data.extend(yellow)
#unknown = load_images("unknown", 3) # JUST TO HAVE NO GAPS, IS ACTUALLY 4 IN THE MESSAGE TrafficLight 
#labeled_data.extend(unknown)

np.random.shuffle(labeled_data)
# splits the entire set into training and test data
train_samples, test_samples = train_test_split(labeled_data, test_size=0.1)
# the training data is split further into training and validation
train_samples, validation_samples = train_test_split(train_samples, test_size=0.2)

X_train, y_train = zip(*train_samples) 
X_valid, y_valid = zip(*validation_samples) 
X_test, y_test = zip(*test_samples) 

###########################################
##             Summary                   ##
###########################################

n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)

# Here is assumed, that all images are of the same size
image_shape = X_train[0].shape

# Assuming that each label occurs at least once in each set.
# Otherwise the data split training/validation/test would have been insufficient
assert( (len(np.unique(y_train)) == len(np.unique(y_valid))) and (len(np.unique(y_train)) == len(np.unique(y_test))) )
n_classes = len(np.unique(y_valid))

print("Number of examples =", len(labeled_data))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of test examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
#print("y_train = ", y_train)

###########################################
##        Data preparation               ##
###########################################

# transform the input data to tensors
from keras.preprocessing import image
from keras.utils import to_categorical

from tqdm import tqdm
def image_to_tensor(img):
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
def images_to_tensor(imgs):
    list_of_tensors = [image_to_tensor(img) for img in imgs]
    return np.vstack(list_of_tensors)
# to tensors and normalize it
train_tensors = images_to_tensor(X_train).astype('float32')/255
valid_tensors = images_to_tensor(X_valid).astype('float32')/255
test_tensors = images_to_tensor(X_test).astype('float32')/255

print("train_tensors shape = ", train_tensors.shape)

y_train = to_categorical(y_train, num_classes=n_classes)
y_valid = to_categorical(y_valid, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

print(y_train.shape)


###########################################
##        Model architecture             ##
###########################################

# load keras libraies and load the MobileNet model
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint  
from keras import applications
from keras import optimizers
from keras.models import load_model

#model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
model = applications.mobilenet.MobileNet(input_shape = image_shape)

# add the custom layers
x = model.output
#x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(n_classes, activation="softmax")(x)

# creating the final modal
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

model_final.summary()

###########################################
##        Training and storage           ##
###########################################

# train the model.
epochs = 20
batch_size = 16

model_filepath = 'saved_models/model.MobileNet-3-classes.h5'

checkpointer = ModelCheckpoint(filepath=model_filepath, 
                               verbose=1, save_best_only=True)

model_final.fit(train_tensors, y_train, 
          validation_data=(valid_tensors, y_valid),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

###########################################
##        Final model test               ##
###########################################

# load the trained model
from keras.utils.generic_utils import CustomObjectScope

del model
with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
    model = load_model(model_filepath)

# get index of predicted signal sign for each image in test set
signal_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# print out test accuracy
test_accuracy = 100*np.sum(np.array(signal_predictions)==np.argmax(y_test, axis=1))/len(signal_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
