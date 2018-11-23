"""
Following https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

Some of data processing is copied from Wen

I followed this tutorial for the fine-tuning
https://www.youtube.com/watch?v=OO4HD-1wRN8&t=0s&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=32

https://www.youtube.com/watch?v=-0Blng0Ww8c
"""

import numpy as np
import keras
from keras.applications import mobilenet
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix

### TODO: Load and prepare the data
SIM_FOLDER = '/home/workspace/CarND-Capstone/data/TLdataset02'

def load_dataset_bosch():
    pass

# define function to load the train, test datasets
def load_dataset_simulator(path):
    data = load_files(path)
    X = np.array(data['filenames'])
    y = np_utils.to_categorical(np.array(data['target']), 4)
    label = data['target_names']
    return X, y, label

def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224)) # PIL image
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_processed = mobilenet.preprocess_input(img_array_expanded_dims) # scales pixel values etc
    return img_processed

from tqdm import tqdm # that adds progree bar in the console
def paths_to_tensor(img_paths):
    list_of_tensors = [prepare_image(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# load the train, test dataset
X, y, y_names = load_dataset_simulator(SIM_FOLDER)
N_classes = y.shape[1]

print('There are %d total images' % X.shape[0])
print('There are {} kinds of signals: {}'.format(y.shape[1], y_names))

#split the data into training/validation/testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state =42)

# to tensors and normalize it
train_tensors = paths_to_tensor(X_train)
valid_tensors = paths_to_tensor(X_valid)
test_tensors = paths_to_tensor(X_test)

# Retrain a MobileNet
# MobileNet contains 88 layers
# It is considerably lighter than other CNN models and performs faster but with less accuracy

#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')#, include_top=False, input_shape=(image_size, image_size))

# Freeze the layers except the last 15 layers
for layer in mobilenet_model.layers[:-15]:
    layer.trainable = False

# Add a fully connected layer to the end
last_layer = mobilenet_model.layers[-6].output
predictions = Dense(N_classes, activation='softmax')(last_layer)
new_model = Model(inputs=mobilenet_model.input, outputs=predictions)

# Check the new model
#print(new_model.summary())

# Training the model
new_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 20
BATCH_SIZE = 128

new_model.fit(train_tensors, y_train, validation_data=(valid_tensors, y_valid),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
#new_model.fit_generator(train_batches, steps_per_epoch=2,
#                        validation_data=valid_batches, validation_steps=2, epochs=30, verbose=2)


new_model.save('my_model_1.h5')  # creates a HDF5 file 'my_model.h5'

print('Test dataset')
score = new_model.evaluate(test_tensors, y_test, verbose=0)
print('Loss: {}, Accuracy: {}'.format(score[0], score[1]))

#Check with test data
predictions = new_model.predict(test_tensors) # predictions returns a list of prob for all labels
conf_mat = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print('Confusion matrix')
print(y_names)
print(conf_mat)