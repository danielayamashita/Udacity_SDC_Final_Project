from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load the train, test datasets
def load_dataset(path):
    data = load_files(path)
    X = np.array(data['filenames'])
    y = np_utils.to_categorical(np.array(data['target']), 4)
    return X, y

# load the train, test dataset
X, y = load_dataset('TLdataset02')
# load the list of signal names
signal_names = [item[12:-1] for item in sorted(glob("TLdataset02/*/"))]

print('There are %d total images' % X.shape[0])
print('There are %d kinds of signals:' % y.shape[1])
print('\t\t\t',signal_names)

# split the data into training/validation/testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.1, random_state =42)

# transform the input data to tensors
from keras.preprocessing import image
from tqdm import tqdm
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
# to tensors and normalize it
train_tensors = paths_to_tensor(X_train).astype('float32')/255
valid_tensors = paths_to_tensor(X_val).astype('float32')/255
test_tensors = paths_to_tensor(X_test).astype('float32')/255

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint  
from keras import applications
from keras import optimizers
from keras.models import load_model

model = Sequential()
### TODO: Define your architecture.
# 224,224,16
model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (224, 224, 3)))
# 112,112,16
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 112,112,32
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
# 56,56,32
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 56,56,64
model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
# 28,28,64
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 28,28,128
model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
# 14,14,128
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 14,14,256
model.add(Conv2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
# 7,7,256
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# flatten
model.add(Flatten())
model.add(Dense(5000, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
model.add(Dense(1000, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
model.add(Dense(500, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
# Fully connected Layer to the number of signal categories
model.add(Dense(4, activation = 'softmax'))

model.summary()

# compile the model
model.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

# train the model.
epochs = 10
batch_size = 64

checkpointer = ModelCheckpoint(filepath='saved_models/weights.test.self_defined.h5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, y_train, 
          validation_data=(valid_tensors, y_val),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

# load the trained model
# model.load_weights('saved_models/weights.best.self_defined.hdf5')

# # get index of predicted signal sign for each image in test set
# signal_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# # print out test accuracy
# test_accuracy = 100*np.sum(np.array(signal_predictions)==np.argmax(y_test, axis=1))/len(signal_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)

del model
model = load_model('saved_models/weights.test.self_defined.h5')
signal_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# print out test accuracy
test_accuracy = 100*np.sum(np.array(signal_predictions)==np.argmax(y_test, axis=1))/len(signal_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
