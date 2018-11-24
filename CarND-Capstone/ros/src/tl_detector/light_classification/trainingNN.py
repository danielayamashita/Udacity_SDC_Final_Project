
import csv
import cv2  
import numpy as np
import os.path

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras.utils import to_categorical
import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)
#Read labels
lines = []
with open('../tl_label.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Load images    
images = []
labels = []
for line in lines:
    image_path = '../Image/' + line[0]
    image = cv2.imread(image_path)
    images.append(image)
    labels.append(int(line[1]))
    
#==============================================================================
#----------------- DATA AUGMENTATION ------------------------------------------
#==============================================================================
augmented_labels = []
augmented_images = []
for image, label  in zip(images,labels):
    augmented_images.append(image)
    augmented_labels.append(label)
    augmented_images.append(cv2.flip(image,1))
    augmented_labels.append(label)

    
#Split the data into training and validating dat
total = len(augmented_labels)
n_train= np.floor(total*0.8).astype(int)

X_train, y_train = np.array(augmented_images[0:n_train]), np.array( augmented_labels[0:n_train])
X_test, y_test = np.array(augmented_images[n_train+1:-1]), np.array( augmented_labels[n_train+1:-1])

y_train = to_categorical(y_train) 
y_test = to_categorical( y_test) 
print('Shape of data y_train (AFTER  encode): %s\n' % str(y_train.shape))
print('Shape of data y_test (AFTER  encode): %s\n' % str(y_test.shape))
#Resume of data input
print('TOTAL images: ',str(total))
print('Image shape',str(image.shape))

#==============================================================================
#----------------- MODEL DEFINITION -------------------------------------------
#==============================================================================	
model = Sequential()

model.add(Lambda(lambda x:x/255-0.5,input_shape=(image.shape)))
model.add(Convolution2D(12, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (2, 2), activation="relu"))
model.add(Convolution2D(64, (2, 2), activation="relu"))
model.add(Convolution2D(64, (2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3, activation = 'softmax'))

"""

# 800,600,16
model.add(Convolution2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu',input_shape = image.shape))
# 400,300,16
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 400,300,32
model.add(Convolution2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
# 200,150,32
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 200,150,64
model.add(Convolution2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
# 100,75,64
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 100,75,128
model.add(Convolution2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
# 14,14,128
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 14,14,256
model.add(Convolution2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
# 7,7,256
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# flatten
model.add(Flatten())
model.add(Dense(3000, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
model.add(Dense(1000, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
model.add(Dense(500, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
# Fully connected Layer to the number of signal categories
model.add(Dense(3, activation = 'softmax'))
"""


model.compile(loss ='mse', optimizer = 'adam')

model.fit(X_train, y_train, shuffle=True, epochs = 10)
print("Finished")


print("Testing")
metrics = model.evaluate( x=X_test, y=y_test)

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    if metrics.size ==1:
        metric_value  = metrics
    else:
        metric_value = metrics[metric_i]    
    print('{}: {}'.format(metric_name, metric_value))


model.save_weights('model_weights6.h5')
