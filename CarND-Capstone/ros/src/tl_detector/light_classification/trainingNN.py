
import csv
import cv2  
import numpy as np
import os.path

from tl_classifier import TLClassifier

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


light_classifier = TLClassifier()

MODEL = 1# 0 = Daniela's Model | 1: Wen's Model
ENABLE_TRAINING = True
NumClasses = 3; #Red | Yellow | Green

model_name = 'model_weights7.h5'
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
    label = int(line[1])
    
    if label == 0: #Red
        label = [1., 0., 0.]
    if label == 1: #Yellow
        label = [0., 1., 0.]
    if label == 2: #Green
        label = [0., 0., 1.]
        
    labels.append(label)
    
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

print(y_train[0])

print('Shape of data y_train (AFTER  encode): %s\n' % str(y_train.shape))
print('Shape of data y_test (AFTER  encode): %s\n' % str(y_test.shape))
#Resume of data input
print('TOTAL images: ',str(total))
print('Image shape',str(image.shape))

#==============================================================================
#----------------- MODEL DEFINITION -------------------------------------------
#==============================================================================	
model = Sequential()
model = light_classifier.create_model(MODEL)


if ENABLE_TRAINING:
    model.compile(loss ='mse', optimizer = 'adam')

    model.fit(X_train, y_train,batch_size=30, shuffle=True, epochs = 10)
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


    model.save_weights(model_name)
