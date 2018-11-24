import rospy
from styx_msgs.msg import TrafficLight
import cv2
import tf
import numpy as np
from sensor_msgs.msg import Image

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import load_model


    

class TLClassifier(object):
    def __init__(self):
        #self.model.load_weights('light_classification/model2.h5')
        pass
    def create_model(self):
        # load the model
        model = Sequential()
        model.add(Lambda(lambda x:x/255-0.5,input_shape=(600,800,3)))
        #model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (image.shape)))
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
        model.add(Dense(3))
        return model
    def get_classification(self, image,model):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        #height, width = image.shape[:2]
        #rospy.logwarn('PREDICT MODEL')
        signal = model.predict(image)
        
        
        signal_out = np.argmax(signal)
        rospy.logwarn('get_classification: %f',signal_out )
        #rospy.logwarn('get_classification: %f',signal )
        #rospy.logwarn('get_classification: %f',width )
        return signal_out
