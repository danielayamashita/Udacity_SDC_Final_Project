from styx_msgs.msg import TrafficLight
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

# build the structure of the model
def create_model():
    model = Sequential()
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
    return model

class TLClassifier(object):
    def __init__(self):
        # load the model
        self.model = create_model()
        self.model.load_weights('saved_models/weights.best.self_defined.h5')

    def img_to_tensor(self, img):
        # resize the image to (224, 224) to input into the model
        img = cv2.resize(img,(224,224))
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(img, axis=0)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light  

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # signal should be a vector of size four [green, red, unknown, yellow]. One of the value is 1 others are 0
        # ex. [0 1 0 0], which indicates it's a red light
        signal = self.model.predict(img_to_tensor(image))
        if (signal[0] == 1):
            return TrafficLight.GREEN
        elif (signal[1] == 1):
            return TrafficLight.RED
        elif (signal[3] == 1):
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
