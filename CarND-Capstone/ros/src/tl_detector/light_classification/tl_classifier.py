import rospy
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

import numpy as np
from keras.models import Model
from keras import applications
from keras.models import load_model
from keras.preprocessing import image as img_preprocessing
import cv2

# load the trained model
from keras.utils.generic_utils import CustomObjectScope

model_filepath = 'saved_models/model.MobileNet-3-classes.h5'
n_classes = 3


class TLClassifier(object):
    def __init__(self):
        # load keras libraies and load the MobileNet model
        self.model_loaded = False

    def load_model(self):
        rospy.loginfo("TLClassifier: Loading model...")
        with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model(model_filepath)
            self.model._make_predict_function() # Otherwise there is a "Tensor %s is not an element of this grap..." when predicting
        self.model_loaded = True
        rospy.loginfo("TLClassifier: Model loaded - READY")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light  

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.model_loaded:
            rospy.logwarn("Model not loaded yet, clssification not possible!")
            return TrafficLight.UNKNOWN
        
        # The model was trained with RGB images.
        # So the image needs to be provided as RGB:
        # self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        # Otherwise a conversion would be necessary
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # The model expects RGB images in (224, 224) as input
        image = cv2.resize(image,(224,224))
        
        # to tensors and normalize it
        x = img_preprocessing.img_to_array(image)
        x = np.expand_dims(x, axis=0).astype('float32')/255
        
        # get index of predicted signal sign for the image
        signal_prediction = np.argmax(self.model.predict(x))

        return signal_prediction

