import rospy
from styx_msgs.msg import TrafficLight

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
        #TODO load classifier
        # load keras libraies and load the MobileNet model
        with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model(model_filepath)
            self.model._make_predict_function() # Otherwise there is a "Tensor %s is not an element of this grap..." when predicting
        rospy.logwarn("TLClassifier: Model loaded - READY")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # transform the input data to tensors

        # The model was trained with RGB images.
        # Is image delivered as BGR or RGB?
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image,(224,224))
        
        # to tensors and normalize it
        x = img_preprocessing.img_to_array(image)
        x = np.expand_dims(x, axis=0).astype('float32')/255
        
        # get index of predicted signal sign for the image
        signal_prediction = np.argmax(self.model.predict(x))

        return signal_prediction
        #return TrafficLight.UNKNOWN
