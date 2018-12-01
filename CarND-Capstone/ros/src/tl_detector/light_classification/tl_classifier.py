import rospy
from styx_msgs.msg import TrafficLight

import numpy as np
from keras.models import Model
from keras import applications
from keras.models import load_model
from keras.preprocessing import image as img_preprocessing
import cv2

# for the YOLO
from utils import *
from darknet import Darknet

# Set the location and name of the cfg file
cfg_file = 'cfg/yolov3.cfg'

# Set the location and name of the pre-trained weights file
weight_file = 'weights/yolov3.weights'

# Set the location and name of the COCO object classes file
namesfile = 'data/coco.names'

# load the trained model
from keras.utils.generic_utils import CustomObjectScope

model_filepath = 'saved_models/tl_classifier_Mobile_3.h5'
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
        # Load the YOLO network architecture
        self.m = Darknet(cfg_file)
        # Load the pre-trained weights
        self.m.load_weights(weight_file)
        # Load the COCO object classes
        self.class_names = load_class_names(namesfile)
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
        boxes = get_bounding_boxes(image, self.m)
        light_state = signal_prediction(image, boxes, self.class_names, self.model)
        return light_state
