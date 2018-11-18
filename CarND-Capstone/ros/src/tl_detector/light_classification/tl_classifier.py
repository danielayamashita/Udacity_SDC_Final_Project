import rospy
from styx_msgs.msg import TrafficLight
import cv2
import tf
import numpy as np
from sensor_msgs.msg import Image

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        height, width = image.shape[:2]
        #rospy.logwarn('get_classification: %f',height )
        #rospy.logwarn('get_classification: %f',width )
        return TrafficLight.UNKNOWN
