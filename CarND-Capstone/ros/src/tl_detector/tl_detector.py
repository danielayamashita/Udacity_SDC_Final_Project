#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

# Only process every Xth image
IMAGE_CB_THRESHOLD = 4

# for debugging and classification
import inspect
import time
from PIL import Image as PILImage
import os
# for collecting classification images, must be a multiple of IMAGE_CB_THRESHOLD 
IMAGE_COUNTER = 3 * IMAGE_CB_THRESHOLD
IMAGE_FOLDER = "./images"

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose           = None
        self.waypoints      = None
        self.camera_image   = None
        self.waypoints_tree = None
        self.lights         = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.image_cb_counter = 0
        
        # for image classification collection
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        
        # TODO: waypoints_2d necessary?
        if not self.waypoints_tree:
            waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(waypoints_2d)

    def traffic_cb(self, msg):
        #rospy.logwarn("Method: %s", inspect.stack()[0][3])
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.image_cb_counter += 1
        if(0 != (self.image_cb_counter % IMAGE_CB_THRESHOLD)):
            return
        
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        # For classification images generation
        if(0 == (self.image_cb_counter % IMAGE_COUNTER)):
            # Just for debugging purposes show the image details
            #rospy.logwarn("Image encoding: %s", msg.encoding)
            #rospy.logwarn("Image HxW: %d x %d", msg.height, msg.width)
            #rospy.logwarn("Image big endian, step: %d, %d", msg.is_bigendian, msg.step)
            #rospy.logwarn("Image data len: %d", len(msg.data))
            # For debugging, show image state
            #rospy.logwarn("Image = %d, light state = %d", self.image_cb_counter % IMAGE_CB_THRESHOLD, state)
    
            car_wp_idx = self.get_closest_waypoint( self.pose.pose.position.x, self.pose.pose.position.y)
            d = light_wp - car_wp_idx 
            # A test image shall be saved whenever a traffic light is visible 
            saveImage = (d >= 0 and d < 150)
            driveState = state
            if (not saveImage) and (0 == (self.image_cb_counter % (10*IMAGE_COUNTER))):
                # No traffic light visible is handled the same way as a green light
                # we need to store some images showing no traffic light so that the car can also be
                # trained for these parts of the road
                # saveImage = True
                driveState = TrafficLight.GREEN
            # TODO: A propoer distance needs to be checked
            if saveImage:
                # Save image and image classification
                rospy.logwarn("Saving image and classification. Index distance = %d", d)
                image = PILImage.frombytes('RGB', (msg.width,msg.height), msg.data)
                filename = "{]/Image_{}.jpg".format(IMAGE_FOLDER, self.image_cb_counter/IMAGE_CB_THRESHOLD)
                image.save(filename)
                with open("./test.txt", "a") as myfile:
                    myfile.write("{},{}\n".format(filename, driveState))
            else:
                rospy.logwarn("No traffic light in visible distance. Index distance = %d", d)
        '''

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # For testing, just return the light state
       # return light.state
        
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        light_state = self.light_classifier.get_classification(cv_image)
        
        #Just for testing the classifier in the simulator
        if light_state != light.state:
            rospy.logwarn("Light state = %d, predicted = %d", light.state, light_state)
        else:
            rospy.loginfo("Light state %d correctly predicted", light.state)
        
        return light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        light_wp_idx = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint( self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = 200 #len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Gets stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Finds closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    light_wp_idx = temp_wp_idx
                     
        if closest_light:
            state = self.get_light_state(closest_light)
            return light_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
