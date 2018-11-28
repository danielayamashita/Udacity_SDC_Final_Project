#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from light_classification.img_collector import IMGCollector
from keras.preprocessing import image as img_preprocessing
from scipy.spatial import KDTree
import tf
import cv2
import yaml

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import load_model

ENABLE_COLLECT_DATA = False
STATE_COUNT_THRESHOLD = 3
MAX_NUM_IMG = 1000
MODEL = 3# 0 = Daniela's Model | 1: Wen's Model | 2: Solverjf's model
class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        rospy.logwarn('tl_detector')
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

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
        self.img_collector = IMGCollector()
        self.listener = tf.TransformListener()
        
        #self.model = self.light_classifier.set_model()
        #self.model._make_predict_function()
        #self.model = self.light_classifier.create_model()
        
        
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.waypoints_2d = None
        self.waypoint_tree = None
        
        self.var = True
        self.count_numImg = 0
        self.fileNum = 0
        self.has_finished_collect_data = not (ENABLE_COLLECT_DATA)
        self.last_pose = None
        rospy.logwarn('TLDetector __init__')
        rospy.spin()
        

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        try:
            self.waypoints = waypoints
            if not self.waypoints_2d:
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
                self.waypoint_tree = KDTree(self.waypoints_2d)
        except:
            rospy.logwarn('waypoints_cb exception')

            
        

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        if self.camera_image == None:
            rospy.logwarn('IMAGE' )
            
            if  MODEL == 0 or MODEL == 1:
                model = self.light_classifier.create_model(MODEL)
                model.load_weights('light_classification/model_weights2.h5')
                self.model = model
            else:
                self.model = self.light_classifier.set_model()
                self.model._make_predict_function()
                #global graph
                #graph = tf.get_default_graph()
                
            
        #rospy.logwarn('image_cb')
        self.has_image = True
        self.camera_image = msg
        #rospy.logwarn('self.camera_image height: %s', msg.height)
        #rospy.logwarn('self.camera_image height: %s', msg.width)
        
        
        
        light_wp, state = self.process_traffic_lights()
        
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

    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx -1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect,pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state
        #if(not self.has_image):
        #    self.prev_light_loc = None
        #    return False

        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #rospy.logwarn('light.state: %f',light.state)
        #self.light_classifier.get_classification(cv_image)
        return light.state
        

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        #light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            farthest_idx = car_wp_idx + 200
            #car_position = self.get_closest_waypoint(self.pose.pose)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                #Get stop line waypoint index
                line = stop_line_positions[i]
                
                #rospy.logwarn('self.camera_image: %s', line)
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                #Find closest stop line waypoints index
                d = temp_wp_idx - car_wp_idx
                if d>=0 and d< diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
                    
            if closest_light:
                if not self.has_finished_collect_data :
                    if not(line_wp_idx == -1 or (line_wp_idx >= farthest_idx)):
                        #rospy.logwarn('light_wp: %f',light_wp)	
                        if not self.last_pose:
                            rospy.logwarn('self.last_pose: %s',self.last_pose)
                            self.last_pose = self.pose
                        if(abs(self.last_pose.pose.position.x - self.pose.pose.position.x)> 1. or abs(self.last_pose.pose.position.y - self.pose.pose.position.y) > 1.):

                            self.last_pose = self.pose
                            self.count_numImg = self.count_numImg  + 1
                            rospy.logwarn('NUM: %d/%d',self.count_numImg,MAX_NUM_IMG)
                            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                            self.img_collector.img_collector(cv_image, self.get_light_state(closest_light))

                            if self.count_numImg > MAX_NUM_IMG and self.var:
                                self.var = False
                                self.has_finished_collect_data = True
                                self.fileNum = self.fileNum+1
                    state = self.get_light_state(closest_light)                     
                else:
                    #state = self.get_light_state(closest_light)
                    
                    #cv_image = cv2.resize(cv_image,(600,800))
                    
                    if MODEL == 0 or MODEL == 1:
                        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                        cv_image = np.expand_dims(cv_image, axis=0)
                        state = self.light_classifier.get_classification(cv_image,self.model )
                    else:
                        if not(line_wp_idx == -1 or (line_wp_idx >= farthest_idx)):
                            if MODEL == 2:
                                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                                state =self.light_classifier.get_classification_2(cv_image,self.model)
                            else:
                                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
                                cv_image = cv2.resize(cv_image,(224,224))
                                # to tensors and normalize it
                                x = img_preprocessing.img_to_array(cv_image)
                                x = np.expand_dims(x, axis=0).astype('float32')/255
                
                                # get index of predicted signal sign for the image
                                state = np.argmax(self.model.predict(x))
                                if state != closest_light.state:
                                    rospy.logwarn("Light state = %d, predicted = %d", closest_light.state, state)
                                else:
                                    rospy.loginfo("Light state %d correctly predicted", closest_light.state)
                        else:
                            state = TrafficLight.GREEN
                     
                    
                    
                        
                return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN
    
    

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
