from styx_msgs.msg import TrafficLight
#import keras
from keras.models import load_model
from keras.applications import mobilenet
from keras.preprocessing import image #contains PIL.Image as pil_image
# mobilenet contains some custom layers which need to get imported while the model is reload
from keras.utils.generic_utils import CustomObjectScope
#import PIL
import cv2 # requires to pip install opencv-python (if run on CPU in workspace)
import numpy as np

import tensorflow as tf

MODEL_SIM = 'light_classification/my_model_1.h5' # Only exactly this path works with roslaunch for some motive i don't understand

def prepare_image(img_arr):#file):
    """ works in this version but has to work without installing opencv for submission """
    #img = image.load_img(file, target_size=(224, 224)) # PIL image
    #img = image.pil_image.fromarray(img_arr)
    #img.resize((224, 224))
    #img_array = image.img_to_array(img) # Error "Cannot handle this data type"
    img_array = cv2.resize(img_arr,(224,224))
    img_array_expanded_dims = np.expand_dims(img_array, axis=0).astype(np.float32) # float conversion needed for true_divide in preprocess_input
    img_processed = mobilenet.preprocess_input(img_array_expanded_dims) # scales pixel values etc
    return img_processed

class TLClassifier(object):
    def __init__(self):
        self.set_model()
        global graph # fixes a keras problem with reloading the model https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
        graph = tf.get_default_graph()

    def set_model(self):
        with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
            self.model = load_model(MODEL_SIM)  
            
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_processed = prepare_image(image)
        with graph.as_default():
            prediction = self.model.predict(image_processed)
        signal = prediction[0].argmax()
        #TODO implement light color prediction
        if signal==0:
            return TrafficLight.GREEN
        elif signal==1:
            return TrafficLight.RED
        elif signal==2:
            return TrafficLight.UNKNOWN
        elif signal==3:
            return TrafficLight.YELLOW
        
'''
testimage = '/home/workspace/CarND-Capstone/data/TLdataset02/red/out00180.png'
import matplotlib.pyplot as plt
img_arr = plt.imread(testimage)
 
classifier = TLClassifier()
classifier.get_classification(img_arr)
'''