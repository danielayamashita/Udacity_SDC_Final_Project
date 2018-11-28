import rospy
from styx_msgs.msg import TrafficLight
import cv2
import tf
import numpy as np
from sensor_msgs.msg import Image

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D,Dropout
from keras.applications import mobilenet
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

MODEL_SIM = 'model.MobileNet-3-classes.h5' # Only exactly this path works with roslaunch for some motive i don't understand
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
        #self.model = self.set_model()
        pass
        
        #global graph # fixes a keras problem with reloading the model https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
        #graph = tf.get_default_graph()
        
    def set_model(self):
        with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
            model = load_model(MODEL_SIM)
            print(MODEL_SIM)
        return model
    def create_model(self,MODEL):
        #MODEL = 1;# 0 = Daniela's Model | 1: Wen's Model
        # load the model
        model = Sequential()
        if MODEL == 0:
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
        elif MODEL == 1:
            # 800,600,16
            model.add(Convolution2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu',input_shape = (600,800,3)))
            # 400,300,16
            model.add(MaxPooling2D(pool_size = 2))
            # Dropout
            model.add(Dropout(0.5))
            # 400,300,32
            
            model.add(Convolution2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
            # 200,150,32
            model.add(MaxPooling2D(pool_size = 2))
            # Dropout
            model.add(Dropout(0.5))
            # 200,150,64
            model.add(Convolution2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
            # 100,75,64
            model.add(MaxPooling2D(pool_size = 2))
            # Dropout
            model.add(Dropout(0.5))
            # 100,75,128
            model.add(Convolution2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
            # 14,14,128
            model.add(MaxPooling2D(pool_size = 2))
            # Dropout
            model.add(Dropout(0.5))
            # 14,14,256
            model.add(Convolution2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
            # 7,7,256
            model.add(MaxPooling2D(pool_size = 2))
            # Dropout
            model.add(Dropout(0.5))
            # flatten

            model.add(Flatten())
            model.add(Dense(1000, activation = 'relu'))

            # Dropout
            model.add(Dropout(0.5))
            model.add(Dense(900, activation = 'relu'))
            # Dropout
            model.add(Dropout(0.5))
            model.add(Dense(500, activation = 'relu'))
            # Dropout
            model.add(Dropout(0.5))
            # Fully connected Layer to the number of signal categories
            model.add(Dense(3, activation = 'softmax'))

            
        return model
    def get_classification(self, image,model):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        signal = model.predict(image)
        
        
        signal_out = np.argmax(signal)
        rospy.logwarn('get_classification: %f',signal_out )
        #rospy.logwarn('get_classification: %f',signal )
        #rospy.logwarn('get_classification: %f',width )
        return signal_out
        
    def get_classification_2(self, image,model):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_processed = prepare_image(image)
        
        prediction = model.predict(image_processed)
        signal = prediction[0].argmax()
        signal_out = signal
        
        #TODO implement light color prediction
        if signal==0:
            signal_out=TrafficLight.GREEN
        elif signal==1:
            signal_out= TrafficLight.RED
        elif signal==2:
            signal_out=TrafficLight.UNKNOWN
        elif signal==3:
            signal_out=TrafficLight.YELLOW
          
        rospy.logwarn('get_classification2: %f',signal_out )    
        return signal_out