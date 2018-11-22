from styx_msgs.msg import TrafficLight
from keras.models import load_model
from keras.applications import mobilenet
from keras.preprocessing import image

MODEL_SIM = 'my_model.h5'

def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224)) # PIL image
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_processed = mobilenet.preprocess_input(img_array_expanded_dims) # scales pixel values etc
    return img_processed

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #
        self.model = load_model(MODEL_SIM)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
