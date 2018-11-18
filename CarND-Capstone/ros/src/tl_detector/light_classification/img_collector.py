import pickle
import numpy as np
import rospy
from PIL import Image
import cv2
class IMGCollector(object):
    def __init__(self):
        #Create pickle file
        self.pickle_file = open("training_data.pickle","wb")
        #self.pickle_file2 = open("d.pickle","wb")
        self.images = []
        self.labels = []
        self.dict = None
        rospy.logwarn('IMGCollector __init__')

    
    def img_collector(self,image,label):
        self.images.append(image)
        self.labels.append(label)
        #dict2 = {"image":image,"label": label}
        #pickle.dump(dict2, self.pickle_file)
        rospy.logwarn('img_collector')
        rospy.logwarn('label: %f',label)
        #rospy.logwarn('image: %f',image)
    
    def save_pickle_file(self):
        
        self.dict = {"image":self.images,"label": self.labels}
        pickle.dump(self.dict, self.pickle_file)
        self.pickle_file.close()
        rospy.logwarn('save_pickle_file')
        
    def read_pickle_file(self,name):
        pickle_in = open(name + ".pickle","rb")
        pickle_dict = pickle.load(pickle_in)
        rospy.logwarn('label pickle read: %s',pickle_dict["label"])
        image = pickle_dict["image"]
        #rospy.logwarn('image shape: %f',image[0].shape)
        
        #im = Image.fromarray(image[0])
        #im.save("your_file.jpeg")
        cv2.imwrite("your_file.jpeg",np.array(image[0]))
        #rospy.logwarn('label pickle read: %s',image[0])
        
        