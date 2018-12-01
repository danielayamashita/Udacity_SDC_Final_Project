import time
import torch
import numpy as np
from keras.preprocessing import image as img_preprocessing
import cv2
from collections import Counter

def boxes_iou(box1, box2):
  
    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx
    
    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Calculate the height of the union of the two bounding boxes
    union_height = My - my
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
   
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area/union_area
    
    return iou


def nms(boxes, iou_thresh):
    
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes
    
    # Create a PyTorch Tensor to keep track of the detection confidence
    # of each predicted bounding box
    det_confs = torch.zeros(len(boxes))
    
    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    _,sortIds = torch.sort(det_confs, descending = True)
    
    # Create an empty list to hold the best bounding boxes after
    # Non-Maximal Suppression (NMS) is performed
    best_boxes = []
    
    # Perform Non-Maximal Suppression 
    for i in range(len(boxes)):
        
        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sortIds[i]]
        
        # Check that the detection confidence is not zero
        if box_i[4] > 0:
            
            # Save the bounding box 
            best_boxes.append(box_i)
            
            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i. 
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                
                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero. 
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
                    
    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh):
    
    # Start the time. This is done to calculate how long the detection takes.
    start = time.time()
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Convert the image from a NumPy ndarray to a PyTorch Tensor of the correct shape.
    # The image is transposed, then converted to a FloatTensor of dtype float32, then
    # Normalized to values between 0 and 1, and finally unsqueezed to have the correct
    # shape of 1 x 3 x 416 x 416
    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    
    # Feed the image to the neural network with the corresponding NMS threshold.
    # The first step in NMS is to remove all bounding boxes that have a very low
    # probability of detection. All predicted bounding boxes with a value less than
    # the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)
    
    # Make a new list with all the bounding boxes returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    
    # Perform the second step of NMS on the bounding boxes returned by the neural network.
    # In this step, we only keep the best bounding boxes by eliminating all the bounding boxes
    # whose IOU value is higher than the given IOU threshold
    boxes = nms(boxes, iou_thresh)
    
    # Stop the time. 
    finish = time.time()
    
    # Print the time it took to detect objects
#     print('\n\nIt took {:.3f}'.format(finish - start), 'seconds to detect the objects in the image.\n')
    
#     # Print the number of objects detected
#     print('Number of Objects Detected:', len(boxes), '\n')
    
    return boxes


def load_class_names(namesfile):
    
    # Create an empty list to hold the object classes
    class_names = []
    
    # Open the file containing the COCO object classes in read-only mode
    with open(namesfile, 'r') as fp:
        
        # The coco.names file contains only one object class per line.
        # Read the file line by line and save all the lines in a list.
        lines = fp.readlines()
    
    # Get the object class names
    for line in lines:
        
        # Make a copy of each line with any trailing whitespace removed
        line = line.rstrip()
        
        # Save the object class name into class_names
        class_names.append(line)
        
    return class_names


def print_objects(boxes, class_names):    
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))

def get_bounding_boxes (img, m, iou_thresh = 0.4, nms_thresh = 0.6):
    # We resize the image to the input width and height of the first layer of the network.    
    resized_image = cv2.resize(img, (m.width, m.height))
    # Detect objects in the image
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    return boxes
            
def signal_prediction(img, boxes, class_names, model):
    signals = []
    # Predict the light state in each bounding box
    for box in boxes:    
        # get the id of the object in the bounding box
        cls_id = box[6]
        # process bounding box on top of the image if it's a traffic light
        # casue YOLO will get not only traffic lights
        if (class_names[cls_id] == 'traffic light'):
            # Get the width and height of the image
            width = img.shape[1]
            height = img.shape[0]
            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
            # of the bounding box relative to the size of the image. 
            x1 = int(np.around((box[0] - box[2]/2.0) * width))
            y1 = int(np.around((box[1] - box[3]/2.0) * height))
            x2 = int(np.around((box[0] + box[2]/2.0) * width))
            y2 = int(np.around((box[1] + box[3]/2.0) * height))
            # **********************************************************
            #              For the Traffic Light Classifier
            # **********************************************************            
            # if the box is out of image, then skip the box
            if (x1<0 or y1 <0 or x2 > img.shape[1] or y2 > img.shape[0]):
                continue
            # cropped the image only contains traffic light
            cropped_image = img[y1:y2,x1:x2,:]
            cropped_image = cv2.resize(cropped_image,(224,224))
            # to tensors and normalize it
            x = img_preprocessing.img_to_array(cropped_image)
            x = np.expand_dims(x, axis=0).astype('float32')
            # get index of predicted signal sign for the image and store it
            signal_prediction = np.argmax(model.predict(x))
            signals.append(signal_prediction)
    if (len(signals) == 0):
        return 2
    else:
        c = Counter(signals)
        value, count = c.most_common()[0]
        return value
    
            
