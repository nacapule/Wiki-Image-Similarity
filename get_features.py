import cv2
import image_feature_tensorflow as tfeature
from scipy import spatial
import numpy as np

'''
What are the dominant colors of an image?
Just count the number of pixels in the hue space belonging to each color.
'''
def extract_color_distribution(img):
    hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert irgbto hsv
    h,s,v = cv2.split(hsv_img)
    #creating a dictionary of 12 colors sampled from the hue space
    color_names=['Red','Orange','Yellow','Yellow-Green','Green','Aqua',
                 'Cyan','Azure','Blue','Violet','Magenta,','Rose']
    #quantize each pixel from a value in the range (180) to a value between 0 and 11
    h_quant=np.floor(np.divide(h,15))
    #compute distribution over these 12 values
    color_values=np.histogram(h_quant,12)[0]
    color_values=color_values/float(h.shape[0]*h.shape[1])
    #assign a label to each bin of the color distribution
    color_dict={color_names[i]:color_values[i] for i in range(len(color_values))}
    return color_dict
'''
What are the dominant objects of an image?
use Imagenet classifier to get probabilities of 1000 objects
'''
def extract_objects(img):
    return tfeature.classify(img)

'''
How distant are 2 images? Use cosine distance to infer similarity between 2 images
'''
def compare_images(dic1,dic2):
    '''
    dic1,dic2: dictionaries with each element being a feature vector from a n image, indexed with the feature name
    '''
    return spatial.distance.cosine(list(dic1.values()),list(dic2.values()))
