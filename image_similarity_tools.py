from matplotlib import pyplot as plt

import cv2
import numpy as np
import os
import tensorflow as tf
import time



import tensorflow_hub as hub

'''
image classifier
'''

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

image_shape = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(mobilenet_v2, input_shape=image_shape+(3,), output_shape=[1001])
])

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

def classify_image(fn):
    """
    Classifies a image using mobilenetv2, and returns an list of
    """
    im = cv2.imread(fn)
    im = cv2.resize(im, image_shape, interpolation = cv2.INTER_AREA)
    im = im/255.0
    preds = classifier.predict(im[np.newaxis, ...])
    preds = tf.math.softmax(preds[0], axis=-1)
    return [(imagenet_labels[i], pred.numpy()) for i,pred in enumerate(preds)]
#     top_k = tf.nn.top_k(dist, k=5)
#     return [(imagenet_labels[i], dist[i].numpy()) for i in top_k.indices]

def timed_classify_image(fn):
    """
    Classifies a image using mobilenetv2, and returns an list of
    """
    start = time.time()
    
    im = cv2.imread(fn)
    imrt = time.time()
    print(f'image read: {imrt - start : .3f} s')
    
    im = cv2.resize(im, image_shape, interpolation = cv2.INTER_AREA)
    imrst = time.time()
    print(f'image resize: {imrst - imrt : .3f} s') 
    
    im = im/255.0
    pxnr = time.time()
    print(f'pixel normalization: {pxnr - imrst : .3f} s')
    
    preds = classifier.predict(im[np.newaxis, ...])
    prdc = time.time()
    print(f'prediction calculation: {prdc - pxnr : .3f} s')

    preds = tf.math.softmax(preds[0], axis=-1)
    smap = time.time()
    print(f'softmax application: {smap - prdc : .3f} s')
    
    return [(imagenet_labels[i], pred.numpy()) for i,pred in enumerate(preds)]
#     top_k = tf.nn.top_k(dist, k=5)
#     return [(imagenet_labels[i], dist[i].numpy()) for i in top_k.indices]

'''
helper method to plot images
'''

def plot(image_fn, grid=None):
    """
    Display images by file path.
    `image_fn`: str or list of strings
    `grid`: optional size of the grid as a tuple (rows, columns)
    """
    if not grid or isinstance(image_fn, str):
        # display a single image
        if isinstance(image_fn, list):
            image_fn = image_fn[0]
        img = cv2.imread(image_fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #transforming to rgb for visualization
        fig = plt.gcf()
        fig.set_size_inches(10.5, 10.5)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
    else:
        # display multiple images
        nrows, ncols = grid
        _, axs = plt.subplots(nrows, ncols, figsize=(18, 9))
        axs = axs.flatten()
        [ax.axis('off') for ax in axs]
        for fn, ax in zip(image_fn, axs):
            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #transforming to rgb for visualization
            ax.imshow(img)

        plt.show()

'''
puts an artificial outlier into the dataset
'''

foxes = 'data/Fox/foxes.jpg'
def run_me():
    """
    Outlier removal (opposite)
    """
    fox = 'data/Fox/File141s_Lisek_na_Kasprowym_Wierchu.jpg'
    image = cv2.imread(fox)
    cv2.imwrite(foxes, np.concatenate([image]*100))

def delete_foxes():
    os.remove(foxes)

