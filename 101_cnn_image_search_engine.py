
#########################################################################
# Convolutional Neural Network - Image Search Engine
#########################################################################


###########################################################################################
# import packages
###########################################################################################

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle


###########################################################################################
# bring in pre-trained model (excluding top)
###########################################################################################

# image parameters

img_width = 224
img_height = 224
num_channels = 3


# network architecture

vgg = VGG16(input_shape=(img_width, img_height, num_channels), include_top = False, pooling = 'avg')


model = Model(inputs = vgg.input, outputs = vgg.layers[-1].output)

# save model file

model.save("models/vgg16_search_engine.h5")