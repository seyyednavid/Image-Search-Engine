
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
