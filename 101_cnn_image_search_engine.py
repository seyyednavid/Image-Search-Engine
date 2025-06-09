
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



###########################################################################################
# preprocessing & featurising functions
###########################################################################################

# image pre-processing function

def preprocess_image(file_path):
    
    image = load_img(file_path, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)  
    
    return image


# featurise image

def featurise_image(image):
    
    feature_vector = model.predict(image)
    
    return  feature_vector



###########################################################################################
# featurise base images
###########################################################################################

# source directory for base images

source_dir = 'data/'

# empty objects to append to

filename_store = []

feature_vector_store = np.empty((0, 512))

# pass in & featurise base image set

for image in listdir(source_dir):
    
    # append image filename for future look up
    filename_store.append(source_dir + image)

    # preprocess the image
    preprocessed_image = preprocess_image(source_dir + image)
    
    # extraxt the feature vectore
    feature_vector = featurise_image(preprocessed_image)
    
    # append feature vector for similarity calculation
    feature_vector_store = np.append(feature_vector_store, feature_vector, axis = 0)

# save key objects for future use
pickle.dump(filename_store, open('models/filename_store.p', 'wb'))
pickle.dump(feature_vector_store, open('models/feature_vector_store.p', 'wb'))

















