import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import NearestNeighbors



# Paths
MODEL_PATH = "models/vgg16_search_engine_compatible.h5"
FILENAME_STORE_PATH = "models/filename_store.p"
FEATURE_STORE_PATH = "models/feature_vector_store.p"

IMG_WIDTH, IMG_HEIGHT = 224, 224


# -----------------------------
# Load everything once at startup
# -----------------------------
model = load_model(MODEL_PATH, compile=False)
filename_store = pickle.load(open(FILENAME_STORE_PATH, "rb"))
feature_vector_store = pickle.load(open(FEATURE_STORE_PATH, "rb"))

# Fit nearest neighbors once
nn = NearestNeighbors(n_neighbors=8, metric="cosine")
nn.fit(feature_vector_store)


# -----------------------------
# Helpers
# -----------------------------
def preprocess_image(path):
    img = load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def featurise_image(path):
    pre = preprocess_image(path)
    vec = model.predict(pre)
    return vec.flatten().reshape(1, -1)


def search_image(query_path, top_k=8):
    query_vec = featurise_image(query_path)
    distances, indices = nn.kneighbors(query_vec)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({"filename": filename_store[idx], "distance": float(dist)})
    return results
