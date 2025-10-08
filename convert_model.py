from tensorflow.keras.models import load_model

# Load your old model
model = load_model("models/vgg16_search_engine.h5", compile=False)

# Save it again in a clean format
model.save("models/vgg16_search_engine_compatible.h5", save_format="h5")
