import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model once
model = load_model('model/revuu.keras', compile=False)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def load_model_and_predict(image_path):
    img_tensor = preprocess_image(image_path)
    density_map = model.predict(img_tensor)[0]
    count = np.sum(density_map)
    return count
