import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time
import os


# open('static/b'Abyssinian_1.jpg'.jpg')

def preprocess_image(model, image_path):
    time.sleep(1)
    input_shape = model.input_shape
    target_size = input_shape[1:3]
    image = load_img(image_path, target_size=target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def main(img_path):
    model = load_model('model_cnn.h5', compile=False)
    predictions = model.predict(preprocess_image(model, img_path))
    predicted_label = np.argmax(predictions, axis=1)
    class_names = ['airplane ✈️', 'automobile 🚗', 'bird 🐦', 'cat 🐱', 'deer 🦌', 'dog 🐶', 'frog 🐸', 'horse 🐴', 'ship 🚢',
                   'truck 🚚']
    return class_names[predicted_label[0]]
