from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def image_preprocess(img_path, input_shape):

    img = load_img(img_path, target_size=input_shape)
    img_array = img_to_array(img)
    img_exp = np.expand_dims(img_array, axis=0)
    img = preprocess_input(img_exp)
    return img

def image_preprocess_tri(img_path):

    img = Image.open(img_path).convert('L')
    img_array = img_to_array(img)
    # # img_exp = np.expand_dims(img_array, axis=0)
    # img = preprocess_input(img_array)
    return img_array