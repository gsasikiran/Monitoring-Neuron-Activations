from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np


def image_preprocess(img_path):

    img = load_img(img_path, target_size=(224,224,3))
    plt.imshow(img)
    img_array = img_to_array(img)
    img_exp = np.expand_dims(img_array, axis=0)
    img = preprocess_input(img_exp)