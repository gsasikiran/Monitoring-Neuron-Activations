import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from PIL import Image
import glob

from src.pre_processing.img_preprocess import image_preprocess, image_preprocess_tri
from src.processing.model_architecture import ModelArchitecture
from src.processing.tools import Tools
from src.post_processing.visualization import display_activation

img = image_preprocess_tri("/home/gsasikiran/Desktop/Semester_3/SDP/Monitoring-Neuron-Activations/triangles/drawing(1).png")

model_architecture = ModelArchitecture()
model_cnn = model_architecture.CNN()

tools = Tools()
conv_layer_list = tools.get_conv_layers(model_cnn)

outputs = [model_cnn.layers[i].output for i in conv_layer_list]
model = Model(inputs=model_cnn.inputs, outputs=outputs)

feature_maps = model.predict(img.reshape(1,28,28,1))

display_activation(feature_maps,8,4,2)

# row = 8
# col = 4
# for fmap in feature_maps:
#
#     ix = 1
#     for _ in range(row):
#         for _ in range(col):
#             ax = plt.subplot(row, col, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#             plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
#
#             ix += 1
#
#     plt.show()
