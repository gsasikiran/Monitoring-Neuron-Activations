import numpy as np
from PIL import Image
from keras.models import Model

from src.post_processing.visualization import DisplayActivations
from src.processing.model_architecture import ModelArchitecture
from src.processing.tools import Tools

# Converting image to array
img_path = input("Enter path of the image")
img = Image.open(img_path).convert('L')
img = np.array(img)

# Model architecture
model_architecture = ModelArchitecture()
model_cnn = model_architecture.CNN()

# Getting convolutional activation layers from the model
tools = Tools()
conv_layer_list = tools.get_conv_layers(model_cnn)
outputs = [model_cnn.layers[i].output for i in conv_layer_list]
model = Model(inputs=model_cnn.inputs, outputs=outputs)

# predicting feature maps/activation patterns
feature_maps = model.predict(img.reshape(1, 28, 28, 1))
layer_num = int(input('Input layer number to be plotted.. (If NONE plots all the layer activations)'))

# Visualizing the activation patterns
display_activation = DisplayActivations()
display_activation.plot_activation(feature_maps, layer_num)
