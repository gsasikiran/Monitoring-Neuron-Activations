import numpy as np
from PIL import Image
from keras.models import Model
import argparse

from src.post_processing.visualization import DisplayActivations
from src.processing.model_architecture import ModelArchitecture
from src.processing.tools import Tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Give command as follows "python image_path layer_number"')
    parser.add_argument('image_path', type=str)

    parser.add_argument('layer_number', type=int)
    args = parser.parse_args()

    # Converting image to array
    img_path = args.image_path
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
    layer_num = args.layer_number

    # Visualizing the activation patterns
    display_activation = DisplayActivations()
    display_activation.plot_activation(feature_maps, layer_num)
