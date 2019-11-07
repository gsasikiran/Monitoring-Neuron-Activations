import matplotlib.pyplot as plt
from keras.models import Model

from src.pre_processing.img_preprocess import image_preprocess
from src.processing.model_architecture import ModelArchitecture
from src.processing.tools import Tools

model_architecture = ModelArchitecture()
model_vgg = model_architecture.VGG16()

tools = Tools()
conv_layer_list = tools.get_conv_layers(model_vgg)

outputs = [model_vgg.layers[i].output for i in conv_layer_list]

model = Model(inputs=model_vgg.inputs, outputs=outputs)

img = image_preprocess('cv.png')
feature_maps = model.predict(img)

square = 8

for fmap in feature_maps:
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
            ix += 1

    plt.show()
