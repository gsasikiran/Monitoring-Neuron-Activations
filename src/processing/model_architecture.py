from keras.applications.vgg16 import VGG16

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout

from keras.models import Model


class ModelArchitecture:
    def __init__(self):
        pass

    @staticmethod
    def VGG16():
        """
         Creates a VGG16 model
         VGG16 model - 'https://arxiv.org/pdf/1409.1556.pdf'
        :return: Model's instance
        """
        return VGG16()

    @staticmethod
    def CNN(input_shape=(28, 28, 1)):
        """
         Return CNN architecture
        :param input_shape: tuple
            The input shape of the image (width, height, channels)
        :return: Model's instance
        """

        input = Input(input_shape)

        x = Conv2D(64, (3, 3), strides=(1, 1), name='layer_conv1', padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), name='maxPool1')(x)

        x = Conv2D(64, (3, 3), strides=(1, 1), name='layer_conv2', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), name='maxPool2')(x)

        x = Conv2D(32, (3, 3), strides=(1, 1), name='conv3', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), name='maxPool3')(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu', name='fc0')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='relu', name='fc1')(x)
        x = Dropout(0.25)(x)
        x = Dense(10, activation='softmax', name='fc2')(x)

        model = Model(inputs=input, outputs=x, name='Predict')

        return model
