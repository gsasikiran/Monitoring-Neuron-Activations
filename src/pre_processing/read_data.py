import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self, path_train="../../MNIST_data/train.csv",
                 path_test='../../MNIST_data/test.csv'):
        self.path_train = path_train
        self.path_test = path_test

    def get_data(self, split=False, valid_size=0.2):
        train_data = pd.read_csv(self.path_train)

        x = train_data.drop(labels=['label'], axis=1)
        y = train_data['label']

        if split:
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, random_state=42)
            return x_train, x_valid, y_train, y_valid

        return x, y

    @staticmethod
    def __get_image_array(series, shape):
        x = series.to_numpy()
        x = np.expand_dims(x, axis=1)
        x = x.reshape(shape)
        return x

    def df_to_images(self, df, img_size=(28, 28)):
        """Convert the dataframe of pixels into images

        param:
        df: Pandas data frame
        img_size: tuple
            Required size of the image.

        returns: list
            Returns list of images corresponding to the rows of the dataframe
        """

        df_images = list()
        for i in range(0, len(df)):
            df_images.append(self.__get_image_array(df.iloc[i], img_size))

        return df_images
