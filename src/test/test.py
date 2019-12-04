import unittest

import numpy as np

from src.pre_processing.read_data import Data


class Image(unittest.TestCase):

    # def image_type(self):
    #     pass

    def test_image_shape(self):

        MNIST_data = Data()
        x_train, y_train = MNIST_data.get_data()
        x_train_images = MNIST_data.df_to_images(x_train)

        for image in x_train_images:
            self.assertEqual(image.shape, (28, 28))



# class Output(unittest.TestCase):
#
#     def __init__(self):
#         super().__init__()
#
#     def test_string(self):
#         output = "class name"
#         assert type(output) == str
#
#     def test_class(self):
#         pass
#
#
# class TestActivationPattern(unittest.TestCase):
#
#     def test_shape(self):
#         saved_activation = np.array([[1, 2], [3, 4]])
#         test_activation = np.array([[1, 3], [2, 4]]).T
#         self.assertTrue((saved_activation == test_activation).all())


if __name__ == "__main__":
    unittest.main()
