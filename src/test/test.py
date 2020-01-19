import unittest

from src.pre_processing.read_data import Data


class Image(unittest.TestCase):

    def image_type(self):
        # self.train_path = train_path
        # self.test_path = test_path
        pass

    def test_image_shape(self):

        train_path = '../../../MNIST_data/train.csv'
        test_path = '../../../MNIST_data/test.csv'
        MNIST_data = Data(path_train=train_path, path_test=test_path)
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
