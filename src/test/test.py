import unittest
import numpy as np


class Output(unittest.TestCase):

    def test_string(self):
        output = "class name"
        assert type(output) == str

    def test_class(self):
        pass


class TestActivationPattern(unittest.TestCase):

    def test_shape(self):
        saved_activation = np.array([[1, 2], [3, 4]])
        test_activation = np.array([[1, 3], [2, 4]]).T
        self.assertTrue((saved_activation == test_activation).all())


if __name__ == "__main__":
    unittest.main()
