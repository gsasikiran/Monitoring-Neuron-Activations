import warnings

from keras.models import Model

from src.post_processing.visualization import DisplayActivations
from src.pre_processing.read_data import Data
from src.processing.model_architecture import ModelArchitecture
from src.processing.tools import Tools

warnings.filterwarnings("ignore")

# img = image_preprocess_tri("/home/gsasikiran/Desktop/Semester_3/SDP/Monitoring-Neuron-Activations/triangles/drawing(1).png")

TRAIN_PATH = "/home/gsasikiran/Desktop/Semester_3/SDP/Monitoring-Neuron-Activations/MNIST_data/train.csv"
TEST_PATH = "/home/gsasikiran/Desktop/Semester_3/SDP/Monitoring-Neuron-Activations/MNIST_data/test.csv"
MNIST_data = Data(TRAIN_PATH, TEST_PATH)

X_train, y_train = MNIST_data.get_data()
X_train_images = MNIST_data.df_to_images(X_train)

model_architecture = ModelArchitecture()
model_cnn = model_architecture.CNN()

tools = Tools()
conv_layer_list = tools.get_conv_layers(model_cnn)

outputs = [model_cnn.layers[i].output for i in conv_layer_list]
model = Model(inputs=model_cnn.inputs, outputs=outputs)

img_num = int(input('Input some number'))
img = X_train_images[img_num]
feature_maps = model.predict(img.reshape(1, 28, 28, 1))

layer_num = int(input('Input layer number to be plotted.. (If NONE plots all the layer activations)'))
display_activation = DisplayActivations()
display_activation.plot_activation(feature_maps, layer_num)
