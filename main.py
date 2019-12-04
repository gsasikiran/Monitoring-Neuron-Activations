from keras.models import Model

from src.processing.model_architecture import ModelArchitecture
from src.processing.tools import Tools
from src.pre_processing.read_data import Data

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
img = X_train_images[5]
feature_maps = model.predict(img.reshape(1, 28, 28, 1))

print(feature_maps)

# display_activation(feature_maps,8,4,2)

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
