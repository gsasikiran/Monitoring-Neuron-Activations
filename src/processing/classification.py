import sys

import keras
from keras.applications.resnet import ResNet50
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 5

x_train = x_train.reshape(224, 224, 3)
x_test = x_test.reshape(224, 224, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = ResNet50()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=0, validation_data=(x_test, y_test))

print("Epoch accuracy", history.history['acc'])
score = model.evaluate(x_test, y_test, verbose=0)

print('accuracy on test set:', score[5])

sys.exit()
