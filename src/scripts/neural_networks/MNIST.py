import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist  # fetch MNIST fashion dataset

(train_images, train_labels), (test_images, test_labels) = data.load_data()  # split data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0  # divide pixel values by 255 -> smaller range
test_images = test_images / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),  # flattened input layer
                          keras.layers.Dense(128, activation="relu"),  # hidden layer (128 neurons)
                          keras.layers.Dense(10, activation="softmax")  # output layer (10 neurons)
                          ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # model settings

model.fit(train_images, train_labels, epochs=5)  # epochs = how many times the same image comes up

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ", test_acc)


# plt.imshow(train_images[7], cmap=plt.cm.binary)  # print an image in b & w
# plt.show()

# print(class_names[train_labels[6]])



