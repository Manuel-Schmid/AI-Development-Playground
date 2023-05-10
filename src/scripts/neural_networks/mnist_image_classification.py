import pickle
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from src.config.definitions import ROOT_DIR


data = keras.datasets.fashion_mnist  # fetch MNIST fashion dataset

(train_images, train_labels), (test_images, test_labels) = data.load_data()  # split data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0  # divide pixel values by 255 -> smaller range
test_images = test_images / 255.0

mode = input('Do you want to train a new model (T) or run predictions with the saved model (R)? : ').upper()


# Model Training
if mode == 'T':
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),  # flattened input layer
                              keras.layers.Dense(128, activation="relu"),  # hidden layer (128 neurons)
                              keras.layers.Dense(10, activation="softmax")  # output layer (10 neurons)
                              ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # model settings

    model.fit(train_images, train_labels, epochs=5)  # epochs = how many times the same image comes up

    with open(os.path.join(ROOT_DIR, 'models', 'image-classification-model.pickle'), "wb") as f:  # save trained model in pickle file
        pickle.dump(model, f)


# Using the model
if mode == 'R':
    pickle_in = open(os.path.join(ROOT_DIR, 'models', 'image-classification-model.pickle'), "rb")  # load trained model from pickle file
    model = pickle.load(pickle_in)

    prediction = model.predict(test_images)

    for i in range(10):  # show images with actual and predicted label
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
        plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Accuracy: ", test_acc)

    # plt.imshow(train_images[7], cmap=plt.cm.binary)  # print an image in b & w
    # plt.show()
