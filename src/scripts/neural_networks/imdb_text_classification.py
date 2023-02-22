import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb  # fetch IMDB movie-review dataset

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)  # only take 10'000 most frequent words

word_index = data.get_word_index()  # gets words w. indices

word_index = {k: (v+3) for k, v in word_index.items()}  # increase all indices by 3 so special chars can be added
word_index["<PAD>"] = 0  # padding for making all reviews the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # swap values & keys


train_data = keras.preprocessing.sequence.pad_sequences(  # normalize/pad data so every review has a length of 250 words
    train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):  # decode integer array to human-readable text
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# print(decode_review(test_data[76]))

# setup neural network model with layers
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))  # turns word-indices into word-vectors with 16 coefficients (array)
model.add(keras.layers.GlobalAvgPool1D())  # averages vectors out (shrinks their data)
model.add(keras.layers.Dense(16, activation="relu"))  # 16 neurons
model.add(keras.layers.Dense(1, activation="sigmoid"))  # produces single value as result of neuron connections
