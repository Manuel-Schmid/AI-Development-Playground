import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb  # fetch IMDB movie-review dataset

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)  # only take 10'000 most frequent words

print(train_data[0])

word_index = data.get_word_index()  # gets words w. indices

word_index = {k: (v+3) for k, v in word_index.items()}  # increase all indices by 3 so special chars can be added
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # swap values & keys


def decode_review(text):  # decode integer array to human-readable text
    return " ".join([reverse_word_index.get(i, "?") for i in text])


print(decode_review(test_data[76]))
