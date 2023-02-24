import os

import pandas as pd
import string
import numpy as np
import json

from tensorflow import keras
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku

import tensorflow as tf

tf.random.set_seed(2)
from numpy.random import seed

seed(1)

# load all the datasets
df1 = pd.read_csv('../../data/USvideos.csv')
# df2 = pd.read_csv('../../data/CAvideos.csv')
# df3 = pd.read_csv('../../data/GBvideos.csv')

# load the datasets containing the category names
data1 = json.load(open('../../data/US_category_id.json'))
# data2 = json.load(open('CA_category_id.json'))
# data3 = json.load(open('GB_category_id.json'))


def category_extractor(data):
    i_d = [data['items'][i]['id'] for i in range(len(data['items']))]
    title = [data['items'][i]['snippet']['title'] for i in range(len(data['items']))]
    i_d = list(map(int, i_d))
    category = zip(i_d, title)
    category = dict(category)
    return category


# create a new category column by mapping the category names to their id
df1['category_title'] = df1['category_id'].map(category_extractor(data1))
# df2['category_title'] = df2['category_id'].map(category_extractor(data2))
# df3['category_title'] = df3['category_id'].map(category_extractor(data3))

# join the dataframes
# df = pd.concat([df1, df2, df3], ignore_index=True)
df = pd.concat([df1], ignore_index=True)

# drop rows based on duplicate videos
df = df.drop_duplicates('video_id')

# collect only titles of entertainment videos
# feel free to use any category of video that you want
entertainment = df[df['category_title'] == 'Entertainment']['title']
entertainment = entertainment.tolist()


# remove punctuations and convert text to lowercase
def clean_text(text):
    text = ''.join(e for e in text if e not in string.punctuation).lower()

    text = text.encode('utf8').decode('ascii', 'ignore')
    return text


corpus = [clean_text(e) for e in entertainment]


tokenizer = Tokenizer()


def get_sequence_of_tokens(corpus):
    # get tokens
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # convert to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
    input_sequences.append(n_gram_sequence)

    return input_sequences, total_words


inp_sequences, total_words = get_sequence_of_tokens(corpus)


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


