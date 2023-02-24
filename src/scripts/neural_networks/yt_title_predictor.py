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
df2 = pd.read_csv('../../data/CAvideos.csv')
df3 = pd.read_csv('../../data/GBvideos.csv')

# load the datasets containing the category names
data1 = json.load(open('../../data/US_category_id.json'))
data2 = json.load(open('CA_category_id.json'))
data3 = json.load(open('GB_category_id.json'))
