import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import os
from src.config.definitions import ROOT_DIR

# import warnings
# warnings.filterwarnings('ignore')


df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'TSLA.csv'))

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

# plt.figure(figsize=(15, 5))
# plt.plot(df['Close'])
# plt.plot(df['Open'])
# plt.plot(df['High'])
# plt.plot(df['Low'])
# plt.plot(df['Close'] - df['Open'])
# plt.plot(df['High'] - df['Low'])
# plt.title('Close price', fontsize=15)
# plt.ylabel('Price in dollars')
# plt.xlabel('Time')
# plt.show()

# print(df[df['Close'] == df['Adj Close']].shape) # check if all values in this column are equal -> column is redundant
df = df.drop(['Adj Close'], axis=1)  # drop redundant column
# print(df.isnull().sum())  # check for null values

# features = ['Open', 'High', 'Low', 'Close']
# plt.subplots(figsize=(25, 10))
# plt.clf()
# for i, col in enumerate(features):
#     plt.subplot(2, 2, i + 1)
#     plt.plot(df[col])
# plt.show()

# features = ['Open', 'High', 'Low', 'Close', 'Volume']
# plt.subplots(figsize=(20, 10))
# for i, col in enumerate(features):
#     plt.subplot(2, 3, i + 1)
#     sb.boxplot(df[col])
# plt.show()


splitted = df['Date'].str.split('.', expand=True)  # split date column by '.'
df['day'] = splitted[0].astype('int')  # creates new columns
df['month'] = splitted[1].astype('int')
df['year'] = 2000 + splitted[2].astype('int')
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)  # new col
# print(df.head(60))


# data_grouped = df.groupby('year').mean()
# plt.subplots(figsize=(20, 10))
#
# for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
#     plt.subplot(2, 2, i + 1)
#     data_grouped[col].plot.bar()
# plt.show()

# print(df.groupby('is_quarter_end').mean(numeric_only=True))

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
# print(X_train.shape, X_valid.shape)

# print(df.head(60))

# plt.pie(df['target'].value_counts().values,  # is target balanced?
#         labels=[0, 1], autopct='%1.1f%%')
# plt.plot(df['target'])
# plt.show()
# plt.figure(figsize=(10, 10))

mode = input('Do you want to plot datapoints (P), train a new model (T) or run predictions with the saved model (R)? : ').upper()


# Plotting
if mode == 'P':
    # visualize heatmap of highly correlated features only.
    sb.heatmap(df.corr(numeric_only=True) > 0.9, annot=True, cbar=False)
    plt.show()


# Training Model
if mode == 'T':
    models = [LogisticRegression(), SVC(
        kernel='poly', probability=True), XGBClassifier()]

    for i in range(3):
        models[i].fit(X_train, Y_train)

        print(f'{models[i]} : ')
        print('Training Accuracy : ', metrics.roc_auc_score(
            Y_train, models[i].predict_proba(X_train)[:, 1]))
        print('Validation Accuracy : ', metrics.roc_auc_score(
            Y_valid, models[i].predict_proba(X_valid)[:, 1]))
        print()

    models[0].fit(X_train, Y_train)
    with open(os.path.join(ROOT_DIR, 'models', 'stock-predictor-model.pickle'), "wb") as f:  # save trained model in pickle file
        pickle.dump(models[0], f)


# Using the model
if mode == 'R':
    pickle_in = open(os.path.join(ROOT_DIR, 'models', 'stock-predictor-model.pickle'), "rb")  # load trained model from pickle file
    model = pickle.load(pickle_in)

    predictions = model.predict(X_valid)
    cm = confusion_matrix(Y_valid, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()
