import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import os
from src.config.definitions import ROOT_DIR


data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'census_income.csv'), sep=",")
data = data[["age", "education-num", "hours-per-week", "income-over-50k"]]  # only keep useful columns
# data[""] = data[""].map({'no': 0, 'yes': 1})  # remap values to integers
predict = "income-over-50k"  # predict income

x = np.array(data.drop([predict], axis=1))  # dataframe without G3
y = np.array(data[predict])  # dataframe with only G3 values
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

mode = input('Do you want to plot datapoints (P) or train a new model (T)? : ').upper()


# Plotting
if mode == 'P':
    p = "hours-per-week"  # x-axis values, change this to analyse correlations
    style.use("ggplot")
    pyplot.scatter(data[p], data["income-over-50k"])
    pyplot.xlabel(p)
    pyplot.ylabel("Income over 50k")
    pyplot.show()


# Training
if mode == 'T':
    best = 0
    for _ in range(1000):  # trains 30 models and overwrites model each time if performance is better than previous one
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)  # find best fit line using linear regression
        acc = linear.score(x_test, y_test)  # check model accuracy

        if acc > best:
            print(acc)
            best = acc
            with open(os.path.join(ROOT_DIR, 'models', 'census-income-model.pickle', 'wb')) as f:  # save trained model in pickle file
                pickle.dump(linear, f)
