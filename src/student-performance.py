import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("../data/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]  # only keep useful columns

predict = "G3"  # predict G3 (final grade)

x = np.array(data.drop([predict], 1))  # dataframe without G3
y = np.array(data[predict])  # dataframe with only G3 values
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# Training
# best = 0
# for _ in range(10000):  # trains 30 models and overwrites model each time if performance is better than previous one
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)  # find best fit line using linear regression
#     acc = linear.score(x_test, y_test)  # check model accuracy
#
#     if acc > best:
#         print(acc)
#         best = acc
#         with open("studentmodel.pickle", "wb") as f:  # save trained model in pickle file
#             pickle.dump(linear, f)


# Using the Model
# pickle_in = open("studentmodel.pickle", "rb")  # load trained model from pickle file
# linear = pickle.load(pickle_in)
#
# print("Coefficient: \n", linear.coef_)
# print("Intercept: \n", linear.intercept_)
#
# predictions = linear.predict(x_test)
#
# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])  # print predicted grades, used input values and actual final grades


# p = "G1"  # x axis values, change this to analyse correlations
# style.use("ggplot")
# pyplot.scatter(data[p], data["G3"])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()