import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import click

data = pd.read_csv("../../../data/student-mat.csv", sep=";")
data = data[data.G3 != 0]
p2 = "internet"
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]  # only keep useful columns
# data[p2] = data[p2].map({'no': 0, 'yes': 1})  # remap values to integers
predict = "G3"  # predict G3 (final grade)

x = np.array(data.drop([predict], axis=1))  # dataframe without G3
y = np.array(data[predict])  # dataframe with only G3 values
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# p = "G2"  # x-axis values, change this to analyse correlations
# p = p2  # x-axis values, change this to analyse correlations
# style.use("ggplot")
# pyplot.scatter(data[p], data["G3"])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()


# Training
best = 0
iterations = 10000
best_model = None
for i in range(iterations):  # trains 10k models and overwrites model each time performance is better than previous one
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)  # find best fit line using linear regression
    acc = linear.score(x_test, y_test)  # check model accuracy

    if i % 1000 == 0:
        print(f'{i}/{iterations}')

    if acc > best:
        print(i, ":", acc)
        best = acc
        best_model = linear

    if i == iterations - 1:
        if click.confirm(f'Do you want to save the model? (Accuracy: {best})', default=True):
            with open("../../../models/studentmodel.pickle", "wb") as f:  # save trained model in pickle file
                pickle.dump(best_model, f)


# Using the Model
# """
pickle_in = open("../../../models/studentmodel.pickle", "rb")  # load trained model from pickle file
linear = pickle.load(pickle_in)

# print("Coefficient: \n", linear.coef_)
# print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

predict_sum = 0
actual_sum = 0
for x in range(len(predictions)):
    predict_sum += round(predictions[x])
    actual_sum += y_test[x]
    # x_test[x]  # input values
    print(round(predictions[x]), y_test[x])  # print predicted grades and actual final grades

print(f'Average Predicted: {predict_sum/len(predictions)}')
print(f'Average Actual: {actual_sum/len(predictions)}')
# """
