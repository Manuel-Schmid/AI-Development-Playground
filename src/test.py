import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("../data/student-mat.csv", sep=";")

# print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"  # predict G3 (final grade)

x = np.array(data.drop([predict], 1))  # dataframe without G3
y = np.array(data[predict])  # dataframe with only G3 values

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)  # find best fit line using linear regression
acc = linear.score(x_test, y_test)  # check model accuracy
print(acc)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])  # print predicted grades, used input values and actual final grades


