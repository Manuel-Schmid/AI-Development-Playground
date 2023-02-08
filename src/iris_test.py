# load iris dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

# store feature matrix (x) and response vector (y)
x = iris.data
y = iris.target

# store feature and target names
feature_names = iris.feature_names
target_names = iris.target_names

# print features and target names of dataset
print("Feature names:", feature_names)
print("Target names:", target_names)

# print first 5 input rows
print("\nFirst 5 rows of x:\n", x[:5])


arr = np.array([[1, 2, 6], [3, 4, 1], [5, 6, 4], [7, 8, 2]])
print(arr)

df = pd.DataFrame(arr, columns=['1st', '2nd', '3rd'], index=['0:', '1:', '2:', '3:'])