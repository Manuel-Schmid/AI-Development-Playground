# load iris dataset
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
