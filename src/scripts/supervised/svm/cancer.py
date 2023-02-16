import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Data Preparation
cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)  # labels of input data
# print(cancer.target_names)  # target classes to predict

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


# Model Training
clf_model = svm.SVC(kernel="linear", C=4)  # create Support Vector classifier model, C = soft margin
# clf_model = KNeighborsClassifier(8)  # create KNN classifier model

clf_model.fit(x_train, y_train)  # train model

y_pred = clf_model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
