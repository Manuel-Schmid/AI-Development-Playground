import sklearn
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
# import matplotlib.pyplot as plt

# Data Preparation
digits = load_digits()  # load handwritten digits dataset
data = scale(digits.data)  # scale numbers down to range between -1 & 1
y = digits.target

# k = len(np.unique(y))
k = 10  # amount of centroids
samples, features = data.shape


# Model Training
def bench_k_means(estimator, name, _data):  # function from sklearn for formatted output
    estimator.fit(_data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(_data, estimator.labels_,
                                      metric='euclidean')))


clf_model = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf_model, "1", data)


# data = np.array([
#     [1, 2],
#     [2, 3],
#     [3, 6],
#     [4, 2],
#     [8, 4],
#     [4, 6],
#     [6, 6],
#     [5, 1],
#     [8, 4],
#     [9, 3],
#     [5, 2],
#     [3, 9],
#     [4, 7],
# ])
# x, y = data.T
# print(x)
# plt.scatter(x, y)
# # plt.show()
#
#
# # centroids
# plt.scatter(3, 1)
# plt.scatter(8, 7)
# plt.show()
#
#
# for p in data:
#     print(p)
