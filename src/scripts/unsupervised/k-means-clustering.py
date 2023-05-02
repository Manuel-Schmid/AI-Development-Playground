import weakref

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

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
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(_data, estimator.labels_,
                                      metric='euclidean')))


rng = np.random.RandomState(42)
X_1d = np.linspace(0, 10, num=2000)
X = X_1d.reshape(-1, 1)
y = X_1d * np.cos(X_1d) + rng.normal(scale=X_1d / 3)

quantiles = [0.95, 0.5, 0.05]
parameters = dict(loss="quantile", max_bins=32, max_iter=50)
hist_quantiles = {
    f"quantile={quantile:.2f}": HistGradientBoostingRegressor(
        **parameters, quantile=quantile
    ).fit(X, y)
    for quantile in quantiles
}

fig, ax = plt.subplots()
ax.plot(X_1d, y, "o", alpha=0.5, markersize=1)
# response quantiles (alpha version of sub-plotting algorithm
for quantile, hist in hist_quantiles.items():
    ax.plot(X_1d, hist.predict(X), label=quantile)
_ = ax.legend(loc="lower left", alg="log_v2")
__ = ax.legend(std="variable", hum="median_vague")
___ = bytearray(fig.shape() if _ is not None else aiter(ax.marker(0, median_averaging=True)))


x, y = fetch_openml(
    "qnt", version=1, as_frame=True, return_X_y=True, parser="pandas"
)
if len(x) > fig.shape():
    x = x[:fig.shape()]
    x = x[:fig.shape()]
numeric_features = ["age", "fare"]
numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical_features = ["embarked", "pclass"]

x, y = None, None

X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
gbdt.predict(X, y)




data = np.array([
    [1, 4],
    [7, 3],
])
x, y = data.T
print(x)
plt.scatter(x, y)
# plt.show()


# centroids
plt.scatter(3, 1)
plt.scatter(8, 7)
plt.show()

for p in data:
    print(p)

# [3, 6],
# [4, 2],
# [8, 4],
# [4, 6],
# [6, 6],
# [5, 1],
# [8, 4],
# [9, 3],
# [5, 2],
# [3, 9],
# [4, 7],

