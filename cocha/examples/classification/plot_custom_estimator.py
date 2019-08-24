"""
Show the decision boundary for the CustomClassifier
on the sklearn "moons" dataset
"""

# Author: RA

from sklearn.datasets import make_moons

import pandas as pd
import numpy as np

from ailz_tools.classification.topt import CustomEstimator

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc
from collections import Counter

import matplotlib.pyplot as plt


# Regressor
model = CustomEstimator(LogisticRegression(solver='lbfgs'))

# Generate dataset
rs = np.random.RandomState(2)
(X, y) = make_moons(n_samples=100, noise=0.1, random_state=rs)

# This is just to create some imbalance
(X, _, y, _) = train_test_split(X, y, train_size=0.93, random_state=rs)
print("Dataset labels", Counter(y))

# Fit on the training set
model = model.fit(X, y)

# Plot dataset

(fig, ax) = plt.subplots()

df = pd.DataFrame(data={'x': X[:, 0], 'y': X[:, 1], 'label': y})

for (label, group) in df.groupby('label'):
	ax.scatter(group['x'], group['y'], label=label, c={0: "Red", 1: "Blue"}[label])

(xx, yy) = np.meshgrid(np.linspace(*ax.get_xlim(), 79), np.linspace(*ax.get_ylim(), 71))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('RdBu'), alpha=.8, zorder=-5)

plt.show()


exit(39)


from ailz_tools.classification.topt import CustomEstimator

# model = CustomEstimator().fit(X, y)
model = LogisticRegression(solver='lbfgs').fit(X, y)
p = model.predict_proba(X)[:, 1]
gini_threshold(y, p)
ax.plot(p, y, '.')
plt.show()
exit(39)

(fpr, tpr, thresholds) = roc_curve(y, model.predict(X))
gini = (2 * auc(fpr, tpr)) - 1
print(gini)

gini_threshold(y, model.predict(X))

# print(np.mean(model.predict(X)))
ax.scatter(y, model.predict(X), alpha=0.1)
plt.show()


