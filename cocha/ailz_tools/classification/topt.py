"""
Classification with threshold optimization
"""

# Author: RA

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils import validation


VALIDATION_1D_WARN = True
VALIDATION_ACCEPT_SPARSE = False


def compute_threshold_gini(y, p):
	"""Computes the optimal Gini impurity threshold.

	Given a 1d array of labels y with a corresponding
	float feature vector p, computes a threshold for
	the feature such that y is partitioned into two "leafs"
	while minimizing the weighted Gini impurity indicator,
	similarly to decision trees.

	This function is a conceptual implementation,
	so it is neither totally accurate nor efficient.

	Parameters
	----------
	y : ndarray, shape (n_samples,)
	    Array of labels.

	p : ndarray, shape (n_samples,)
	    Array of float features.

	Returns
	-------
	t : float
	    The "Gini threshold".
	"""

	df = pd.DataFrame(data={'p': p, 'y': y}).sort_values(by='p')

	def gini_impurity(c: pd.Series):
		return 1 - np.sum(np.power(c / np.sum(c), 2))

	def gini_split_impurity(t: float):
		L = df[df.p < t].groupby('y').count()['p']
		R = df[df.p >= t].groupby('y').count()['p']
		return (gini_impurity(L) * L.sum() + gini_impurity(R) * R.sum()) / (L.sum() + R.sum())

	t = minimize_scalar(gini_split_impurity, bracket=[min(p), max(p)]).x

	return t


class CustomEstimator(BaseEstimator, ClassifierMixin):
	"""Binary logistic regression classifier with a Gini threshold binarizer.

	The class concatenates sklearn's LogisticRegression
	with a class binarizer. The class binarizer is based
	on the Gini impurity index similar to decision trees.

	Only the member functions fit and predict are implemented.

	Parameters
	----------
	estimator : object or None
	    The underlying estimator.
	    Defaults to LogisticRegression(solver='lbfgs').


	Attributes
	----------

	logistic_ : object
	    The underlying estimator.

	Example
	--------
	>>> from sklearn.datasets import make_moons
	>>> (X, y) = make_moons(n_samples=100, noise=0.1)
	>>> model = CustomEstimator(LogisticRegression(solver='lbfgs')).fit(X, y)
	"""

	def __init__(self, estimator):
		if estimator:
			self.logistic_ = estimator
		else:
			self.logistic_ = LogisticRegression(solver='lbfgs')

	def fit(self, X, y=None, sample_weight=None):
		y = validation.column_or_1d(y, warn=VALIDATION_1D_WARN)

		# expected class labels for binary classification
		CLASS0_LABEL = 0
		CLASS1_LABEL = 1

		assert (set(y).issubset({CLASS0_LABEL, CLASS1_LABEL})), \
			"Expected class labels: {}, {}".format(CLASS0_LABEL, CLASS1_LABEL)

		# fit the logistic part of the model
		self.logistic_.fit(X, y=y, sample_weight=sample_weight)

		# predict_proba-index of the "positive" class
		self.class1_index_ = self.logistic_.classes_.tolist().index(CLASS1_LABEL)

		# estimate P(y_pred in class1)
		p = self.logistic_.predict_proba(X)[:, self.class1_index_]

		# binarizer threshold
		self.threshold_ = compute_threshold_gini(p, y)

		self.is_fitted_ = True
		return self

	def predict(self, X):
		X = validation.check_array(X, accept_sparse=VALIDATION_ACCEPT_SPARSE)
		validation.check_is_fitted(self, ['is_fitted_', 'class1_index_', 'threshold_'])

		# estimate of P(y in class1)
		p = self.logistic_.predict_proba(X)[:, self.class1_index_]

		# apply binarizer
		y = (p >= self.threshold_)

		return y
