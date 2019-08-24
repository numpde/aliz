# Author: RA

import pytest

from sklearn.utils.estimator_checks import *

from ailz_tools.classification.topt import CustomEstimator


def test_basic_sklearn_compliance():
	check_estimators_unfitted("CustomEstimator", CustomEstimator)
	check_estimator("CustomEstimator", CustomEstimator)

