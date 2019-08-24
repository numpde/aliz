# Author: RA

import pytest

from sklearn.utils.estimator_checks import check_transformer_general

from ailz_tools.classification.topt import CustomEstimator, ThresholdBinarizer


def test_basic_sklearn_compliance():
	check_transformer_general("ThresholdBinarizer", ThresholdBinarizer)

