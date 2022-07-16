from __future__ import annotations
from typing import NoReturn

import sklearn.metrics

from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.random_forest_classifier_ = RandomForestClassifier(n_estimators=80)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.random_forest_classifier_.fit(X, y)
        # print(np.round(self.random_forest_classifier_.feature_importances_,4))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = self.random_forest_classifier_.predict(X)
        return pred

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        pred = self._predict(X)
        true_positive = np.sum(np.array(y) * pred == 1)
        true_negative = np.sum((np.array(y) + pred) == 0)
        positive = np.sum(y)
        print("num positive in y = ", positive)
        print("num positive in prediction = ", np.sum(pred))
        print("true positive = ", true_positive)
        print("true negative = ", true_negative)
        # print("zero one acc: ", 1 - sklearn.metrics.zero_one_loss(y, pred))
        return f1_score(pred, y, average='macro')
