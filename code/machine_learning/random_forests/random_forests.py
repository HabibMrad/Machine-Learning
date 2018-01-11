#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module implementing Random Forest Discrete Classifier."""

import multiprocessing
import typing

import numpy as np
import pandas as pd

from binary_decision_tree import DecisionTree


class RandomForest:
    """Class implementing random forest classifier."""

    def __init__(self,
                 *,
                 number_of_estimators: int = 10,
                 criterion: str = 'gini',
                 number_of_features: typing.Union[int, typing.Callable[
                     [int], int]] = None,
                 max_depth: int = 10,
                 number_of_jobs: int = multiprocessing.cpu_count()):
        """Class constructor.

        :param number_of_estimators: int how many estimators should be created?
        :param criterion: str Comparison criterion for estimators
        :param number_of_features: custom Number of features to be considered
        for each tree (default: sqrt(X.shape[1]), see fit method)
        :param max_depth: Maximum depth for each tree
        :param number_of_jobs: How many threads should be used for fit and
        predict algorithm (default: same as number of cores)

        """

        self.number_of_estimators = number_of_estimators

        self.criterion = criterion
        self.number_of_features = number_of_features
        self.max_depth = max_depth
        self.number_of_jobs = number_of_jobs
        self.trees = []

    def fit(self, X, y):
        """Fits classifier accordingly to dataset.

        :param dataset: Dataset with feature columns (Pandas or numpy arrays)
        :param y: Dataset with respective labels (Pandas/numpy column)
        :returns: self

        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y

        if self.number_of_features is None:
            self.number_of_features = int(np.sqrt(X.shape[1]))
        elif self.number_of_features is not int:
            self.number_of_features = self.number_of_features(X.shape[1])

        with multiprocessing.Pool(self.number_of_jobs) as pool:
            async_trees = []
            for _ in range(self.number_of_estimators):
                indices = np.random.randint(
                    0, X.shape[1], size=self.number_of_features)
                subset = X[:, indices]
                async_trees.append(
                    pool.apply_async(
                        DecisionTree(self.criterion, self.max_depth).fit, (
                            subset,
                            y,
                        )))
            self.trees.extend([async_tree.get() for async_tree in async_trees])
        return self

    def predict(self, X) -> np.array:
        """Predicts label for Pandas or Numpy matrix.

        :param X: Dataset with feature columns (Pandas or numpy arrays)
        :returns: np.array Numpy array with predictions

        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        with multiprocessing.pool.Pool(self.number_of_jobs) as pool:
            results = []
            for tree in self.trees:
                results.append(pool.apply_async(tree.predict, (X, )))
            results = np.array([result.get() for result in results]).T
            return np.array(
                [np.bincount(example).argmax() for example in results])
