#!/usr/bin/env python
# coding=utf-8
"""Naive bayes classifier."""

import itertools
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator


class sparse_naive_bayes(BaseEstimator):
    def __init__(self, alpha=1, priors=None):
        self.alpha = alpha
        self.priors = priors
        self.class_rows = None

    def _check_features_and_target(self, X, y):
        X_rows, x_columns = np.atleast_2d(X).shape
        y_rows, y_columns = np.atleast_2d(y).shape
        # CHECK IF Y_columns = 1 AND X_rows = y_rows
        if X_rows != y_rows:
            raise ValueError('X (features matrix) has different number of rows'
                             'than y (target vector).')
        if y_columns != 1:
            raise ValueError('y (target vector) has more than one column.')

    def _get_class_rows(self, class_column):
        groups = defaultdict(list)
        for row_index, value in enumerate(class_column.data):
            groups[int(value)].append(row_index)
        return groups

    def fit(self, X, y, sample_weight=None):
        self._check_features_and_target(X, y)
        self.class_rows = self._get_class_rows(y)
        sample_count = X.shape[0]
        # Calculate prior based on class frequency
        if self.priors is None:
            self.priors = [
                np.log(len(rows) / sample_count)
                for _, rows in self.class_rows.items()
            ]

        class_count = np.array([
            abs(X[rows, :]).sum(axis=0) for _, rows in self.class_rows.items()
        ]).reshape(len(self.class_rows), -1) + self.alpha
        self.feature_log_prob_ = np.log(
            class_count / class_count.sum(axis=1)[np.newaxis].T)
        print(self.feature_log_prob_)
        # if sample_weight is not None:
        #     if y.shape != sample_weight.shape:
        #         raise ValueError('y (target vector) has differen shape'
        #                          'than sample_weight.')
        #     y *= sample_weight

        return self

    # SKLEARN'S NAIVE BAYES DOESN'T RETURN DICTIONARY, FOR COMPATIBILITY REASONS
    # LIST OF LISTS IS RETURNED
    def predict_log_proba(self, X):
        return [
            (self.feature_log_prob_ * X.getrow(i).T).sum(axis=1) + self.priors
            for i in range(X.shape[0])
        ]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)
