#!/usr/bin/env python3
# -*- coding: utf-8  -*-
"""decision_tree.

Module implementing:
    1. Discrete valued DecisionTree class with post-prunning using entropy
    2. Gini and entropy metrics
    3. Accuracy measurement between two labels

"""

import copy
import operator

import numpy as np
import pandas as pd

__all__ = ['accuracy_score', 'entropy', 'gini', 'DecisionTree']


def accuracy_score(y_true, y_pred):
    """accuracy_score.

    :param y_true: True values of y (numpy or Pandas column)
    :param y_pred: Predicted values of y (numpy or Pandas column)

    """
    return np.mean(y_true == y_pred)


def entropy(dataset, epsilon=1e-7):
    """entropy.

    Calculates entropy of a dataset

    :param dataset: Dataset to calculate entropy for
    :param epsilon: Small constant to avoid division by zero

    """
    _, frequencies = np.unique(dataset, return_counts=True)
    frequencies = frequencies / (len(dataset) + epsilon)
    return np.sum(np.multiply(frequencies, -np.log2(frequencies)))


def gini(dataset, epsilon=1e-7):
    """gini.

    Calculates gini index for dataset

    :param dataset: Dataset to calculate gini index for
    :param epsilon: Small constant to avoid division by zero

    """
    _, frequencies = np.unique(dataset, return_counts=True)
    frequencies = frequencies / (len(dataset) + epsilon)
    return np.sum(frequencies**2)


class DecisionTree:
    """DecisionTree."""

    @staticmethod
    def mutual_information(y, x, epsilon=1e-7):
        """mutual_information.

        Calculates mutual information between x and labels y

        :param y: Pandas or numpy column with labels
        :param x: Column with
        :param epsilon: Small constant to avoid division by zero

        """
        values, frequencies = np.unique(x, return_counts=True)
        frequencies = frequencies / (len(x) + epsilon)
        return entropy(y, epsilon) - np.sum([
            probability * entropy(y[x == value])
            for probability, value in zip(frequencies, values)
        ])

    @staticmethod
    def partition(dataset):
        """partition. Partitions dataset based on values of attribute.

        :param dataset: Dataset to partition
        :returns: Dictionary {class_value: elements}

        """
        return {c: (dataset == c).nonzero()[0] for c in np.unique(dataset)}

    def _create_tree(self, X, y):
        """_create_tree.

        :param X:
        :param y: Dataset with respective labels (Pandas/numpy column)

        """
        labels, frequencies = np.unique(y, return_counts=True)
        if len(labels) == 1:
            return labels[0]
        gain = np.array(
            [self.mutual_information(y, x_column) for x_column in X.T])
        attribute = np.argmax(gain)
        most_frequent_label = max(
            dict(zip(labels, frequencies)).items(),
            key=operator.itemgetter(1))[0]

        sets = self.partition(X[:, attribute])
        subtree = {}
        for key, values in sets.items():
            x_subset = X.take(values, axis=0)
            y_subset = y.take(values, axis=0)

            subtree[(attribute, key)] = [
                self._create_tree(x_subset, y_subset), most_frequent_label
            ]

        return subtree

    def __init__(self):
        """__init__.

        Non-parametric constructor for class

        """
        self.tree = None
        self.most_frequent = None

    def fit(self, X, y):
        """fit.

        Fits classifier accordingly to dataset

        :param X: Dataset with feature columns (Pandas or numpy arrays)
        :param y: Dataset with respective labels (Pandas/numpy column)

        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(X, pd.DataFrame) else y

        values, frequencies = np.unique(y, return_counts=True)
        most_frequent = max(
            dict(zip(values, frequencies)).items(),
            key=operator.itemgetter(1))[0]

        self.tree = [self._create_tree(X, y), most_frequent]

    def rules_count(self):
        """rules_count.

        :returns: Number of rules created by this classifier

        """

        def recursive_count(root, rules=0):
            """recursive_count.

            :param root: Root of the classifier
            :param rules: How many dictionaries are in a given subnode, this
            parameter should not be used

            """
            if not isinstance(root[0], dict) or not root:
                return rules
            return sum(
                recursive_count(values, rules + 1)
                for _, values in root[0].items())

        return recursive_count(self.tree)

    def prune(self, validation_X, validation_y, epsilon=0):
        """prune.

        Performs post-prunning on created classifier using validation datasets.
        If accuracy on validation set increased, node is removed.
        Removes nodes in top down fashion (faster execution)

        :param validation_X: Pandas or numpy dataset with features
        :param validation_y: Pandas or numpy column with respective labels
        :param epsilon: If post_prunning_accuracy > pre_prunning_accuracy -
        epsilon, remove node. Factor for accuracy increase (can be negative)

        """
        validation_y = validation_y.values if isinstance(
            validation_y, pd.DataFrame) else validation_y
        validation_X = validation_X.values if isinstance(
            validation_X, pd.DataFrame) else validation_X

        roots = [self.tree[0][key] for key in self.tree[0]]
        for root in roots:
            if isinstance(root[0], dict):
                base_accuracy = accuracy_score(
                    self.predict(validation_X), validation_y)
                root[0], backup = root[1], copy.deepcopy(root[0])
                prunned_accuracy = accuracy_score(
                    self.predict(validation_X), validation_y)
                if prunned_accuracy < base_accuracy - epsilon:
                    root[0] = backup
                    roots.extend([root[0][key] for key in root[0]])

    def predict(self, X):
        """predict.

        :param X: Dataset with feature columns (Pandas or numpy arrays)
        :returns: Numpy array with predictions

        """

        def _recursive_predict(keys, root):
            """_recursive_predict recursively predict label for classifier.

            :param keys:
            :param root: root of classification tree
            :returns: label for a given example

            """
            if isinstance(root[0], dict):
                attribute_value = list(root[0].keys())[0][0]
                data_value = keys[attribute_value]
                next_root = root[0].get((attribute_value, data_value))
                #
                # IF NO RULE FOUND RETURN MOST FREQUENT CLASS IN DATASET
                if next_root is None:
                    return root[1]
                return _recursive_predict(keys, next_root)
            return root[0]

        X = X.values if isinstance(X, pd.DataFrame) else X
        return np.array(
            [_recursive_predict(dict(enumerate(x)), self.tree) for x in X])
