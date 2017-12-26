#!/usr/bin/env python
# -*- coding: utf-8  -*-
"""Decision tree, entropy, gini index and accuracy_score.

Module implementing:
    1. Discrete valued binary DecisionTree class with custom tree depth
    2. Gini and entropy metrics

"""

import sys

import numpy as np
import pandas as pd

__all__ = ['entropy', 'gini', 'DecisionTree']


def entropy(dataset) -> float:
    """Calculates entropy of a dataset.

    :param dataset: Dataset to calculate entropy for
    :param epsilon: Small constant to avoid division by zero
    :returns: evaluated entropy for dataset as np.array

    """
    _, frequencies = np.unique(dataset, return_counts=True)
    frequencies = frequencies / len(dataset)
    return np.sum(np.multiply(frequencies, -np.log2(frequencies)))


def gini(dataset) -> float:
    """Calculates gini index for dataset

    :param dataset: Dataset to calculate gini index for
    :param epsilon: Small constant to avoid division by zero
    :returns: evaluated gini for dataset as np.array

    """
    _, frequencies = np.unique(dataset, return_counts=True)
    frequencies = frequencies / len(dataset)
    return 1 - np.sum(frequencies**2)


class DecisionTree:
    """Class implementing Decision Tree classifier used in random forest.
    """

    class Node:
        """Node of DecisionTree (more readable than nested dict).
        """

        def __init__(self):
            """__init__"""
            self.label = None
            self.attribute = None
            self.value = None
            self.left = None
            self.right = None

        def __str__(self):
            """Representation of the Node.

            Representation form: L[abel], A[ttribute], V[alue].
            """
            return "L:{}, A:{}, V:{}".format(self.label, self.attribute,
                                             self.value)

    def __init__(self, criterion='gini', max_depth=200):
        """__init__

        :param criterion: Critertion for comparison (gini or entropy or own
        function)
        :param max_depth: Maximum depth of the tree
        """
        if criterion == 'gini':
            self.criterion = gini
        elif criterion == 'entropy':
            self.criterion = entropy
        else:
            print(
                'Unknown metric value (neither gini nor entropy), \
                    use at your own risk!',
                file=sys.stderr)
            self.criterion = self.criterion

        self.max_depth = max_depth
        self.tree = None

    @staticmethod
    def _split_on_condition(array: np.array, condition: np.array):
        """Implementation detail

        :param array: array to be splitted
        :param condition: condition on which array should be splitted
        """
        return np.array([array[condition], array[~condition]])

    def binary_measurement(self, dataset, column: int, value: int):
        """binary_measurement

        :param dataset: dataset containg features and labels (np.array)
        :param column: column to split dataset on
        :param value: value in column to split dataset on
        """
        groups = self._split_on_condition(dataset[-1], dataset[column] < value)

        groups_relative_size = [
            group.shape[0] / dataset.shape[1] for group in groups
        ]
        return np.sum(
            self.criterion(group) * group_relative_size
            for group, group_relative_size in zip(groups,
                                                  groups_relative_size))

    def fit(self, X, y):
        """Fits classifier accordingly to dataset.

        :param dataset: Dataset with feature columns (Pandas or numpy arrays)
        :param y: Dataset with respective labels (Pandas/numpy column)
        :returns: self for easier creation of random trees

        """

        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(X, pd.DataFrame) else y

        def __recursive_fit(dataset, node, depth):
            """Implementation detail.

            :param dataset: Dataset to fit on
            :param node: Current node
            :param depth: Current depth of subtree
            """
            labels, frequencies = np.unique(dataset[-1], return_counts=True)
            if len(labels) == 0:
                return None
            node.label = labels[np.argmax(frequencies)]
            if depth == self.max_depth:
                return node

            labels = np.array([np.unique(column) for column in dataset[:-1]])
            unique_indices = list(enumerate(labels))
            information_gain = [[
                self.binary_measurement(dataset, column, value)
                for value in values
            ] for column, values in unique_indices]

            node.attribute = np.argmin(
                np.array([np.min(column) for column in information_gain]))
            node.value = (np.array([
                np.argmin(column) for column in information_gain
            ]))[node.attribute]

            groups = self._split_on_condition(
                dataset.T, dataset[node.attribute] < node.value)

            node.left = __recursive_fit(groups[0].T, DecisionTree.Node(),
                                        depth + 1)
            node.right = __recursive_fit(groups[1].T, DecisionTree.Node(),
                                         depth + 1)
            return node

        self.tree = __recursive_fit(np.c_[X, y].T, DecisionTree.Node(), 0)
        return self

    def print(self):
        """Print visual representation of tree."""

        def __recursive_print(root, indent):
            if root.left is not None:
                __recursive_print(root.left, indent + 1)
            if root.right is not None:
                __recursive_print(root.right, indent + 1)

        __recursive_print(self.tree, 0)

    def predict(self, X):
        """Predicts label for Pandas or Numpy matrix.

        :param X: Dataset with feature columns (Pandas or numpy arrays)
        :returns: Numpy array with predictions

        """

        def __recursive_predict(example, node):
            if node.value is None:
                return node.label
            if example[node.attribute] < node.value:
                if node.left is not None:
                    return __recursive_predict(example, node.left)
                return node.label
            else:
                if node.right is not None:
                    return __recursive_predict(example, node.right)
                return node.label

        X = X.values if isinstance(X, pd.DataFrame) else X
        return np.array(
            [__recursive_predict(example, self.tree) for example in X])
