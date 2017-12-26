#!/usr/bin/env python
# -*- coding: utf-8  -*-
"""Decision tree, entropy, gini index and accuracy_score.

Module implementing:
    1. Discrete valued DecisionTree class with post-prunning using entropy
    2. Gini and entropy metrics
    3. Accuracy measurement between two labels

"""

import copy
import operator
import pprint
import sys

import numpy as np
import pandas as pd

__all__ = ['accuracy_score', 'entropy', 'gini', 'DecisionTree']


def accuracy_score(y_true, y_pred):
    """Calculates accuracy between two labels

    :param y_true: True values of y (numpy or Pandas column)
    :param y_pred: Predicted values of y (numpy or Pandas column)
    :returns: Accuracy between two labels [np.array]

    """
    return np.mean(y_true == y_pred)


def entropy(dataset):
    """Calculates entropy of a dataset.

    :param dataset: Dataset to calculate entropy for
    :param epsilon: Small constant to avoid division by zero
    :returns: evaluated entropy for dataset as np.array

    """
    _, frequencies = np.unique(dataset, return_counts=True)
    frequencies = frequencies / len(dataset)
    return np.sum(np.multiply(frequencies, -np.log2(frequencies)))


def gini(dataset):
    """Calculates gini index for dataset

    :param dataset: Dataset to calculate gini index for
    :param epsilon: Small constant to avoid division by zero
    :returns: evaluated gini for dataset as np.array

    """
    _, frequencies = np.unique(dataset, return_counts=True)
    frequencies = frequencies / len(dataset)
    return 1 - np.sum(frequencies**2)


class DecisionTree:
    """Class implementing Decision Tree classifier with post prunning and
    various other capabilites.
    """

    @staticmethod
    def _attribute_split(feature):
        """Implementation detail.

        Partitions dataset based on values of
        attribute.

        :param dataset: Dataset to _attribute_split
        :returns: Dictionary {class_value: elements}

        """
        return {
            value: (feature == value).nonzero()[0]
            for value in np.unique(feature)
        }

    @staticmethod
    def _most_frequent_label(labels, frequencies):
        """Implementation detail.

        Returns most frequent labal in dataset

        :param labels: Labels in subdataset
        :param frequencies: How often labels occure
        :returns: most frequent labal in dataset
        """
        return max(
            dict(zip(labels, frequencies)).items(),
            key=operator.itemgetter(1))[0]

    def _create_tree(self, X, y):
        """Implementation detail.

        Recursively creates dataset (used by fit method)

        :param X: Numpy matrix dataset containing features
        :param y: Numpy array with respective labels
        :returns: Fitted tree accordingly to dataset

        """
        labels, frequencies = np.unique(y, return_counts=True)
        # IF PURE RETURN LABEL
        if len(labels) <= 1:
            return labels[0]

        # FIND THE BEST ATTRIBUTE TO SPLIT ON USING MUTUAL INFORMATION
        gain = np.array(
            [self.mutual_information(y, x_column) for x_column in X.T])
        attribute = np.argmax(gain)
        sets = self._attribute_split(X[:, attribute])

        # FIND THE MOST FREQUENTLY OCCURING LABEL IN OUR DATASET/SUBSET
        most_frequent_label = self._most_frequent_label(labels, frequencies)

        subtree = {}
        for attribute_value, values in sets.items():
            x_subset = X.take(values, axis=0)
            y_subset = y.take(values, axis=0)

            subtree[(attribute, attribute_value)] = [
                self._create_tree(x_subset, y_subset), most_frequent_label
            ]

        return subtree

    def _is_tree_valid(self):
        """Implementation detail.
        Checks whether tree has been created (is not None), otherwise prints
        diagnostics to stderr.

        :returns: bool indicating whether tree is valid
        """
        if self.tree is None:
            print(
                'Tree is None, maybe you forgot to call fit function?',
                file=sys.stderr)
            return False
        return True

    def __init__(self, metric='entropy'):
        """Class constructor

        :param metric: Metric used for split evaluation (gini or entropy), if
        you want to specify another function do so at your own risk.
        """
        self.tree = None
        if metric == 'gini':
            self.metric = gini
        elif metric == 'entropy':
            self.metric = entropy
        else:
            print(
                'Unknown metric value (neither gini nor entropy), \
                    use at your own risk!',
                file=sys.stderr)
            self.metric = self.metric

    def mutual_information(self, y, x, epsilon=1e-7):
        """Calculates mutual information between x and labels y.

        :param y: Pandas or numpy column with labels
        :param x: Column with corresponding feature values
        :param epsilon: Small constant to avoid division by zero

        """
        values, frequencies = np.unique(x, return_counts=True)
        frequencies = frequencies / (len(x) + epsilon)
        return self.metric(y) - np.sum([
            probability * self.metric(y[x == value])
            for probability, value in zip(frequencies, values)
        ])

    def fit(self, X, y):
        """Fits classifier accordingly to dataset.

        :param X: Dataset with feature columns (Pandas or numpy arrays)
        :param y: Dataset with respective labels (Pandas/numpy column)

        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(X, pd.DataFrame) else y

        labels, frequencies = np.unique(y, return_counts=True)
        most_frequent_label = self._most_frequent_label(labels, frequencies)

        self.tree = [self._create_tree(X, y), most_frequent_label]

    def __repr__(self):
        """Returns pretty-printed representation of contained dictionary.

        :returns: string represting tree
        """
        return pprint.pformat(self.tree, indent=1)

    def pprint(self):
        """Print more human readable format of tree to stdout.

        Format for roots:
            Attribute number: attribute id
            Attribute value: value associated with attribute id
            Most frequent class: Class occuring most often in the subtree
        Format for leaves:
            Leaf with class: class associated with leaf


        """

        def recursive_print(subtree, indent):
            """Implementation detail.

            Recursively prints contained tree

            :param subtree: current subtree [nested list-dictionary]
            :param indent: indentation for current level [int]
            """
            if isinstance(subtree[0], dict):
                for attribute, subtree in subtree[0].items():
                    print('\t' * indent, 'Attribute number: {}'.format(
                        attribute[0]))
                    print('\t' * indent, 'Attribute value: {}'.format(
                        attribute[1]))
                    print('\t' * indent, 'Most frequent class: {}'.format(
                        subtree[1]))
                    recursive_print(subtree, indent + 1)
            else:
                print('\t' * indent, 'Leaf with class: {}'.format(subtree[1]))

        if self._is_tree_valid():
            recursive_print(self.tree, indent=0)

    def rules_count(self):
        """Calculates number of rules learned by classifier

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

        if self._is_tree_valid():
            return recursive_count(self.tree)
        return 0

    def prune(self, validation_X, validation_y, epsilon=0):
        """Post prunes trained tree.

        Performs post-prunning on created classifier using validation datasets.
        If accuracy on validation set increased, node is removed.
        Removes nodes in top down fashion (faster execution)

        :param validation_X: Pandas or numpy dataset with features
        :param validation_y: Pandas or numpy column with respective labels
        :param epsilon: If post_prunning_accuracy > pre_prunning_accuracy -
        epsilon, remove node. Factor for accuracy increase (can be negative)

        """
        # TRANSFORM PANDAS TO NUMPY IF PANDAS PROVIDED AS AN ARGUMENT
        validation_y = validation_y.values if isinstance(
            validation_y, pd.DataFrame) else validation_y
        validation_X = validation_X.values if isinstance(
            validation_X, pd.DataFrame) else validation_X

        # PREPARE TREE
        roots = [self.tree[0][key] for key in self.tree[0]]
        for root in roots:
            if isinstance(root[0], dict):
                # COMPARE ACCURACIES
                base_accuracy = accuracy_score(
                    self.predict(validation_X), validation_y)
                root[0], backup = root[1], copy.deepcopy(root[0])
                prunned_accuracy = accuracy_score(
                    self.predict(validation_X), validation_y)
                # REVERSE PRUNE IF NO ACCURACY INCREASE
                if prunned_accuracy < base_accuracy - epsilon:
                    root[0] = backup
                    roots.extend([root[0][key] for key in root[0]])

    def predict(self, X):
        """Predicts label for Pandas or Numpy matrix.

        :param X: Dataset with feature columns (Pandas or numpy arrays)
        :returns: Numpy array with predictions

        """

        def _recursive_predict(keys, root):
            """Implementation detail.

            _recursive_predict recursively predict label for classifier.

            :param keys:
            :param root: root of classification tree
            :returns: label for a given example

            """
            # IF WE ARE NOT IN LEAVE, GO DEEPER IN TREE
            if isinstance(root[0], dict):
                attribute_value = list(root[0].keys())[0][0]

                data_value = keys[attribute_value]
                next_root = root[0].get((attribute_value, data_value))

                # IF NO RULE FOUND RETURN MOST FREQUENT CLASS IN SUB-DATASET
                if next_root is None:
                    return root[1]

                return _recursive_predict(keys, next_root)
            return root[0]

        X = X.values if isinstance(X, pd.DataFrame) else X
        if self._is_tree_valid():
            return np.array(
                [_recursive_predict(dict(enumerate(x)), self.tree) for x in X])
        return None
