#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""example.

Example usage of functionalities provided in: data_processing
decision_tree

"""

from sklearn.metrics import accuracy_score

from binary_decision_tree import DecisionTree
from data_processing import convert_to_numerical, get_dataset, split_data
from random_forests import RandomForest

df = get_dataset(
    './kr-vs-kp.data',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data'
)

df.columns = list(range(37))
convert_to_numerical(df, columns=list(range(37)), inplace=True)

training, validation, test = split_data(df, inplace=True)
training_X = training.iloc[:, :-1]
training_y = training.iloc[:, -1]

clf = RandomForest()
clf.fit(training_X, training_y)

validation_X = validation.iloc[:, :-1]
validation_y = validation.iloc[:, -1]

predictions_validation = clf.predict(validation_X)
print("Validation set accuracy: ",
      accuracy_score(predictions_validation, validation_y))
test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
predictions_test = clf.predict(test_X)
print("Test set accuracy: ", accuracy_score(predictions_test, test_y))
