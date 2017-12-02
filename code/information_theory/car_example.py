#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""example.

Example usage of functionalities provided in:     data_processing
decision_tree

"""

import pandas as pd

from data_processing import convert_to_numerical, split_data
from decision_tree import DecisionTree, accuracy_score

# READ DATA AND NAME COLUMNS
df = pd.read_csv('./car.data', sep=',', header=None)
df.columns = [
    'buying', 'maintenance', 'doors', 'people', 'lug_boot', 'safety', 'class'
]

# CONVERT STRING VALUES TO THEIR NUMERICAL COUNTERPARTS
convert_to_numerical(
    df,
    columns=[
        'buying', 'maintenance', 'doors', 'people', 'lug_boot', 'safety',
        'class'
    ],
    inplace=True)

# SPLIT DATASET INTO TRAINING, VALIDATION, TESTING
training, validation, test = split_data(df, inplace=True)

# CREATE CLASSIFIER AND FIT IT TO TRAINING DATA
training_X = training.iloc[:, :-1]
training_y = training.iloc[:, -1]
clf = DecisionTree()
clf.fit(training_X, training_y)

# SPLIT VALIDATION DATASETS INTO X AND y
validation_X = validation.iloc[:, :-1]
validation_y = validation.iloc[:, -1]

# PRINT METRICS FOR PRUNNED AND UNPRUNNED DECISION TREE
predictions_validation = clf.predict(validation_X)
print("Validation set accuracy before prunning: ",
      accuracy_score(predictions_validation, validation_y))
print("Number of classification rules: ", clf.rules_count())
test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
predictions_test = clf.predict(test_X)
print("Test set accuracy for unprunned tree: ",
      accuracy_score(predictions_test, test_y))

print("----------------------------------------------------------------------")
clf.prune(validation_X, validation_y)
print("----------------------------------------------------------------------")

predictions_validation = clf.predict(validation_X)
print("Validation set accuracy after prunning: ",
      accuracy_score(predictions_validation, validation_y))
print("Number of classification rules after prunning: ", clf.rules_count())
predictions_test = clf.predict(test_X)
print("Test set accuracy for prunned tree: ",
      accuracy_score(predictions_test, test_y))
