#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""example.

Example usage of functionalities provided in:     data_processing
decision_tree

"""

import pandas as pd

from data_processing import convert_to_numerical, split_data
from decision_tree import DecisionTree, accuracy_score

df = pd.read_csv('./car.data', sep=',', header=None)
df.columns = [
    'buying', 'maintenance', 'doors', 'people', 'lug_boot', 'safety', 'class'
]
convert_to_numerical(
    df,
    columns=[
        'buying', 'maintenance', 'doors', 'people', 'lug_boot', 'safety',
        'class'
    ],
    inplace=True)

training, validation, test = split_data(df, inplace=True)
training_x = training.iloc[:, :-1]
training_y = training.iloc[:, -1]

clf = DecisionTree()
clf.fit(training_x, training_y)

true_validation = validation.iloc[:, -1].values
validation_set = validation.iloc[:, :-1]

predictions_validation = clf.predict(validation_set)
print("Accuracy before prunning: ",
      accuracy_score(predictions_validation, true_validation))
print("Number of classification rules: ", clf.rules_count())
clf.prune(validation_set, true_validation)
predictions_validation = clf.predict(validation_set)
print("Accuracy after prunning: ",
      accuracy_score(predictions_validation, true_validation))
print("Number of classification rules after prunning: ", clf.rules_count())
