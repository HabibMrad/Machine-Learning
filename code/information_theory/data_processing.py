#!/usr/bin/env python
# coding=utf-8
"""Data processing utilities.

This module allows you to:
1. Split datasets
2. Convert Pandas string column values to their discrete numerical counterparts

"""

import pandas as pd

# from decision_tree import DecisionTree, accuracy_score

__all__ = ['split_data', 'convert_to_numerical']


def split_data(df, training=5, validation=1, testing=1, inplace=False):
    """split_data.

    :param df: Pandas dataframe to split into training, testing and validation
    :param training: Proportion of testing dataset
    :param validation: Proportion of validation dataset
    :param testing: Proportion of testing dataset
    :param inplace: [Default: False] Shuffle pandas dataframe or make copy and
    shuffle
    :returns: List containg 3 Pandas Dataframes (training, validation, testing)

    """
    part = len(df) / (training + validation + testing)
    training = int(training * part)
    validation = training + int(validation * part)

    # COPY OR USE DATAFRAME
    dataframe = df if inplace else df.copy()
    # SHUFFLE DATA
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return dataframe[:training], \
        dataframe[training:validation],\
        dataframe[validation:]


def convert_to_numerical(df, columns, inplace=False):
    """convert_to_numerical.

    Converts values to their numerical categorical counterparts.

    Enumeration of objects starts from 1 (easier digestion by sparse matrices
    of pandas or scipy).

    :param df: unified pandas Dataframe object
    :param columns: df column's to be transformed
    :param inplace: perform transformation in-place or return dataframe
    (default behaviour, value=False)

    """
    dataframe = pd.DataFrame(df) if inplace else df
    codes = {}
    for column in columns:
        dataframe[column] = pd.Categorical(dataframe[column])
        codes[column] = dict(
            enumerate(dataframe[column].cat.categories), start=1)
        # WORKAROUND IF YOU WANT TO WORK WITH SPARSE MATRICES AFTERWARDS
        dataframe[column] = dataframe[column].cat.codes + 1
    return dataframe, codes
