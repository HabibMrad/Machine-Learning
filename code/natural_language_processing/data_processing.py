#!/usr/bin/env python
# coding=utf-8
"""Data processing utilities.

This module allows you to:
1. Parse xml data into Pandas dataframe (read_xml)
2. Create datasets generator from .tar.gz compressed file (with chosen method
of parsing, e.g. read_xml or other Pandas provided utility function)
3. Safely create datasets, if it's not found on your hard drive, it shall
download it using provided url
4. Group datasets into efficient hierarchical structure (problem-specific
functionality). In this case category->type of review (positive/negative)->data
about the review

"""

import pathlib
import re
import tarfile
from functools import reduce

# fault tolerant XML parsing
import lxml.etree as ET
import nltk
import pandas as pd
import requests
import scipy
from sklearn.feature_extraction.text import HashingVectorizer


def read_xml(data: bytes, *, add_header=False):
    """read_xml parses xml into memory-efficient pandas dataframe.

    :param data: Byte type data, see usage in demo, create_datasets
    :type data: bytes
    :param add_header: Adds header (if the file is not xml conformant)
    :returns: pandas DataFrame consisting of xml data

    """
    parser = ET.XMLParser(recover=True)

    processed_data = data
    # dirty workaround - adds <workaround_root> root clause to the beggining of file
    # to make the xml valid for processing. Closing </workaround_root> appended as well
    if add_header:
        processed_data = "<workaround_root>\n{}\n</workaround_root>".format(
            data)

    reviews = ET.fromstring(processed_data, parser=parser)
    all_records = []
    for review in reviews:
        record = {}
        for category in review:
            record[category.tag] = category.text.strip(r'\n')
        all_records.append(record)
    return pd.DataFrame(all_records)


def create_datasets(file_handle, function):
    """create_datasets yields datasets from tar files.

    :param file_handle: file handle consisting of tar localitzation
    :param function: function to process each folder
    :yields: pandas dataset named exactly like it's localization in tar, e.g.
    dvd/review/negative.review

    """
    with tarfile.open(file_handle) as tar:
        for member_file in tar:
            pd_dataset = function(
                tar.extractfile(member_file).read(), add_header=True)
            pd_dataset.name = member_file.name
            yield pd_dataset


def safely_create_datasets(dataset_name, dataset_path, dataset_url, function):
    """safely_create_datasets Creates datasets from path if file exists,
    otherwise downloads it from the dataset_url. Returns dictionaries
    preserving data structure (nested disctionaries of pandas dataframes)

    For usage example check demo function

    :param function: Function to process each folder (see create_datasets)
    :param dataset_name: name of dataset
    :param dataset_path: path to dataset
    :param dataset_url: url to dataset
    :returns: generator of dataframes
    :raises: OSError if file was moved during check of it's existence

    """
    file_handle = pathlib.Path('{}{}'.format(dataset_path, dataset_name))
    if file_handle.is_file():
        try:
            generator = create_datasets(file_handle, function)
            return generator
        except OSError:
            raise
    else:
        generator = create_datasets(requests.get(dataset_url), function)
        return generator


def group_datasets(datasets):
    """group_datasets groups datasets into categories (can use generator or
    normal datasets [generators advised, see create_datasets]).

    This function is problem-specific, for overall usage it should be
    appropriately adjusted (contact author for informations)

    :param datasets: list of datasets to group
    :returns: Dictionary of pandas dataframes

    """
    grouped_datasets = {}
    for dataset in datasets:
        _, category, review = dataset.name.split(r"/")
        if category not in grouped_datasets:
            grouped_datasets[category] = {}
        grouped_datasets[category][review] = dataset
    return grouped_datasets


def unify_datasets(grouped_datasets):
    """unify_datasets Unifies dictionaries of pandas Dataframes into one data
    frame, where each category is a separate column.

    :param grouped_datasets: dictionary of grouped datasets (dictionary of
    pandas dataframes)
    :returns: One unified dataframed preserving all information

    """
    frames = []
    for category_name, category in grouped_datasets.items():
        for review_name, review in category.items():
            df = pd.DataFrame(review)
            df['category'] = category_name
            df['review'] = review_name
            frames.append(df)
    return pd.concat(frames)


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
            enumerate(dataframe[column].cat.categories, start=1))
        # WORKAROUND IF YOU WANT TO WORK WITH SPARSE MATRICES AFTERWARDS
        dataframe[column] = dataframe[column].cat.codes + 1
    return dataframe, codes


class StemmingHashingVectorizer(HashingVectorizer):
    """StemmingHashingVectorizer.

    Extension of sklearn's HashingVectorizer (uses additional stemming)
    All functionalities of normal HashingVectorizer are otherwise the
    same

    """

    def __init__(self, *, stemmer=nltk.stem.PorterStemmer(), **kwargs):
        """__init__.

        :param stemmer: Stemming object used for word stemming (default:
        nltk.stem.PorterStemmer())
        :param **kwargs: keyword arguments for base sklearn's HashingVectorizer

        """
        super().__init__(**kwargs)
        self._stemmer = stemmer

    def build_analyzer(self):
        """build_analyzer.

        Overloaded function for build_analyzer (additionally stems the
        words)

        """

        analyzer = super().build_analyzer()
        return lambda doc: ([self._stemmer.stem(word)
                             for word in analyzer(doc)])


class LemmatizingHashingVectorizer(HashingVectorizer):
    """LemmatizingHashingVectorizer.

    Extension of sklearn's HashingVectorizer (uses additional
    lemmatizing) All functionalities of normal HashingVectorizer are
    otherwise the same.

    """

    def __init__(self,
                 *,
                 lemmatizer=nltk.stem.wordnet.WordNetLemmatizer(),
                 **kwargs):
        """__init__

        :param lemmatizer: Lemmatizing object used for word lemmatizing
        (default: nltk.stem.wordnet.WordNetLemmatizer())
        :param **kwargs: keyword arguments for base sklearn's HashingVectorizer

        """
        super().__init__(**kwargs)
        self._lemmatizer = lemmatizer

    def build_analyzer(self):
        """build_analyzer.

        Overloaded function for build_analyzer (additionally lemmatizes
        the words)

        """
        analyzer = super().build_analyzer()
        return lambda doc: ([self._lemmatizer.lemmatize(word)
                             for word in analyzer(doc)])


def vectorize_columns(df,
                      *,
                      columns,
                      vectorizer=StemmingHashingVectorizer(
                          stop_words='english', norm=None)):
    """vectorize_columns.

    Creates bag of words representation of given columns in pandas dataframe.
    Returns them as scipy's sparse matrix in COO format

    :param stop_words:
    :param norm:

    """

    # JOINS STRING COLUMNS INTO A SINGLE STRING COLUMN(SMALLER FEATURES VECTOR)
    dataframe = reduce(lambda first, second:
                       first.astype('U').str.cat(second.astype('U'), sep=' '),
                       [df[column] for column in columns])
    return vectorizer.transform(dataframe.iloc[:50])


# HELPFUL SOMETIMES ZERO, EPSILON FOR UNKNOWN VALUE
def reviews_to_numeric(df, columns, inplace=False):
    """reviews_to_numeric.

    Converts to numerical value how helpful was a certain review.
    E.g. 4 of 11 is transformed into 4/11 and returned as floating point value

    :param df: Pandas dataframe containg column to be transformed
    :param columns: Python's list: which review columns should be transformed
    :param inplace: Transform input dataframe (in-place True) or return new
    dataframe

    """
    integers_pattern = re.compile(r"[+-]?(?<!\.)\b[0-9]+\b(?!\.[0-9])")

    def create_factorials(found_numbers):
        if found_numbers:
            return int(found_numbers[0]) / int(found_numbers[1])
        return 0

    dataframe = pd.DataFrame(df) if inplace else df
    for column in columns:
        dataframe[column] = df.apply(
            lambda row: create_factorials(
                integers_pattern.findall(row[column])),
            axis=1)
    return dataframe


def demo():
    """demo.

    Demonstration of module possibilities.

    Uses every function from the module and prints object received.

    """
    dataset_name = 'domain_sentiment_data.tar.gz'
    dataset_path = './data/'
    dataset_url = 'http://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz'

    dataset_generator = safely_create_datasets(dataset_name, dataset_path,
                                               dataset_url, read_xml)
    grouped_datasets = group_datasets(dataset_generator)
    df = unify_datasets(grouped_datasets)
    text_features = vectorize_columns(
        df, columns=['title', 'review_text', 'product_name'])
    # REVIEWER LOCATION TOO HARD TO PARSE AND IT'S INPUT SHOULD BE SMALL
    df.drop(
        [
            'asin', 'date', 'reviewer', 'unique_id', 'product_type', 'title',
            'review_text', 'product_name', 'reviewer_location'
        ],
        axis=1,
        inplace=True)
    _, codes = convert_to_numerical(
        df, columns=['category', 'review'], inplace=True)
    df['rating'] = pd.to_numeric(df['rating'], errors='ignore')
    reviews_to_numeric(df, ['helpful'], inplace=True)

    # CONCATENATE SCIPY'S ARRAYS
    # LEAVES CATEGORY AS IT'S VALUE IS ZERO
    df_sparse_array = scipy.sparse.csr_matrix(df[:10].values)
    df_sparse_array = scipy.sparse.hstack([text_features, df_sparse_array])
    print(df_sparse_array.shape)


if __name__ == '__main__':
    demo()
