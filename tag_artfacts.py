#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import codecs
import os
import sys
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def get_stop_words(stop_file_path="resources/stopwords.txt"):
    with codecs.open(stop_file_path, 'r', 'utf-8') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip().lower() for m in stopwords)
        return frozenset(stop_set)

def features_to_doc(features):
    return ' '.join(features)

def process_documents(docs):
    """The docs are tuples: (name, [features]), features is a list of 'words'"""
    cv = CountVectorizer(max_df=0.85, max_features=1000)
    word_count_vector = cv.fit_transform([features_to_doc(features) for name, features in docs])

    feature_names=cv.get_feature_names()

    # print(list(cv.get_feature_names()))

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    for name, features in docs:
        doc = features_to_doc(features)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        print(name, '/'.join(keywords))

def feature_extraction(doc):
    tb = TextBlob(doc)
    return [x.lower() for x in tb.words if x.lower() not in stopwords]

def sort_coo(coo_matrix):
    """A sparse matrix in COOrdinate format"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def load_docs(folder='docs'):
    docs = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                    doc = feature_extraction(f.read())
                    docs.append((filename, doc))

    return docs


if __name__ == '__main__':
    stopwords = get_stop_words()
    docs = load_docs()

    process_documents(docs)
