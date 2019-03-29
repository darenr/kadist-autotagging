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
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


def process_documents(docs):

    cv = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=1000)
    word_count_vector = cv.fit_transform([x[1] for x in docs])

    feature_names=cv.get_feature_names()

    # print(list(cv.get_feature_names()))

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    for name, doc in docs:
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        print(name, '/'.join(keywords))

def featureize(doc):
    return [x.lower() for x in tb.words]


def sort_coo(coo_matrix):
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

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def load_docs(folder='docs'):
    docs = set([])
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                    doc = ' '.join(TextBlob(f.read()).words)
                    docs.add((filename, doc))

    return docs


if __name__ == '__main__':
    stopwords = get_stop_words()
    docs = load_docs()

    process_documents(docs)
