#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import os
import sys

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def get_stop_words(stop_file_path="resources/stopwords.txt"):
    with codecs.open(stop_file_path, 'r', 'utf-8') as f:
        stopwords = f.readlines()
        return set(m.strip().lower() for m in stopwords)


def features_to_doc(features):
    """convert list of features into a doc"""
    return ' '.join(features)


def doc_to_features(doc):
    """Simple featurizer is to use all the words"""
    tb = TextBlob(doc)
    return [x.lower() for x in tb.words if x.lower() not in stopwords]


def process_documents(docs):
    """The docs are tuples: (name, [features]), features is a list of 'words'"""
    cv = CountVectorizer(max_df=0.85, max_features=1000)
    word_count_vector = cv.fit_transform([features_to_doc(features) for name, features in docs])

    feature_names = cv.get_feature_names()

    # print(list(cv.get_feature_names()))

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    for name, features in docs:
        doc = features_to_doc(features)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = _sort_coo(tf_idf_vector.tocoo())
        keyword_scores = extract_topn_from_vector(feature_names, sorted_items, 10)
        print(name, keyword_scores)


def _sort_coo(coo_matrix):
    """A sparse matrix in COOrdinate format"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=15):
    """get the feature names and tf-idf score of top n items"""

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items[:topn]:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    return list(zip(feature_vals, score_vals))


def load_docs(folder='docs'):
    docs = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                    features = doc_to_features(f.read())
                    docs.append((filename, features))

    return docs


if __name__ == '__main__':
    stopwords = get_stop_words()
    stopwords.update(['contemporary'])
    docs = load_docs()

    process_documents(docs)
