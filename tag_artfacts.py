#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import os
import sys
import re
import json
import string

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def get_stop_words(language='english'):
    """stopwords can be just words or regex patterns,
    eg artist.* would match artist, artists, artistic"""
    file_path = 'resources/{}.txt'.format(language)
    if os.path.isfile(file_path):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            stopwords = f.readlines()
            return set(m.strip().lower() for m in stopwords)
    return []


def features_to_doc(features):
    """convert list of features into a doc"""
    return ' '.join(features)


def doc_to_features(doc, min_word_length=2):
    """Simple featurizer is to use all the words"""
    tb = TextBlob(doc)
    pat = re.compile('|'.join(stopwords), re.I)

    clean_features = [x.lower() for x in tb.words \
        if len(x) > min_word_length \
            and x[0] not in string.ascii_uppercase
            and not pat.match(x)]

    #clean_features.extend([x.strip().replace(' ', '_') for x in tb.noun_phrases])
    print([x.strip().replace(' ', '_') for x in tb.noun_phrases])

    return clean_features


def pprint_keywords(r):
    for doc in r.keys():
        print(u'[{}]: {}'.format(doc, ', '.join([u"{}/{}".format(t[0], t[1]) for t in r[doc]])))


def process_documents(docs, vocab_size=1000):
    """The docs are tuples: (name, [features]), features is a list of 'words'"""

    cv = CountVectorizer(
        max_df=0.85,
        stop_words=None,
        strip_accents='unicode',
        max_features=vocab_size)
    word_count_vector = cv.fit_transform([features_to_doc(features) for name, features in docs])

    feature_names = cv.get_feature_names()

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    result = {}
    for name, features in docs:
        doc = features_to_doc(features)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = _sort_coo(tf_idf_vector.tocoo())
        keyword_scores = extract_topn_from_vector(feature_names, sorted_items, 10)
        result[name] = keyword_scores

    return result


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
    stopwords = get_stop_words(language='english')
    stopwords.update(['paintings?', 'exhibition', 'projects', 'art', 'artist.*', 'years?'])
    docs = load_docs()

    result = process_documents(docs, vocab_size=500)
    pprint_keywords(result)
