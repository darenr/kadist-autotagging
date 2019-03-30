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


class DocumentTagger():


    def __init__(self, stopword_languages='english', stopword_folder='resources'):
        """Initialize the DocumentTagger, pass in an array of stop wordslanguages"""
        self.stopword_folder = stopword_folder
        stopwords = []
        if not isinstance(stopword_languages, (list, tuple)):
            stopword_languages = [stopword_languages]

        for language in stopword_languages:
            stopwords.extend(self._load_stop_words(language))

        self.stopwords = frozenset(stopwords)
        self.docs = []

    def _load_stop_words(self, language):
        """stopwords can be just words or regex patterns,
        eg artist.* would match artist, artists, artistic"""
        file_path = '{}/{}.txt'.format(self.stopword_folder, language)
        if os.path.isfile(file_path):
            with codecs.open(file_path, 'r', 'utf-8') as f:
                stopwords = f.readlines()
                return set(m.strip().lower() for m in stopwords)
        return []


    def _features_to_doc(self, features):
        """convert list of features into a doc"""
        return ' '.join(features)


    def _doc_to_features(self, doc, min_word_length=2):
        """Simple featurizer is to use all the words"""
        tb = TextBlob(doc)
        pat = re.compile('|'.join([u"^{}$".format(x) for x in self.stopwords]), re.I)

        clean_features = [x.lower() for x in tb.words \
            if len(x) > min_word_length \
                and x[0] not in string.ascii_uppercase
                and not pat.match(x)]

        print(clean_features)
        sys.exit(0)
        clean_features.extend([x.strip().replace(' ', '_') for x in tb.noun_phrases])
        return clean_features


    def pprint_keywords(self, r):
        """print out the document->keywords"""
        for doc in r.keys():
            print(u'[{}]: {}'.format(doc, ', '.join([u"{}/{}".format(t[0], t[1]) for t in r[doc]])))


    def process_documents(self, docs, vocab_size=1000):
        """The docs are tuples: (name, [features]), features is a list of 'words'"""

        cv = CountVectorizer(
            max_df=0.85,
            stop_words=None,
            strip_accents='unicode',
            max_features=vocab_size)
        word_count_vector = cv.fit_transform([_features_to_doc(features) for name, features in docs])

        feature_names = cv.get_feature_names()

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        result = {}
        for name, features in docs:
            doc = self._features_to_doc(features)
            tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
            sorted_items = self._sort_coo(tf_idf_vector.tocoo())
            keyword_scores = self._extract_topn_from_vector(feature_names, sorted_items, 10)
            result[name] = keyword_scores

        return result


    def _sort_coo(self, coo_matrix):
        """A sparse matrix in COOrdinate format"""
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=15):
        """get the feature names and tf-idf score of top n items"""

        score_vals = []
        feature_vals = []

        for idx, score in sorted_items[:topn]:
            fname = feature_names[idx]
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        return list(zip(feature_vals, score_vals))


    def load_docs(self, folder):
        if not os.path.isdir(folder):
            raise ValueError("{} does not exist".format(folder))
        """Load a set of document from the `folder` with .txt extensions"""
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for filename in filenames:
                if filename.endswith('.txt'):
                    with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                        features = self._doc_to_features(f.read())
                        docs.append((filename, features))

if __name__ == '__main__':
    dt = DocumentTagger(['english', 'art'])

    docs = dt.load_docs('docs')

    result = dt.process_documents(docs, vocab_size=500)
    dt.pprint_keywords(result)
