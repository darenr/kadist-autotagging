#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import os
import re
import sys
import json
import string

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if sys.version_info[0] >= 3:
    unicode = str


class DocumentTagger():

    def __init__(self, stopword_languages='english', stopword_folder='resources'):
        """Initialize the DocumentTagger, pass in an array of stop wordslanguages"""
        self.stopword_folder = stopword_folder
        self.stopwords = []
        if not isinstance(stopword_languages, (list, tuple)):
            stopword_languages = [stopword_languages]

        for language in stopword_languages:
            self._load_stop_words(language)

        self.stopwords = frozenset(self.stopwords)
        self.docs = []

    def _load_stop_words(self, language):
        """load language/domain stop words"""
        file_path = '{}/{}.txt'.format(self.stopword_folder, language)
        if os.path.isfile(file_path):
            with codecs.open(file_path, 'r', 'utf-8') as f:
                stopwords = f.readlines()
                self.stopwords.extend([unicode(x).strip().replace(' ', '_').lower()
                                       for x in stopwords])

    def _features_to_doc(self, features):
        """convert list of features into a doc"""
        return u' '.join(features)

    def _clean_tokens(self, token_sequence):
        """returns a new sequence with the input token sequence cleaned"""
        return [re.sub(r'\s+', '_', x).lower() for x in token_sequence]

    def _contains_number(self, s):
        """returns True iff string:s contains a digit"""
        return any(i.isdigit() for i in s)

    def _doc_to_features(self, raw_doc, include_noun_phrases=True, min_word_length=3):
        """Simple featurizer is to use all the words"""

        # clean up the doc
        doc = re.sub(u"['’]s", 's', raw_doc)
        doc = re.sub(u"[‘’–]", "", doc)

        tb = TextBlob(doc)

        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        candidate_features = [x[0] for x in tb.tags if x[1] in ['NNS', 'JJ', 'VBN', 'NN']]

        # if include_noun_phrases:
        #     candidate_features.extend(tb.noun_phrases)

        tokens = [token for token in candidate_features
                  if len(token) >= min_word_length
                    and not self._contains_number(token)
                  and not token.lower() in self.stopwords]

        return self._clean_tokens(tokens)

    @staticmethod
    def pprint_keywords(r):
        """print out the document->keywords"""
        for doc in r.keys():
            print(u'[{}]: {}'.format(doc, ', '.join([u"{}/{}".format(t[0], t[1]) for t in r[doc]])))
            print('-'*80)
            
    def process_documents(self, vocab_size=5000, topn=15):
        """The docs are tuples: (name, [features]), features is a list of 'words'"""

        cv = CountVectorizer(
            ngram_range=(1, 3),
            min_df=0.03,       # ignore terms that appear in less than x% of the documents
            max_df=0.80,       # ignore terms that appear in more than x% of the corpus
            stop_words=None,
            tokenizer=unicode.split,
            strip_accents='unicode',
            max_features=vocab_size)

        word_count_vector = cv.fit_transform([self._features_to_doc(features) for name, features in self.docs])

        feature_names = cv.get_feature_names()

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True, sublinear_tf=True)
        tfidf_transformer.fit(word_count_vector)

        result = {}
        for name, features in self.docs:
            doc = self._features_to_doc(features)
            tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
            sorted_items = self._sort_coo(tf_idf_vector.tocoo())
            keyword_scores = self._extract_topn_from_vector(feature_names, sorted_items, topn)
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
            if score> 0.1:
                score_vals.append(round(score, 3))
                feature_vals.append(feature_names[idx])

        return list(zip(feature_vals, score_vals))

    def load_docs(self, folder):
        """Load a set of document from the `folder` with .txt extensions"""

        if not os.path.isdir(folder):
            raise ValueError("{} does not exist".format(folder))

        for (dirpath, dirnames, filenames) in os.walk(folder):
            filenames.sort()
            for filename in filenames:
                if filename.endswith('.txt'):
                    with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                        self.docs.append((filename, self._doc_to_features(f.read())))
            return self


if __name__ == '__main__':
    dt = DocumentTagger(['english', 'art'])

    dt.load_docs('docs')

    result = dt.process_documents(vocab_size=500)

    DocumentTagger.pprint_keywords(result)
