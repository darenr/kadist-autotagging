#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import os
import re
import sys
import json
import string
import glob

from textblob import TextBlob
from textblob.wordnet import VERB, ADJ, NOUN
from textblob import Word

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from collections import Counter

if sys.version_info[0] >= 3:
    unicode = str


class TFIDFDocumentTagger():

    def __init__(self, stopword_folder='resources', vocab_size=1000):
        """Initialize the DocumentTagger, pass in an array of stop wordslanguages"""
        print("  *", "TFIDFDocumentTagger")

        self.stopword_folder = stopword_folder
        self.stopwords = []
        self.vocab_size = vocab_size

        stopword_languages = glob.glob('{}/*_stopwords.txt'.format(stopword_folder))

        for language_file in stopword_languages:
            self._load_stop_words(language_file)

        self.stopwords = frozenset(self.stopwords)

        self.docs = []

    def _find_nnp_runs(self, tags):
        # iterate over tags and return runs of NNP's (noun phrase detection)
        candidates = []
        accum = []
        for word, pos in tags:
            if pos == 'NNP':
                accum.append(word)
            else:
                accum.clear()

            if len(accum) >= 2:
                candidates.append(' '.join(accum))

        return candidates

    def _load_stop_words(self, file_path):
        """load language/domain stop words"""
        if os.path.isfile(file_path):
            print('  *', 'loading stopwords from: [{}]'.format(file_path))
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

    def convert_to_wordnet_form(self, word_or_phrase, min_word_length, pos=None):
        if len(word_or_phrase) >= min_word_length:
            if not word_or_phrase.lower() in self.stopwords and not self._contains_number(word_or_phrase):
                preped_for_wn = '_'.join(word_or_phrase.lower().split()).strip()
                return [w.name() for w in Word(preped_for_wn).get_synsets(pos=pos)]
        return None

    def _doc_to_features(self, raw_doc, include_noun_phrases=True, min_word_length=3):
        """featurizer based on part of speech"""

        # clean up the doc

        doc = re.sub(u"['’]s", ' ', raw_doc)
        doc = re.sub(u"[‘’“”–]", " ", doc)
        doc = re.sub(u"[.]{3}", " ", doc)
        doc = re.sub(u"\s+", " ", doc)

        tb = TextBlob(doc)

        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        #
        # extract candidates by POS type and convert to their wordnet form
        #
        candidate_features = []

        include_types = {'NNS': NOUN, 'NNP': NOUN, 'NN': NOUN, 'JJ': ADJ, 'VB': VERB, 'VBN': VERB}
        include_tags = include_types.keys()
        for x in tb.tags:
            if x[1] in include_tags:
                tb_type = include_types[x[1]]
                wn_opts = self.convert_to_wordnet_form(x[0], min_word_length, pos=tb_type)
                if wn_opts:
                    candidate_features.append(wn_opts[0])

        if include_noun_phrases:
            for np in self._find_nnp_runs(tb.tags):
                wn_opts = self.convert_to_wordnet_form(np, min_word_length, pos=NOUN)
                if wn_opts:
                    candidate_features.append(wn_opts[0])

        tokens = self._clean_tokens(candidate_features)

        return tokens

    @staticmethod
    def pprint_keywords(r):
        """print out the document->keywords"""
        for doc in r.keys():
            print(u'[{}]: {}'.format(doc, ', '.join([u"{}/{}".format(t[0], t[1]) for t in r[doc]])))
            print('-' * 80)

    def process_documents(self, topn=25):
        """The docs are tuples: (name, [features]), features is a list of 'words'"""

        cv = CountVectorizer(
            min_df=0.01,        # ignore terms that appear in less than x% of the documents
            max_df=0.80,        # ignore terms that appear in more than x% of the corpus
            stop_words=None,
            ngram_range=(1, 2),
            tokenizer=unicode.split,
            strip_accents='unicode',
            max_features=self.vocab_size)

        word_count_vector = cv.fit_transform([self._features_to_doc(features) for name, features in self.docs])

        feature_names = cv.get_feature_names()

        tfidf_transformer = TfidfTransformer(
            smooth_idf=True,
            use_idf=True,
            sublinear_tf=True)

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
            if score > 0.1:
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

    def load_string_docs(self, arr):
        """Load the documents from an array of strings"""

        for i, str in enumerate(arr):
            self.docs.append((i, self._doc_to_features(str)))

        return self


if __name__ == '__main__':

    dt = TFIDFDocumentTagger(vocab_size=1000)

    dt.load_docs('docs')

    result = dt.process_documents()

    DocumentTagger.pprint_keywords(result)
