#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import codecs
import glob

from tqdm import tqdm

from textblob import TextBlob
from textblob.wordnet import VERB, ADJ, NOUN
from textblob import Word

from rake_nltk import Rake

from tqdm import tqdm

if sys.version_info[0] >= 3:
    unicode = str


class RAKEDocumentTagger():

    def __init__(self, stopword_folder='resources'):
        """Initialize the DocumentTagger, pass in an array of stop wordslanguages"""
        print("RAKEDocumentTagger")

        self.stopword_folder = stopword_folder
        self.stopwords = []
        self.docs = []

        stopword_languages = glob.glob('{}/*_stopwords.txt'.format(stopword_folder))

        for language_file in stopword_languages:
            self._load_stop_words(language_file)

    def load_string_docs(self, docs):
        self.docs.extend([(i, doc) for i, doc in enumerate(docs)])
        print('  *', 'loaded {} docs'.format(len(self.docs)))
        return self

    def load_docs(self, folder):
        """Load a set of document from the `folder` with .txt extensions"""

        if not os.path.isdir(folder):
            raise ValueError("{} does not exist".format(folder))

        for (dirpath, dirnames, filenames) in os.walk(folder):
            filenames.sort()
            for filename in filenames:
                if filename.endswith('.txt'):
                    with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                        self.docs.append((filename, f.read()))

        return self

    def _load_stop_words(self, file_path):
        """load language/domain stop words"""
        if os.path.isfile(file_path):
            print('  *', 'loading stopwords from: [{}]'.format(file_path))
            with codecs.open(file_path, 'r', 'utf-8') as f:
                stopwords = f.readlines()
                self.stopwords.extend([unicode(x).strip().replace(' ', '_').lower() for x in stopwords])

    def _contains_number(self, s):
        """returns True iff string:s contains a digit"""
        return any(i.isdigit() for i in s)

    def convert_to_wordnet_form(self, word_or_phrase, min_word_length=3, pos=None):
        if len(word_or_phrase) >= min_word_length:
            if not word_or_phrase.lower() in self.stopwords and not self._contains_number(word_or_phrase):
                # try both underscore and then hyphenated
                preped_for_wn_underscore = '_'.join(word_or_phrase.lower().split()).strip()
                preped_for_wn_hyphen = '-'.join(word_or_phrase.lower().split()).strip()
                candidates = [w.name() for w in Word(preped_for_wn_underscore).get_synsets(pos=pos)]
                candidates.extend([w.name() for w in Word(preped_for_wn_hyphen).get_synsets(pos=pos)])
                return list(set(candidates))

        return []

    def process_documents(self, lang='en', top_n=10, threshold=0.74):
        result = {}

        for name, text in tqdm(self.docs):

            r = Rake(min_length=1, max_length=3, stopwords=self.stopwords)
            r.extract_keywords_from_text(text)

            result[name] = []

            for (score, keyphrase) in r.get_ranked_phrases_with_scores():
                wn_form = self.convert_to_wordnet_form(keyphrase)
                if wn_form:
                    print('keyphrase: [{}], wn_form: [{}]'.format(keyphrase, wn_form[0]))
                    result[name].append((wn_form[0], score))

        return result

    @staticmethod
    def pprint_keywords(r):
        """print out the document->keywords"""
        for doc in r.keys():
            print(u'[{}]: {}'.format(doc, ', '.join([u"{}/{}".format(t[0], t[1]) for t in r[doc]])))
            print('-' * 80)


if __name__ == '__main__':

    dt = RAKEDocumentTagger()

    dt.load_docs('docs')

    result = dt.process_documents()

    dt.pprint_keywords(result)
