#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import codecs
import sys
import pandas as pd
import numpy as np
from nltk.corpus import wordnet

from tqdm import tqdm

#
# helper functions to turn words into synsets and vice versa
#


def _mk_synset(w):
    #
    # turn cat.n.01 into the Synset object form
    #
    word = w.strip().replace(' ', '_')
    if '.' in word:
        try:
            return wordnet.synset(word)
        except Exception as ex:
            print(' * Error, invalid synset name', w)

    else:
        print(' * Error, invalid synset name', w, 'skipping...')
        return None


def _mk_wv_word(s):
    #
    # turn wordnet Synset into regular word form
    #   e.g. cat.n.01 -> 'cat'
    #   e.g. free_trade.n.01 -> free-trade
    return s.lemmas()[0].name()

#
# two distance methods, wup and path
#

def wup(w1, w2, t):
    distance = w1.wup_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0


def path(w1, w2, t):
    distance = w1.path_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0

def preprocess_clusters(clusters):
    d = {}

    for k in clusters.keys():
        d[k] = [_mk_synset(x) for x in clusters[k]]

    return d

def find_clusters(user_tags):
    pass

def tag_trials(clusters, trials):
    for work in tqdm(trials):
        #print(' *', 'tagging', '[%s]' % (work['artist_name']))
        auto_tags = find_clusters(work['user_tags'])

if __name__ == '__main__':

    print(' *', 'using WordNet version:', wordnet.get_version())
    file_clusters = 'data/clusters.json'
    with codecs.open(file_clusters, 'rb', 'utf-8') as f_clusters:
        clusters = preprocess_clusters(json.loads(f_clusters.read()))

        file_trials = 'data/trials.json'
        with codecs.open(file_trials, 'rb', 'utf-8') as f_trials:
            trials = json.loads(f_trials.read())

            tag_trials(clusters, trials)
