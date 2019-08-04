#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import codecs
import sys
import re
import pandas as pd
import numpy as np

from nltk.corpus import wordnet
from collections import defaultdict
from tqdm import tqdm
import operator
from copy import deepcopy
import random
import functools

from metrics import hamming_score

# begin word vectors
import os
import json
from gensim.models import KeyedVectors


from textblob import TextBlob

# end word vectors


#
# helper functions to turn words into synsets and vice versa
#

@functools.lru_cache(maxsize=10000)
def _synset_to_word(s):
    return s.lemmas()[0].name()

@functools.lru_cache(maxsize=10000)
def _mk_synset(w):
    #
    # (synset form) cat.n.01 into the Synset object form
    # (lemma form) syndicate.n.01.crime_syndicate
    #

    word = w.strip().replace(' ', '_')

    pat_regular_form = re.compile(r".*[.]\d{2}$")
    pat_regular_lemma_form = re.compile(r".*[.]\d{2}[.].+$")

    if pat_regular_form.match(word):
        try:
            return wordnet.synset(word)
        except Exception as ex:
            try:
                # try the first for the stem word
                return wordnet.synsets(word.split('.')[0])[0]
            except Exception as ex:
                return None

    elif pat_regular_lemma_form.match(word):
        try:
            return wordnet.lemma(word).synset()
        except Exception as ex:
            return None

    else:
        print(' * Error, invalid synset name: [{}] skipping'.format(w))
        return None

#
# two distance methods, wup and path
#

@functools.lru_cache(maxsize=1000000)
def wup(w1, w2, t):
    distance = w1.wup_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0


@functools.lru_cache(maxsize=1000000)
def path(w1, w2, t):
    distance = w1.path_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0



def preprocess_clusters(clusters):
    #
    # convert cluster tags to synsets
    #
    d = defaultdict(list)

    for k in clusters:
        for tag in clusters[k]:
            ss = _mk_synset(tag)
            if ss:
                d[k].append(ss)
            else:
                print('skipping tag: [%s] does not have a valid sysnset' % (tag))

        clusters[k] = frozenset(clusters[k])
    return dict(d)



def find_clusters(clusters, tags, t, similarity, top_n=3, debug=False):
    scores = defaultdict(int)
    for cluster in clusters:
        for cluster_tag in clusters[cluster]:
            if debug:
                for works_tag in tags:
                    sim = similarity(cluster_tag, works_tag, t)
                    print('DEBUG', cluster, sim, cluster_tag.name(), works_tag.name())

            scores[cluster] += sum([similarity(cluster_tag, works_tag, t) for works_tag in tags])

    scores = {k: v for k, v in scores.items() if v}

    sorted_scores = sorted(scores.items(), reverse=True, key=operator.itemgetter(1))[:top_n]
    if debug:
        print(["%s/%.2f" % (c, s) for c, s in sorted_scores])
    return [c for c, s in sorted_scores]
