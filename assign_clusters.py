#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import codecs
import sys
import pandas as pd
import numpy as np

from nltk.corpus import wordnet
from collections import defaultdict
from tqdm import tqdm
import operator
from copy import deepcopy

from metrics import hamming_score

# begin word vectors
import os
import json
from gensim.models import KeyedVectors

glove_file = "glove.6B.300d_word2vec.txt"
model_file = os.environ['HOME'] + "/models/" + glove_file

glove_model = None

# end word vectors


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
            try:
                # try the first for the stem word
                print('trying alternatives to [%s]...' % (word))
                return wordnet.synsets(word.split('.')[0])[0]
            except Exception as ex:
                return None

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


def wv(w1, w2, t):
    global glove_model

    if not glove_model:
        #
        # lazy load the model
        #
        print(' *', 'loading model, please wait...')
        glove_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    word1 = _mk_wv_word(w1).lower().replace('_', '-')
    word2 = _mk_wv_word(w2).lower().replace('_', '-')
    if word1 in glove_model and word2 in glove_model:
        distance = glove_model.similarity(word1, word2)
        if distance > t:
            return distance
    else:
        print(' *', word1, word1 in glove_model, word2, word2 in glove_model)

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


def preprocess_trials(trials):
    #
    # convert works user_tags to synsets
    #

    for work in trials:
        work['user_tags'] = [_mk_synset(x) for x in work['user_tags']]

    return trials


def find_clusters(clusters, user_tags, t, similarity, top_n=3):
    scores = defaultdict(int)
    for cluster in clusters:
        for cluster_tag in clusters[cluster]:
            scores[cluster] += sum([similarity(cluster_tag, works_tag, t) for works_tag in user_tags])

    scores = {k: v for k, v in scores.items() if v}

    sorted_scores = sorted(scores.items(), reverse=True, key=operator.itemgetter(1))[:top_n]
    #print(["%s/%.2f" % (c, s) for c,s in sorted_scores])
    return [c for c, s in sorted_scores]


def tag_trials(clusters, trials, t, similarity):
    results = trials
    for work in tqdm(results):
        #print(' *', 'tagging', '[%s]' % (work['title']))
        work['machine_clusters'] = find_clusters(clusters, work['user_tags'], t, similarity)


if __name__ == '__main__':

    similarity = wup
    T = 0.75

    print(' *', 'using WordNet version:', wordnet.get_version())
    print(' *', 'using WordVector Glove Model:', glove_file)
    print(' *', 'using', 'similarity fn', similarity.__name__, 'T', T)

    file_clusters = 'data/clusters.json'
    with codecs.open(file_clusters, 'rb', 'utf-8') as f_clusters:
        clusters = preprocess_clusters(json.loads(f_clusters.read()))

        file_trials = 'data/trials.json'
        with codecs.open(file_trials, 'rb', 'utf-8') as f_trials:

            trials = preprocess_trials(json.loads(f_trials.read()))
            tag_trials(clusters, trials, t=T, similarity=similarity)

            data_df = []
            total_hits = 0
            clusters_when_success = []
            y_true = []
            y_pred = []
            for work in trials:
                y_true.append(work['human_clusters'])
                y_pred.append(work['machine_clusters'])

                hits = set(work['human_clusters']).intersection(set(work['machine_clusters']))
                total_hits += len(hits)

                if len(hits):
                    clusters_when_success.extend(work['human_clusters'])
                data_df.append([
                    similarity.__name__,
                    work['human_assessment_type'],
                    ','.join(work['human_clusters']),
                    ','.join(work['machine_clusters']),
                    len(hits),
                    work['artist_name'],
                    work['title'],
                    work['permalink']
                ])

            #
            # make a results dataframe for easy visualization
            #

            df = pd.DataFrame(data_df, columns=["metric", "human_assessment_type", "human_clusters",
                                                "machine_clusters", "hits", "artist_name", "title", "permalink"])

            print(T, similarity.__name__, total_hits)
            output_filename = 'results_%s_%.2f.csv' % (similarity.__name__, T)
            df.to_csv(output_filename, index=False)

            print(' *', 'written file results to', output_filename)

            #
            # standard multi-label metrics
            #

            print('[T={}], [sim={}], hamming score (label-based accuracy): {}'.format(T, similarity.__name__, hamming_score(y_true, y_pred)))
