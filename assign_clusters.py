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
    #
    # convert cluster tags to synsets
    #

    return {k: [_mk_synset(x) for x in clusters[k]] for k in clusters}


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

    print(' *', 'using WordNet version:', wordnet.get_version())
    file_clusters = 'data/clusters.json'
    with codecs.open(file_clusters, 'rb', 'utf-8') as f_clusters:
        clusters = preprocess_clusters(json.loads(f_clusters.read()))

        file_trials = 'data/trials.json'
        with codecs.open(file_trials, 'rb', 'utf-8') as f_trials:

            similarity = wup
            T = 0.8

            trials = preprocess_trials(json.loads(f_trials.read()))
            tag_trials(clusters, trials, t=T, similarity=similarity)

            data_df = []
            total_hits = 0
            for work in trials:
                hits = set(work['human_clusters']).intersection(set(work['machine_clusters']))
                total_hits += len(hits)
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

            df = pd.DataFrame(data_df, columns=["metric", "human_assessment_type", "human_clusters", \
                "machine_clusters", "hits", "artist_name", "title", "permalink"])
            print(T, similarity.__name__, total_hits)
            output_filename = 'results_%s_%.1f.csv' % (similarity.__name__, T)
            df.to_csv(output_filename, index=False)
            print(' *', 'written results to', output_filename)
