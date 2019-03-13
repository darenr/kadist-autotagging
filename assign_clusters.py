#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import codecs
import sys
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import metrics

from nltk.corpus import wordnet
from collections import defaultdict
from tqdm import tqdm
import operator
from copy import deepcopy


#
# metrics
#

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None, label_encode=True):
    '''
    Compute the Hamming score (label-based accuracy) for multi-label predictions
    takes an array of array of strings
    '''

    if label_encode:
        arr = []
        for t in y_true + y_pred:
            arr.extend(t)

        e = preprocessing.LabelEncoder()
        e.fit(arr)

        _y_true = np.array([e.transform(x) for x in y_true])
        _y_pred = np.array([e.transform(x) for x in y_pred])
    else:
        _y_true = y_true
        _y_pred = y_pred

    acc_list = []
    for i in range(_y_true.shape[0]):
        set_true = set(np.where(_y_true[i])[0])
        set_pred = set(np.where(_y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    hamming_score = np.mean(acc_list)

    #subset_accuracy = metrics.accuracy_score(_y_true, _y_pred, normalize=True, sample_weight=None)
    subset_accuracy=0
    return (hamming_score, subset_accuracy)


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
            clusters_when_success = []
            y_true_tags = []
            y_pred_tags = []
            y_true_desc = []
            y_pred_desc = []
            for work in trials:

                if work['human_assessment_type'] == 'tags':
                    y_true_tags.append(work['human_clusters'])
                    y_pred_tags.append(work['machine_clusters'])
                else:
                    y_true_desc.append(work['human_clusters'])
                    y_pred_desc.append(work['machine_clusters'])

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
            output_filename = 'results_%s_%.1f.csv' % (similarity.__name__, T)
            df.to_csv(output_filename, index=False)
            print(' *', 'written results to', output_filename)

            #
            # standard multi-label metrics
            #

            print('[tags group] Hamming score (label-based accuracy): {0}'.format(*hamming_score(y_true_tags, y_pred_tags)))
            print('[desc group] Hamming score (label-based accuracy): {0}'.format(*hamming_score(y_true_desc, y_pred_desc)))
