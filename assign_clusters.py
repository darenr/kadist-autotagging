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
import random
import functools

from metrics import hamming_score

from people import people

from common_functions import find_clusters, _mk_synset, wup, preprocess_clusters

def preprocess_trials(trials):
    #
    # convert works user_tags to synsets
    #

    for work in trials:
        work['user_tags'] = [_mk_synset(x) for x in work['user_tags']]
        work['machine_tags'] = [_mk_synset(x) for x in work['machine_tags']]

    return trials


def tags_to_clusters(clusters, trials, t, similarity):
    """using the set of `clusters` find cluster from tags"""
    results = trials
    for work in tqdm(results):

        if 'user_tags' in work:
            work['machine_clusters_from_user_tags'] = find_clusters(clusters, work['user_tags'], t, similarity)

        if 'machine_tags' in work:
            work['machine_clusters_from_machine_tags'] = find_clusters(clusters, work['machine_tags'], t, similarity)


def word_count(doc):
    return len(TextBlob(doc).words)

if __name__ == '__main__':

    similarity = wup
    T = 0.76
    results_prefix = 'all_kadist_works'
    file_trials = 'data/all_annotated_trials.json'  # annotated with tag_kadist_docs.py
    compute_person_metrics = False
    abbreviated = False
    abbreviated_size = 100

    cluster_types = ['clusters', 'superclusters']

    random.seed(42)

    print(' *', 'using WordNet version:', wordnet.get_version())
    print(' *', 'using', 'similarity fn', similarity.__name__, 'T', T)
    print(' *', 'compute_person_metrics', compute_person_metrics)
    print(' *', 'results_prefix', results_prefix)
    if abbreviated:
        cluster_types = ['clusters']
        print(' *', 'abbreviated mode, limiting to {} (stable sample) trials'.format(abbreviated_size))

    for cluster_type in cluster_types:
        file_clusters = f'data/{cluster_type}.json'
        with codecs.open(file_clusters, 'rb', 'utf-8') as f_clusters:
            clusters = preprocess_clusters(json.loads(f_clusters.read()))

            with codecs.open(file_trials, 'rb', 'utf-8') as f_trials:

                trials = preprocess_trials(json.loads(f_trials.read()))

                if abbreviated:
                    trials = random.sample(trials, abbreviated_size)

                tags_to_clusters(clusters, trials, t=T, similarity=similarity)

                data_df = []
                for work in trials:

                    hits = len(set(work['machine_clusters_from_user_tags']).intersection(set(work['machine_clusters_from_machine_tags'])))

                    data_df.append([
                        work['region'],
                        work['artist_name'],
                        work['title'],
                        work['permalink'],
                        ','.join([_mk_wv_word(x) for x in work['user_tags']]),
                        ','.join(work['machine_clusters_from_user_tags']),
                        ','.join(work['machine_clusters_from_machine_tags']),
                        hits,
                        word_count(work['description'])
                    ])

                df = pd.DataFrame(data_df, columns=["region", "artist_name", "title", "permalink", "user_tags",
                                                    "machine_clusters_from_user_tags", "machine_clusters_from_machine_tags", "hits", "word_count"])

                output_filename = 'results/%s_%s_results_%s_%.2f.csv' % (results_prefix, cluster_type, similarity.__name__, T)
                df.to_csv(output_filename, index=False)
                print(' *', 'written file results to', output_filename)
                s = df[pd.notnull(df['machine_clusters_from_machine_tags'])].hits
                print(' *', cluster_type, 'hits histogram:', df[pd.notnull(df['machine_clusters_from_machine_tags'])].hits.value_counts().to_json())
