#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import os
import sys
import re
import json
import string

from common_functions import find_clusters, _mk_synset, _word_to_synset, wup, preprocess_clusters
from tfidf_document_tagger import TFIDFDocumentTagger
from cortical_document_tagger import CorticalDocumentTagger
from rake_document_tagger import RAKEDocumentTagger
from text_rank_document_tagger import TextRankDocumentTagger

from nltk.metrics.scores import f_measure

"""

Load the MCA json corpus:

    - assign clusters for all the works with user_tags. The results are
      written to "results/mca_cluster_assignments.csv"
"""

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
from html_processor import normalize_text

random.seed(42)

def prep_tagged_works(source_trials):

    trials = []

    for trial in source_trials:
        if "description" in trial:
            if "user_tags" in trial and trial["user_tags"]:
                trial['user_tags'] = [x.strip().lower() for x in trial['user_tags'].split(',') if x]
                trial['user_tags_synsets'] = [_word_to_synset(x) for x in trial['user_tags'] if _word_to_synset(x)]
                trials.append(trial)

    print('  *', 'prep trials, number of trials: {}'.format(len(trials)))
    return trials


def tags_to_clusters(clusters, trials, t, similarity, tag_col_prefix, cluster_type, use_only_n_tags=99):
    """using the set of `clusters` find cluster from tags"""
    results = trials
    source_col_name = '{}_tags_synsets'.format(tag_col_prefix)
    target_col_name = '{}'.format(tag_col_prefix)
    for work in tqdm(results):
        if source_col_name in work:
            sorted_scores = find_clusters(clusters, work[source_col_name][:use_only_n_tags], t, similarity)
            work["{}_{}_{}".format(target_col_name, cluster_type, "formatted")] = ['{}/{}'.format(c, s) for c, s in sorted_scores]
            work["{}_{}_{}".format(target_col_name, cluster_type, "no_scores")] = [c for c, s in sorted_scores]
            work["{}_{}_{}".format(target_col_name, cluster_type, "sum_of_scores")] = sum([s for c, s in sorted_scores])


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

    return sorted_scores


def assign_clusters_to_works(trials):
    dest_file = "results/mca_assignments.csv"
    similarity = wup
    T = 0.76

    cluster_types = ['clusters']
    for (tag_col_prefix, use_only_n_tags) in [('user', 6), ('machine', 25)]:
        for cluster_type in cluster_types:
            print('  *', 'processing type: {} for {}'.format(tag_col_prefix, cluster_type))
            with codecs.open(f'data/{cluster_type}.json', 'rb', 'utf-8') as f_clusters:
                clusters = preprocess_clusters(json.loads(f_clusters.read()))
                tags_to_clusters(
                    clusters,
                    trials,
                    t=T,
                    similarity=similarity,
                    tag_col_prefix=tag_col_prefix,
                    cluster_type=cluster_type,
                    use_only_n_tags=use_only_n_tags)

    #
    # score results
    #
    for work in trials:
        for cluster_type in cluster_types:
            machine = "{}_{}_{}".format('machine', cluster_type, "no_scores")
            human = "{}_{}_{}".format('user', cluster_type, "no_scores")
            work["{}_fmeasure".format(cluster_type)] = f_measure(set(work[human]), set(work[machine]))

    df = pd.DataFrame(trials)
    df.drop(columns=['user_tags_synsets', 'machine_tags_synsets'], inplace=True)

    # move some columns to front
    cols = df.columns.tolist()

    for col in ['user_tags',  'machine_tags',  'title',  'description', 'artist_name']:
        cols.insert(0, cols.pop(cols.index(col)))
    df = df.reindex(columns=cols)

    df.to_csv(dest_file, index=False)
    print(' *', 'written file: {}'.format(dest_file))

    for j, cluster_type in enumerate(cluster_types):
        s = df['{}_fmeasure'.format(cluster_type)]
        print(cluster_type,
              'mean f-measure:',
              s.mean(),
              'hit percentage:',
              100 * s.where(s > 0).count() / len(s)
              )

    # generate best/worst top n for gsheet analysis
    # df_gsheet = df.dropna(subset=['clusters_fmeasure']) # drop any we don't tag
    # df_gsheet.sort_values(by=['clusters_fmeasure'])\
    #     .tail(25)\
    #     .to_csv("results/best_performing_kadist_assignments.csv", index=False)
    # df_gsheet.sort_values(by=['clusters_fmeasure'])\
    #     .head(25)\
    #     .to_csv("results/worst_performing_kadist_assignments.csv", index=False)


def tag_works_from_text(works, vocab_size=500):
    """assign machine_tags to all works with descriptions"""

    print('  *',  'tag_works_from_text (vocab_size: {})'.format(vocab_size))

    results = TFIDFDocumentTagger(vocab_size=vocab_size) \
        .load_string_docs([x['description'] for x in works]) \
        .process_documents()

    for doc_id, machine_tags in results.items():
        works[doc_id]['machine_tags'] = [x[0] for x in machine_tags]
        works[doc_id]['machine_tags_synsets'] = [_mk_synset(x) for x in works[doc_id]['machine_tags']]

if __name__ == "__main__":

    with codecs.open("data/mca_trials.json", "rb", "utf-8") as f:
        trials = prep_tagged_works(json.loads(f.read()))
        tag_works_from_text(trials, vocab_size=500)
        assign_clusters_to_works(trials)
