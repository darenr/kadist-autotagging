#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import os
import sys
import re
import json
import string

from common_functions import find_clusters, _mk_synset, wup, preprocess_clusters
from tfidf_document_tagger import TFIDFDocumentTagger
from cortical_document_tagger import CorticalDocumentTagger
from rake_document_tagger import RAKEDocumentTagger

from nltk.metrics.scores import f_measure

"""

Load the Kadist json corpus:

    - assign both clusters and super clusters to
      all the works with user_tags. The results are
      written to "results/kadist_<cluster|supercluster>_assignments.csv"
    - generate a hierahical json ready for bubble/sunburt consumption,
      the "size" is set to the number of children from that level, the levels
      themselves being super-cluster >> cluster >> works.
      The results are written to "results/kadist_<cluster|supercluster>_hierachies.csv"

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


def prep_tagged_works(works):

    print('  *', 'prep_tagged_works, number of works: {}'.format(len(works)))

    trials = []

    for work in works:
        if "description" in work:
            if "user_tags" in work and work["user_tags"]:
                if "_thumbnails" in work and "medium" in work["_thumbnails"]:
                    artist_name = ", ".join(
                        [x["post_title"] for x in work["_artists"]]
                    )
                    permalink = work["permalink"]
                    title = work["title"].strip()
                    tags = work["user_tags"]
                    description = normalize_text(work["description"]).strip()
                    if "artist_description" in work and work["artist_description"]:
                        artist_description = normalize_text(work["artist_description"]).strip()
                    else:
                        artist_description = ""
                    thumbnail_url = work["_thumbnails"]["medium"]["url"]
                    region = (
                        work["_region"][0]
                        if "_region" in work and work["_region"]
                        else "Unspecified"
                    )
                    trials.append(
                        {
                            "artist_name": artist_name,
                            "title": title,
                            "description": description,
                            "artist_description": artist_description,
                            "region": region,
                            "user_tags": tags,
                            "user_tags_synsets": [_mk_synset(x) for x in tags],
                            "thumbnail": thumbnail_url,
                            "image_url": work['image_url'],
                            "permalink": permalink,
                            "doc": "{}. {}. {}.".format(description, artist_description, title)
                        }
                    )

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
    dest_file = "results/kadist_assignments.csv"
    cluster_types = ['superclusters', 'clusters']
    similarity = wup
    T = 0.76

    for (tag_col_prefix, use_only_n_tags) in [('user', 2), ('machine', 20)]:
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
            human   = "{}_{}_{}".format('user', cluster_type, "no_scores")
            work["{}_fmeasure".format(cluster_type)] = f_measure(set(work[human]), set(work[machine]))

    df = pd.DataFrame(trials)
    df.drop(columns=['user_tags_synsets', 'machine_tags_synsets'], inplace=True)

    # move some columns to front
    cols = df.columns.tolist()

    for col in ['permalink', 'thumbnail', 'image_url', 'user_tags',  'machine_tags', 'region', 'title', 'artist_description', 'description', 'artist_name']:
        cols.insert(0, cols.pop(cols.index(col)))
    df = df.reindex(columns=cols)

    df.to_csv(dest_file, index=False)
    print(' *', 'written file: {}'.format(dest_file))


def generate_cluster_hierachies():
    in_file = "results/kadist-cluster-assignments.csv"
    dest_file = "results/kadist-cluster-hierachies.csv"

    cluster_types = ['superclusters', 'clusters']

    for cluster_type in cluster_types:
        file_clusters = f'data/{cluster_type}.json'

    #
    #
    # TODO
    #
    #


def tag_works_from_text(works, vocab_size=1000):
    """assign machine_tags to all works with descriptions"""

    print('  *',  'tag_works_from_text (vocab_size: {})'.format(vocab_size))

    results = RAKEDocumentTagger() \
        .load_string_docs([x['doc'] for x in works]) \
        .process_documents()

    for doc_id, machine_tags in results.items():
        # print("doc_id: {}, machine_tags: {}".format(doc_id, machine_tags))
        # works[doc_id]['machine_tags'] = [x[0] for x in machine_tags if x[1] >= machine_tags[0][1] / 2.0]
        works[doc_id]['machine_tags'] = [x[0] for x in machine_tags]
        works[doc_id]['machine_tags_synsets'] = [_mk_synset(x) for x in works[doc_id]['machine_tags']]

if __name__ == "__main__":

    with codecs.open("data/kadist.json", "rb", "utf-8") as f:

        source = json.loads(f.read())

        works = prep_tagged_works(source)

        tag_works_from_text(works)

        # for i, x in enumerate(works):
        #     print('\n\n{}: "{}"\n\n->user: {}\n->machine: {}'.format(i, x['doc'], '/'.join(x['user_tags']), '/'.join(x['machine_tags'])))
        #
        # sys.exit(0)

        assign_clusters_to_works(works)

        # generate_cluster_hierachies()
