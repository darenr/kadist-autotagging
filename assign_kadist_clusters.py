#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import os
import sys
import re
import json
import string
import html

from common_functions import find_clusters, _mk_synset, wup, preprocess_clusters
from document_tagger import DocumentTagger


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
from html_processor import strip_tags

random.seed(42)


def prep_tagged_works(works):

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
                    description = strip_tags(work["description"]).strip()
                    if "artist_description" in work and work["artist_description"]:
                        artist_description = strip_tags(work["artist_description"]).strip()
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
                            "synsets": [_mk_synset(x) for x in tags],
                            "thumbnail": thumbnail_url,
                            "image_url": work['image_url'],
                            "permalink": permalink,
                        }
                    )

    return trials


def tags_to_clusters(clusters, trials, t, similarity, col_name, use_only_n_tags):
    """using the set of `clusters` find cluster from tags"""
    results = trials
    for work in tqdm(results):
        if 'user_tags' in work:
            sorted_scores = find_clusters(clusters, work['synsets'][:use_only_n_tags], t, similarity)
            work["{}_{}".format(col_name, "formatted")] = ['{}/{}'.format(c, s) for c, s in sorted_scores]
            work["{}_{}".format(col_name, "no_scores")] = [c for c, s in sorted_scores]
            work["{}_{}".format(col_name, "sum_of_scores")] = sum([s for c, s in sorted_scores])


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


def assign_clusters_to_works(trials, use_only_n_tags=6):
    dest_file = "results/kadist_assignments.csv"
    cluster_types = ['superclusters', 'clusters']
    similarity = wup
    T = 0.76

    for cluster_type in cluster_types:
        file_clusters = f'data/{cluster_type}.json'

        with codecs.open(file_clusters, 'rb', 'utf-8') as f_clusters:
            clusters = preprocess_clusters(json.loads(f_clusters.read()))

            tags_to_clusters(clusters, trials, t=T, similarity=similarity, col_name=cluster_type, use_only_n_tags=use_only_n_tags)

    df = pd.DataFrame(trials)
    df.drop(columns=['synsets'], inplace=True)

    # move some columns to front
    cols = df.columns.tolist()

    for col in ['permalink', 'thumbnail', 'image_url', 'user_tags',  'region', 'title', 'artist_description', 'description', 'artist_name']:
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


def tag_descriptions(works):
    """assign machine_tags to all works with descriptions"""

    docs = [x['artist_description'] for x in kadist]
    docs = ["{}. {}.".format(x['description'], x['artist_description']) for x in kadist]

    results = DocumentTagger() \
        .load_string_docs(docs) \
        .process_documents(vocab_size=1000)

    for doc_id, machine_tags in results.items():
        kadist[doc_id]['machine_tags'] = [x[0] for x in machine_tags if x[1] >= machine_tags[0][1] / 2.0]


if __name__ == "__main__":

    with codecs.open("data/kadist.json", "rb", "utf-8") as f:

        works = json.loads(f.read())

        works = prep_tagged_works()
        works = tag_descriptions(works)

        assign_clusters_to_works(works, use_only_n_tags=2)

        # generate_cluster_hierachies()
