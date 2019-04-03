#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import codecs
import sys
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import random


def _resolve_topic_to_wordnet(term):
    wordnet_lookup_url = 'http://arpedia.herokuapp.com:80/arpedia/v1/wordsense_lookup'
    r = requests.get(wordnet_lookup_url, params={
        'q': term
    })

    if r.status_code == requests.codes.ok:
        if r.json()['results']:
            for word_sense in r.json()['results']:
                if 'n.01' in word_sense['sense']:
                    return word_sense['sense']

            # can't find a noun form, use first one
            return r.json()['results'][0]['sense']

    print('  *', 'no word sense match for:', term)
    return None


def process_trials_sheets():
    source_file_trials_tags = 'data/KB Clusters trial v.2 - A - tags only.csv'
    source_file_trials_desc = 'data/KB Clusters trial v.2 - B - description only.csv'
    dest_file_trials = 'data/trials.json'

    trials = []

    for filename in [source_file_trials_tags, source_file_trials_desc]:

        df = pd.read_csv(filename)

        for index, row in df.iterrows():
            human_cluster_assignments = []
            for col in ['Cluster 1', 'Cluster 2', 'Cluster 3']:
                if col in df.columns and not pd.isnull(row[col]):
                    human_cluster_assignments.append(row[col])

            if human_cluster_assignments:
                trials.append({
                    "artist_name": row['artist_name'],
                    "permalink": row['permalink'],
                    "human_assessment_type": "tags" if filename == source_file_trials_tags else "description",
                    "title": row['title'],
                    "human_clusters": list(set([x.strip() for x in human_cluster_assignments])),
                    "user_tags": list(set([x.strip() for x in row['user_tags'].split(',')]))
                })

    with codecs.open(dest_file_trials, 'wb', 'utf-8') as f:
        f.write(json.dumps(trials, indent=True))
        print('  *', "written", len(trials), "trials")

    return trials

def validated_tags(cluster_name, tags):
    return [x for x in tags if x]

def process_clusters_sheet():
    source_file_clusters = 'data/KB Clusters trial v.2 - Clusters v.2.csv'
    dest_file_clusters = 'data/clusters.json'

    clusters = {}
    df = pd.read_csv(source_file_clusters, header=None)
    columns = []
    drops = []
    for col in df.columns:
        if isinstance(df[col][0], str):
            columns.append(df[col][0].split(',')[0])
        else:
            drops.append(col)
    df = df.drop(drops, axis=1)
    df.columns = columns

    for col in df.columns:
        s = df[col]
        s = s[~s.isnull()]
        header, terms = str(col).strip(), list(s)
        L = [[y.strip() for y in x.split(',')[1:]] for x in terms]
        clusters[header] = validated_tags(header, [item for sublist in L for item in sublist])

    with codecs.open(dest_file_clusters, 'wb', 'utf-8') as f:
        f.write(json.dumps(clusters, indent=True))
        print('  *', "written", len(clusters), "clusters")

    return clusters


if __name__ == '__main__':

    trials = process_trials_sheets()
    clusters = process_clusters_sheet()
