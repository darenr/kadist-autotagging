#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import sys
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import random
import unidecode

from people import people


def safe_filename(accented_string):
    """ make a safe filename with no non-ascii chars """
    return "".join([c for c in unidecode.unidecode(accented_string) \
        if c.isascii() or c.isdigit() or c == ' ']).rstrip()

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
    source_file_trials_tags = 'data/MASTER Clusters trial v.2 - A - tags only.csv'
    source_file_trials_desc = 'data/MASTER Clusters trial v.2 - B - description only.csv'
    dest_file_trials = 'data/trials.json'

    trials = []
    docs = []

    for filename in [source_file_trials_tags, source_file_trials_desc]:

        df = pd.read_csv(filename)

        for index, row in df.iterrows():
            if 'description' in row and row['description']:
                docs.append((row['artist_name'], row['description']))

            human_cluster_assignments = {}
            for person in people:
                human_cluster_assignments[person] = []
                for col in ['%s Cluster 1' % (person), '%s Cluster 2' % (person), '%s Cluster 3' % (person)]:
                    if col in df.columns and not pd.isnull(row[col]):
                        human_cluster_assignments[person].append(row[col])

            if human_cluster_assignments:
                trial = {
                    "artist_name": row['artist_name'],
                    "permalink": row['permalink'],
                    "human_assessment_type": "tags" if filename == source_file_trials_tags else "description",
                    "title": row['title'],
                    "user_tags": list(set([x.strip() for x in row['user_tags'].split(',')]))
                }
                for person in people:
                    human_assignment = "%s_assignments" % (person.lower())
                    trial[human_assignment] = list(set([x.strip() for x in human_cluster_assignments[person]]))

                trials.append(trial)

    with open(dest_file_trials,  mode="w", encoding="utf8") as f:
        print(json.dumps(trials, indent=True), file=f)
        print('  *', "written", len(trials), "trials")

    for artist, doc in docs:
        with open("docs/%s.txt" % (safe_filename(artist)), mode="w", encoding="utf8") as f:
            print(doc, file=f)


    print('  *', "written", len(docs), "documents")

    return trials


def validated_tags(cluster_name, tags):
    return [x for x in tags if x]


def process_clusters_sheet():
    source_file_clusters = 'data/Cluster trials v.4 and v.5 - Clusters.csv'
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

    #
    # "Arts" cluster is not useful, so remove it
    #
    if 'Arts' in clusters:
        del clusters['Arts']

    with open(dest_file_clusters,  mode="w", encoding="utf8") as f:
        print(json.dumps(clusters, indent=True), file=f)
        print('  *', "written", len(clusters), "clusters")

    return clusters


if __name__ == '__main__':

    trials = process_trials_sheets()
    clusters = process_clusters_sheet()
