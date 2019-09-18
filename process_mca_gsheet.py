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
import html
from itertools import chain

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


def validated_tags(cluster_name, tags):
    return [x for x in tags if x]

def df_next_column(df, col):
    # returns the next column given a dataframe (df) and a column (col)
    for i, c in enumerate(df.columns):
        if c == col and i<len(df.columns):
            return df.columns[i+1]

def process_mca_trials():
    """
    Read CSV of data from spreadsheet with the following colums, then generate trials

    Artist,Description,Title,user_tags,Ivan Cluster 1,Ivan Cluster 2,Ivan Cluster 3,Vincent Cluster 1,Vincent Cluster 2,Vincent Cluster 3,Marie Cluster 1,Marie Cluster 2,Marie Cluster 3,Kathleen Cluster 1,Kathleen Cluster 2,Kathleen Cluster 3

    """

    dest_file_trials = 'data/mca_trials.json'

    df = pd.read_csv('data/MCA Australia Trial  - MCA Collection.csv')

    trials = []

    for index, row in df.iterrows():

        trial = {
            "artist_name": row['Artist'],
            "description": row['Description'],
            "title": row['Title'],
            "user_tags": row['user_tags']
        }

        if 'user_tags' in row and not pd.isnull(row['user_tags']):
            for person in people:
                human_cluster_assignments = []
                for col in ['%s Cluster 1' % (person), '%s Cluster 2' % (person), '%s Cluster 3' % (person)]:
                    if col in df.columns and not pd.isnull(row[col]):
                        human_cluster_assignments.append(row[col])

                if human_cluster_assignments:
                    trial[person] = human_cluster_assignments

        trials.append(trial)

    if trials:
        with open(dest_file_trials,  mode="w", encoding="utf8") as f:
            print(json.dumps(trials, indent=True, ensure_ascii=False), file=f)
            print('  *', "written {} trials [{}]".format(len(trials), dest_file_trials))


def process_clusters_sheet():
    source_file_clusters = 'data/MCA Australia Trial  - Clusters.csv'
    dest_file_clusters = 'data/mca_clusters.json'
    # dest_file_superclusters = 'data/mca_superclusters.json'

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

    with open(dest_file_clusters,  mode="w", encoding="utf8") as f:
        print(json.dumps(clusters, indent=True, ensure_ascii=False), file=f)
        print('  *', "written {} clusters [{}]".format(len(clusters), dest_file_clusters))

    # now aggregate and produce superclusters
    #
    # super_cluster_definition = {
    #     'Society': ['Colonization', 'Community', 'Economy', 'History', 'Urbanization', 'Violence', 'War'],
    #     'Politics':  ['Activism', 'Geopolitics', 'Inequality',  'Politics', 'Power'],
    #     'Individual_Personal': ['Body', 'Emotion', 'Familial', 'Identity', 'Mind', 'Spirituality', 'Values'],
    #     'Material_Physical': ['Environmental', 'Immaterial', 'Land', 'Materiality', 'Physics', 'Space', 'Time'],
    #     'Cultural': ['Arts', 'Culture', 'Design', 'Fantasy', 'Language', 'Media', 'Technology']
    # }
    #
    # superclusters = {
    #     k: list(chain.from_iterable([clusters[c] for c in subgroup]))
    #     for k,subgroup in super_cluster_definition.items()
    # }
    #
    # with open(dest_file_superclusters,  mode="w", encoding="utf8") as f:
    #     print(json.dumps(superclusters, indent=True, ensure_ascii=False), file=f)
    #     print('  *', "written {} clusters [{}]".format(len(superclusters), dest_file_superclusters))


if __name__ == '__main__':

    #trials = process_trials_sheets()
    clusters = process_clusters_sheet()
    process_mca_trials()
