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
from operator import itemgetter
import csv

from html_processor import normalize_text

random.seed(42)
min_word_length=50

def select_random_kadist_exhibitions(n=10):
    source_file_trials = 'data/kadist.json'
    candidates = []

    with codecs.open(source_file_trials, 'rb', 'utf-8') as f:
        for work in json.loads(f.read()):
            if 'description' in work and work['description'] and len(work['description'].split()) >= min_word_length:
                if work["object_type"] == "program" and "exhibition" in work["object_sub_type"]:
                    exhibition_title = work['title']
                    description = normalize_text(work['description'])
                    candidates.append({
                        "source": "kadist",
                        "exhibition_title": exhibition_title,
                        "description": description
                    })

    #
    # select sample
    #
    return candidates[:n]


def select_random_artfacts_exhibitions(n=10):

    source_file = 'data/processed_current_or_upcoming_exhibitions.json'

    candidates = []

    with codecs.open(source_file, 'rb', 'utf-8') as f:
        for work in json.loads(f.read()):
            if 'description' in work and work['description'] and len(work['description'].split()) >= min_word_length:
                exhibition_title = work['exhibition_title']
                description = normalize_text(work['description'])

                candidates.append({
                    "source": "artfacts",
                    "exhibition_title": exhibition_title,
                    "description": description
                })

    #
    # select sample
    #
    return candidates[:n]


if __name__ == '__main__':

    artfacts_sample = select_random_artfacts_exhibitions(10)
    kadist_sample = select_random_kadist_exhibitions(10)

    #
    # now write a csv for gsheet import
    #
    gsheet_csv_exhibitions = 'data/artfacts_kadist_sampled_exhibitions.csv'

    df = pd.DataFrame(artfacts_sample + kadist_sample, columns=['source', 'exhibition_title', 'description'])
    df.to_csv(gsheet_csv_exhibitions, sep=',', quoting=csv.QUOTE_ALL, index=False)

    print('written %d artfacts and %d kadist exhibition examples' % (len(artfacts_sample), len(kadist_sample)))
