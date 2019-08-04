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
import csv

from html_processor import normalize_text


def select_random_tagged_works(n=75):
    random.seed(42)

    source_file_trials = 'data/kadist.json'
    dest_file_trials = 'data/trials.json'
    gsheet_csv_trials = 'data/trials.csv'
    trials = []

    with codecs.open(source_file_trials, 'rb', 'utf-8') as f:
        for work in json.loads(f.read()):
            if 'description' in work:
                if 'user_tags' in work and work['user_tags']:
                    if '_thumbnails' in work and 'medium' in work['_thumbnails']:
                        artist_name = ', '.join([x['post_title'] for x in work['_artists']])
                        permalink = work['permalink']
                        title = work['title']
                        tags = work['user_tags']
                        description = normalize_text(work['description'])
                        thumbnail_url = work['_thumbnails']['medium']['url']
                        trials.append({
                            "artist_name": artist_name,
                            "title": title,
                            "description": description,
                            "user_tags": tags,
                            "thumbnail": thumbnail_url,
                            "permalink": permalink,
                        })

    #
    # select sample
    #
    random.shuffle(trials)
    sample = trials[:n]

    #
    # write json output
    #

    with codecs.open(dest_file_trials, 'wb', 'utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False, indent=True))
        print('\n  *', "written", len(sample), "trials", "\n" )

    #
    # now write a csv for gsheet import
    #
    df = pd.DataFrame(sample)
    df['user_tags'] = df['user_tags'].str.join(',')
    df['thumbnail'] = df['thumbnail'].apply(lambda url: '=IMAGE("%s", 1)' % (url))
    df.to_csv(gsheet_csv_trials, sep=',', encoding='utf-8', index=False)

    return sample


if __name__ == '__main__':

    samples = select_random_tagged_works()
