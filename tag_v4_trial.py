#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import os
import sys
import re
import json
import string

from document_tagger import DocumentTagger

if __name__ == '__main__':

    file_trials = "data/trial_v.4.json"
    with codecs.open(file_trials, 'rb', 'utf-8') as f_trials:
        docs = [x['title'] + ". " + x['description'] for x in json.loads(f_trials.read())]

    #.load_docs('docs') \
    result = DocumentTagger(['english', 'art']) \
        .load_string_docs(docs) \
        .process_documents(vocab_size=500)

    DocumentTagger.pprint_keywords(result)
