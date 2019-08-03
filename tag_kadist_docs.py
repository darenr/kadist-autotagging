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


"""

Takes the corpus of Kadist (that has human tags) and using the text assigns
machine tags.

"""

from document_tagger import DocumentTagger

if __name__ == '__main__':

    source_file_trials = "data/all_trials.json"
    dest_file_trials = "data/all_annotated_trials.json"

    with codecs.open(source_file_trials, "rb", "utf-8") as f:
        kadist = json.loads(f.read())
        docs = [x['artist_description'] for x in kadist]
        docs = ["{}. {}.".format(x['description'], x['artist_description']) for x in kadist]

        results = DocumentTagger(['english', 'art']) \
            .load_string_docs(docs) \
            .process_documents(vocab_size=1000)

        for doc_id, machine_tags in results.items():
            kadist[doc_id]['machine_tags'] = [x[0] for x in machine_tags if x[1] >= machine_tags[0][1]/2.0]

        with codecs.open(dest_file_trials, "wb", "utf-8") as f:
            f.write(json.dumps(kadist, ensure_ascii=False, indent=True))
            print("\n  *", "written", len(kadist), "kadist documents with machine_tags:", dest_file_trials, "\n")
