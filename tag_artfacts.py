#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import os
import sys
import re
import json
import string

def load_docs(folder='docs'):
    docs = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                with codecs.open(os.path.join(dirpath, filename), 'rb', 'utf-8') as f:
                    features = doc_to_features(f.read())
                    docs.append((filename, features))

    return docs


if __name__ == '__main__':
    dt = DocumentTagger(['english', 'art'])

    docs = load_docs()

    result = dt.process_documents(docs, vocab_size=500)

    pprint_keywords(result)
