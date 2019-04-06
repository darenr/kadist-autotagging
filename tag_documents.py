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

    result = DocumentTagger(['english', 'art']) \
        .load_docs('docs') \
        .process_documents(vocab_size=500)

    DocumentTagger.pprint_keywords(result)
