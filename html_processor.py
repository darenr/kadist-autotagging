#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import html
import string
from nltk.tokenize import word_tokenize

def normalize_text(value):
    """Returns the given HTML with all tags stripped."""

    text = html.unescape(re.sub(r'<[^>]*?>', '', value)).replace('\n', '  ')
    return text
    # tokens = word_tokenize(text)

    # table = str.maketrans('', '', string.punctuation.replace('.', ''))
    # stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    # words = [word for word in stripped if word.isalpha() or word is '.']

    # return ' '.join(tokens)
