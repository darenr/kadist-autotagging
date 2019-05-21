#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from nltk.corpus import wordnet
import sys

#
# helper functions to turn words into synsets and vice versa
#


def _show_possible_synsets(w):
    return wordnet.synsets(w)


def _mk_synset(w):
    #
    # (synset form) cat.n.01 into the Synset object form
    # (lemma form) syndicate.n.01.crime_syndicate
    #

    word = w.strip().replace(" ", "_")

    if word.count(".") == 2:
        try:
            return wordnet.synset(word)
        except Exception as ex:
            try:
                # try the first for the stem word
                return wordnet.synsets(word.split(".")[0])[0]
            except Exception as ex:
                return None

    elif word.count(".") == 3:
        try:
            return wordnet.lemma(word).synset()
        except Exception as ex:
            return None

    else:
        print(" * Error, invalid synset name", w, "skipping...")
        return None


def _mk_wv_word(s):
    #
    # turn wordnet Synset into regular word form
    #   e.g. cat.n.01 -> 'cat'
    #   e.g. free_trade.n.01 -> free-trade
    return s.lemmas()[0].name()


#
# two distance methods, wup and path
#


def wup(w1, w2, t):
    distance = w1.wup_similarity(w2)
    if distance:
        if distance >= t:
            return distance
    return 0


if __name__ == "__main__":

    if len(sys.argv) == 2:
        print(_show_possible_synsets(sys.argv[1]))
    elif len(sys.argv) == 3:
        print("{} -> {} == {}".format(sys.argv[1], sys.argv[2], wup(_mk_synset(sys.argv[1]), _mk_synset(sys.argv[2]), 0)))
    else:
        print("usage: <plain english word: show synsets>|<synset> <synset>: show path distance")
