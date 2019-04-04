import nltk
from nltk.corpus import wordnet
import sys

def main(words):
    for word in words:
        print('word:', word)
        for syn in wordnet.synsets(word):
            print('  sense:', syn.name(), '[%s]' % (syn.definition()))
            print('    synonym:', str([l.name() for l in syn.lemmas()]))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print('usage: [word ..]')
