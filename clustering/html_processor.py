import re
from HTMLParser import HTMLParser
from htmlentitydefs import entitydefs

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
        self.entityref = re.compile('&[a-zA-Z][-.a-zA-Z0-9]*[^a-zA-Z0-9]')

    def handle_data(self, d):
        self.fed.append(d)

    def handle_starttag(self, tag, attrs):
        self.fed.append(' ')

    def handle_endtag(self, tag):
        self.fed.append(' ')

    def handle_entityref(self, name):
        if entitydefs.get(name) is None:
            m = self.entityref.match(self.rawdata.splitlines()[self.lineno-1][self.offset:])
            entity = m.group()
            # semicolon is consumed, other chars are not.
            if entity[-1] != ';':
                entity = entity[:-1]
            self.fed.append(entity)
        else:
            self.fed.append(' ')

    def get_data(self):
        self.close()    # N.B. ensure all buffered data has been processed
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data().strip()
