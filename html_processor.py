#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import html

def strip_tags(value):
    """Returns the given HTML with all tags stripped."""
    return html.unescape(re.sub(r'<[^>]*?>', '', value))
