# -*- coding: utf-8 -*-

"""
Preprocess gigaword chinese 5th edition

We are only process text type of "story", and ignore the rest.
"story" is the most frequent type in this corpus.
"""

import os
import io
import sys
import json
import gzip
import argparse

import logging
from util import rephrase
from os.path import join as pjoin

import xml.etree.ElementTree as ET

from parser import depparse_ssplit, setup_corenlp
from cfg import CH_DISCOURSE_MARKERS

"""
1. Scan through the directory, save all folders
2. Unzip each file, parse them (XML), extract stories
3. Build: 
    1). Map HTML entities back to normal characters
    2). Remove parentheses and their content
    3). <P> tag is not entirely "paragraphs", need to merge all paragraph and then sent tokenization
    4). Map `` and '' to " and " (which is more common) 
    5) Map ` and ' to ' and '
"""