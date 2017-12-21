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
3. Save 
"""

parser = argparse.ArgumentParser(description='DisExtract Gigaword Chinese')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")
parser.add_argument("--extract", action='store_true')

parser.add_argument("--filter", action='store_false',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--filter_print_every", default=10000, type=int)

parser.add_argument("--parse", action='store_false',
                    help="Stage 3: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")

args, _ = parser.parse_known_args()

with open(args.json, 'rb') as f:
    json_config = json.load(f)

gigaword_cn_dir = json_config['gigaword_cn_dir']

def extrat_raw_gigaword():
    news_sources = os.listdir(pjoin(gigaword_cn_dir, 'data'))
    for news_source in news_sources:
        files = os.listdir(pjoin(gigaword_cn_dir, 'data', news_source))
        files = filter(lambda s: '.gz' in s, files)

def parse_one_file():
    file = "/Users/Aimingnie/Documents/School/Stanford/LING 236/DisExtract/data/cmn_gw_5/data/afp_cmn/afp_cmn_200010.gz"
    with gzip.open(file, 'rb') as f:
        file_content = f.read()

    root = ET.fromstring(file_content)

if __name__ == '__main__':
    if args.extract:
        parse_one_file()