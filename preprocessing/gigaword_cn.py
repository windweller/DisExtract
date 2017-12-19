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

from parser import depparse_ssplit, setup_corenlp
from cfg import CH_DISCOURSE_MARKERS

parser = argparse.ArgumentParser(description='DisExtract Gigaword Chinese')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")

parser.add_argument("--filter", action='store_false',
                    help="Stage 1: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--max_ratio", default=5.0, type=float)
parser.add_argument("--filter_print_every", default=10000, type=int)

parser.add_argument("--parse", action='store_false',
                    help="Stage 2: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")

args, _ = parser.parse_known_args()
args.min_ratio = 1 / args.max_ratio  # auto-generate min-ratio
