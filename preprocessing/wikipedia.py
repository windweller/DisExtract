# -*- coding: utf-8 -*-

"""
This processes Wikipedia dump (wikimedia)
Needs to first process the xml with: Wikiextractor (https://github.com/attardi/wikiextractor)

python WikiExtractor.py --no-templates -o /home/anie/wikimedia/20180801/extracted_en_wiki.txt --filter_disambig_pages \
                        --processes 6  /home/anie/wikimedia/20180801/enwiki-latest-pages-articles.xml
(takes about 1.5 hour)

This file is a bit different. We filter at one directory, and store in one file
We then parse in a different place...
"""

import os
import re
import io
import sys
import json
import gzip
import string
import argparse

import logging
from util import rephrase
import nltk
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from cfg import DISCOURSE_MARKER_SET_TAG, EN_BECAUSE_MARKER, EN_DISCOURSE_MARKERS  # we only get "because", this will save a lot of parsing time

parser = argparse.ArgumentParser(description='DisExtract Gigaword English')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")

parser.add_argument('--extract', action='store_true',
                    help="Stage 1: compress all files into one flattened file")
parser.add_argument('--extract_out', type=str, default='flattened.txt', help="file name of extracted file")
parser.add_argument("--filter", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--filter_because", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence) that has because")
parser.add_argument("--filter_print_every", default=10000, type=int)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)

parser.add_argument("--parse", action='store_true',
                    help="Stage 3: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--exclude_list", action='store_true', help="use exclusion list defined in this file")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")

args, _ = parser.parse_known_args()

printable = set(string.printable)

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

with open(args.json, 'rb') as f:
    json_config = json.load(f)

wiki_en_dir = json_config['wikipedia_dir']
# gigaword_en_file = 'gigaword_en_flattened.txt'

wiki_files_path = []

wiki_file_dirs = os.listdir(wiki_en_dir)
for wiki_file_dir in wiki_file_dirs:
    wiki_files = os.listdir(pjoin(wiki_en_dir, wiki_file_dir))
    for w_f in wiki_files:
        wiki_files_path.append(pjoin(wiki_en_dir, wiki_file_dir, w_f))

print "total number of wikipedia files: ", len(wiki_files_path)

# if we want to extract context, we should re-process these files
def flatten_files():
    all_sentences = []
    for f_i, f_path in enumerate(wiki_files_path):
        with open(f_path, 'r') as f:
            title_mark = False
            for line in f:
                if '<doc id=' in line:
                    title_mark = True
                    continue
                elif '</doc>' in line:
                    continue
                elif line.strip() == '':  # empty line
                    continue
                elif title_mark is True: # so no titles
                    title_mark = False
                    continue

                processed = filter(lambda x: x in printable, line).strip()
                if len(processed.split()) < 5:  # take out non-ascii non-utf-8, if the sent is too short, we throw out
                    continue
                sentences = nltk.sent_tokenize(processed)
                all_sentences.extend(sentences)

        if f_i % 50 == 0:
            logger.info("processing {}".format(f_i))

    logger.info("writing to file...")
    with open(pjoin(wiki_en_dir, args.extract_out), 'w') as f:
        for s in all_sentences:
            f.write(s + '\n')

if __name__ == '__main__':
    if args.extract:
        flatten_files()