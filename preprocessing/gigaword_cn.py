# -*- coding: utf-8 -*-

"""
Preprocess gigaword chinese 5th edition

We are only process text type of "story", and ignore the rest.
"story" is the most frequent type in this corpus.
"""

import os
import re
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

"""
1. Scan through the directory, save all folders
2. Unzip each file, parse them (XML), extract stories
3. Build: 
    1). Take out English words/characters (no need to do this step for Spanish)
    2). Map HTML entities back to normal characters
    3). Remove parentheses and their content
    4). <P> tag is not entirely "paragraphs", need to merge all paragraph and then sent tokenization
    5). Map 「 and 」to “ ” (which is more common) 
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
gigaword_cn_file = 'gigaword_cn.txt'

def process_sent(sent, lang="ch"):
    sent = re.sub(r"\(.+\)", "", sent)  # get rid of parentheses (many content inside are English/other languages)

    sent = sent.replace("&amp;gt;", "")

    # HTML entities
    sent = sent.replace("&lt;", '<')
    sent = sent.replace("&gt;", '>')
    sent = sent.replace("&amp;", '&')
    sent = sent.replace("&apos;", '\'')
    sent = sent.replace("&quot;", '"')

    if lang == "ch":
        sent = re.sub(r'[A-Z a-z.]+', "", sent)  # get rid of English characters
        # and all spaces in the sentence. This will only work in Chinese
        sent = re.sub(r'[0-9]+', "", sent)

    sent = re.sub(r"\(", "", sent)
    sent = re.sub(r"\)", "", sent)

    # resolve weird 「 symbol
    sent = sent.replace("「", '“')
    sent = sent.replace("」", "”")

    return sent


def extract_stories(lines):
    # pass in all lines from a gigaword xml file
    sentences = []

    story_doc = False
    paragraph = False
    paragraph_text = []
    for line in lines:
        if 'DOC' in line and 'type="story"' in line:
            story_doc = True
        if '<P>' in line and story_doc:
            paragraph = True
            continue
        if '</P>' in line and story_doc:
            paragraph = False
            sentence = "".join(paragraph_text).strip()
            # preprocess the sentence
            sentence = process_sent(sentence)
            sentences.append(sentence)
            paragraph_text = []
        if '</DOC>' in line and story_doc:
            story_doc = False

        if paragraph:
            paragraph_text.append(line)

    return sentences


def extrat_raw_gigaword():
    news_sources = os.listdir(pjoin(gigaword_cn_dir, 'data'))
    articles_processed = 0
    sentences = []
    for news_source in news_sources:
        files = os.listdir(pjoin(gigaword_cn_dir, 'data', news_source))
        files = filter(lambda s: '.gz' in s, files)
        for file in files:
            with gzip.open(pjoin(gigaword_cn_dir, 'data', news_source, file), 'rb') as f:
                file_content = f.read()
                lines = file_content.split('\n')
                sents = extract_stories(lines)
                sentences.extend(sents)
            articles_processed += 1
            if articles_processed % 20 == 0:
                print("processed {} articles".format(articles_processed))
                print("{} sentences are collected".format(len(sentences)))

    with open(pjoin(gigaword_cn_dir, gigaword_cn_file), 'wb') as f:
        for sent in sentences:
            f.write(sent + '\n')

if __name__ == '__main__':
    if args.extract:
        extrat_raw_gigaword()
