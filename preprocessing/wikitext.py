#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This goes through the corpus,
select sentences that have the discourse marker
filtering based on length of sentence (so we can discard ill-formed sentences) 100 words
and save them as intermediate files
shuffle within each discourse marker
"""

import numpy as np
import argparse
import io
import nltk
import pickle
import requests
import re

from cfg import DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

import json
from itertools import izip

from copy import deepcopy as cp
from util import rephrase

np.random.seed(123)

"""
This file contains WikiText-specific information

1. Sentence tokenization (make it config)
2. Grab pairs of sentences where the 2nd sentence has one of the discourse markers
3. Save them, a file for each discourse markers (a json file with [,] is good enough)
for_example.txt, in side it's [{prev: "", sent: ""}]
"""

def collect_raw_sentences(source_dir, dataset, caching):
    markers_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    output_dir = pjoin(markers_dir, "files")

    if not os.path.exists(markers_dir):
        os.makedirs(markers_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset == "wikitext-103":
        filenames = [
            "wiki.train.tokens",
            "wiki.valid.tokens", 
            "wiki.test.tokens"
        ]
    else:
        raise Exception("not implemented")

    sentences = {marker: {"sentence": [], "previous": []} for marker in EN_DISCOURSE_MARKERS}
    
    for filename in filenames:
        print("reading {}".format(filename))
        file_path = pjoin(source_dir, "orig", filename)
        with io.open(file_path, 'rU', encoding="utf-8") as f:
            # tokenize sentences
            sentences_cache_file = file_path + ".CACHE_SENTS"
            if caching and os.path.isfile(sentences_cache_file):
                sent_list = pickle.load(open(sentences_cache_file, "rb"))
            else:
                tokens = f.read().replace("\n", ". ")
                print("tokenizing")
                sent_list = nltk.sent_tokenize(tokens)
                if caching:
                    pickle.dump(sent_list, open(sentences_cache_file, "wb"))

        # check each sentence for discourse markers
        previous_sentence = ""
        for sentence in sent_list:
            words = rephrase(sentence).split()  # replace "for example"
            for marker in EN_DISCOURSE_MARKERS:
                if marker == "for example":
                    proxy_marker = "for_example" 
                else:
                    proxy_marker = marker

                if proxy_marker in [w.lower() for w in words]:
                    sentences[marker]["sentence"].append(sentence)
                    sentences[marker]["previous"].append(previous_sentence)
            previous_sentence = sentence

    print('writing files')
    statistics_lines = []
    for marker in sentences:
        sentence_path = pjoin(output_dir, "{}_s.txt".format(marker))
        previous_path = pjoin(output_dir, "{}_prev.txt".format(marker))
        n_sentences = len(sentences[marker]["sentence"])
        statistics_lines.append("{}\t{}".format(marker, n_sentences))
        with open(sentence_path, "w") as sentence_file:
            for s in sentences[marker]["sentence"]:
                sentence_file.write(s + "\n")
        with open(previous_path, "w") as previous_file:
            for s in sentences[marker]["previous"]:
                previous_file.write(s + "\n")

    statistics_report = "\n".join(statistics_lines)
    open(pjoin(markers_dir, "VERSION.txt"), "w").write(
        "commit: \n\ncommand: \n\nmarkers:\n" + statistics_report
    )


