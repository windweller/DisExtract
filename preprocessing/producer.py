#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import json
import argparse
import logging
from itertools import izip

import random
import numpy as np
from util import rephrase
from os.path import join as pjoin
from os.path import dirname, abspath

from cfg import DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS
from re import compile as _Re

_unicode_chr_splitter = _Re('(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split


def split_unicode_chrs(text):
    return [chr for chr in _unicode_chr_splitter(text) if chr]


import sys

reload(sys)
sys.setdefaultencoding('utf8')

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

np.random.seed(123)
random.seed(123)

"""
Add additional postprocessing steps for (s1, s2) here, such as delete punctuations

We apply filtering to balance s1 and s2 length
Merge them into one set, train/val/test split, np.shuffle (fix random seed)

(then Torchtext can take it from there!)
"""

parser = argparse.ArgumentParser(description='DisExtract Producer')

# parser.add_argument("--json", type=str, default="example_config.json", help="load in config params")
parser.add_argument("--corpus", type=str, default='books',
                    help="books|gigaword_ch|gigaword_es, marked by Spanish and Chinese")
parser.add_argument("--train_size", default=0.9, type=float)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--max_ratio", default=5.0, type=float)
parser.add_argument("--data_dir", type=str, default='default', help="the path for the data file")
parser.add_argument("--data_file", type=str, required=True,
                    help="Load in the tsv file name generated by bookcorpus.py or gigaword.py")
parser.add_argument("--out_prefix", type=str, required=True,
                    help="Prefix the produced files")
parser.add_argument("--balanced", action='store_true', help="use this flag to cut all markers off at the minimum count")
parser.add_argument("--count_per_marker", type=int, default=-1,
                    help="use this for modifying the cutoff for a 'balanced' dataset, by default perfectly balanced")
parser.add_argument("--exclude", type=str, default="")
parser.add_argument("--stf_seg_path", type=str, default="")
parser.add_argument("--stf_slf4j_path", type=str, default="")
parser.add_argument("--char", action='store_true',
                    default="only used to generate Chinese in char level, no word segmentation")

args, _ = parser.parse_known_args()
args.min_ratio = 1 / args.max_ratio  # auto-generate min-ratio

if args.stf_slf4j_path != "":
    path_to_slf4j = pjoin(args.stf_slf4j_path, 'slf4j-api.jar')
if args.stf_seg_path != "":
    path_to_jar = pjoin(args.stf_seg_path, 'stanford-segmenter-3.8.0.jar')

# ======== Split =========

assert (args.train_size < 1 and args.train_size > 0)
split_proportions = {
    "train": args.train_size,
    "valid": (1 - args.train_size) / 2,
    "test": (1 - args.train_size) / 2
}
assert (sum([split_proportions[split] for split in split_proportions]) == 1)

print("the data split is: {}".format(split_proportions))

# ======== Data Path =========
if args.data_dir == "default":
    root_dir = dirname(dirname(abspath(__file__)))
    args.data_dir = pjoin(root_dir, "data", args.corpus)


def write_to_tsv(data, file_name):
    with open(file_name, 'wb') as f:
        for line in data:
            f.write(line)


def add_one_to_dict(dic, entry):
    if entry in dic:
        dic[entry] += 1
    else:
        dic[entry] = 1


def print_dict(dict):
    # prepare for UTF-8 Chinese
    for key, value in dict.iteritems():
        print "{}: {}".format(key, value)


if __name__ == '__main__':

    examples = []

    with open(pjoin(args.data_dir, args.data_file), 'rb') as f:
        for line in f:
            examples.append(line)

    if args.corpus == "gigaword_ch" and not args.char:
        print "segmenting each example for Chinese, could take a while"
        from nltk.tokenize.stanford_segmenter import StanfordSegmenter

        seg = StanfordSegmenter(path_to_slf4j=path_to_slf4j, path_to_jar=path_to_jar)
        seg.default_config('zh')

    # ==== Filtering =====
    data_dist = {}
    filtered_examples = {}
    number_of_filtered_examples = 0
    for i, ex in enumerate(examples):
        s1, s2, label = ex[:-1].split('\t')

        if args.corpus == 'gigaword_ch':
            s1 = s1.replace(' .', '。')  # parser appended normal period
            s2 = s2.replace(' .', '。')

        if args.char and args.corpus == "gigaword_ch":
            # we presplit into chars
            s1 = " ".join(split_unicode_chrs(s1.decode('utf-8'))).encode('utf-8')
            s2 = " ".join(split_unicode_chrs(s2.decode('utf-8'))).encode('utf-8')

        s1_len = len(s1.split()) if args.corpus != "gigaword_ch" else len(s1.decode('utf-8'))
        s2_len = len(s2.split()) if args.corpus != "gigaword_ch" else len(s2.decode('utf-8'))

        ratio = float(s1_len) / max(s2_len, 0.0001)

        if s1_len < args.min_seq_len or args.max_seq_len < s1_len:
            continue
        elif s2_len < args.min_seq_len or args.max_seq_len < s2_len:
            continue
        elif ratio < args.min_ratio or args.max_ratio < ratio:
            continue
        else:
            example_line = "\t".join([s1, s2, label]) + "\n"
            if label in filtered_examples:
                filtered_examples[label].append(example_line)
            else:
                filtered_examples[label] = [example_line]
            # filtered_examples.append("\t".join([s1, s2, label]))
            # collect stats
            add_one_to_dict(data_dist, label)
            number_of_filtered_examples += 1

    print("original number: {}, filtered out number: {}".format(len(examples), number_of_filtered_examples))

    assert number_of_filtered_examples != 0

    print("label distribution:")
    print(print_dict(data_dist))

    minimum_count_per_marker = min(data_dist.values())

    exclude_marker_list = args.exclude.split(",")

    examples = []
    for label in filtered_examples:
        if label in exclude_marker_list:
            pass
        else:
            if args.balanced:
                random.shuffle(filtered_examples[label])
                if args.count_per_marker == -1:
                    count_per_marker = minimum_count_per_marker
                else:
                    count_per_marker = args.count_per_marker
                examples += filtered_examples[label][:count_per_marker]
            else:
                examples += filtered_examples[label]

    print "total number in produced dataset: {}".format(len(examples))

    # now we word segment for Chinese
    if args.corpus == "gigaword_ch" and not args.char:
        s1_list, s2_list, labels = [], [], []
        for ex in examples:
            s1, s2, label = ex.split('\t')

            s1_list.append(s1.decode('utf-8'))
            s2_list.append(s2.decode('utf-8'))
            labels.append(label)

        logging.info("s1, s2 collected, segmentation begins")
        s1_list = seg.segment_sents(s1_list)
        s1_list = s1_list.split('\n')[:-1]
        logging.info("s1 segmented")

        s2_list = seg.segment_sents(s2_list)
        s2_list = s2_list.split('\n')[:-1]
        logging.info("s2 segmented")

        examples = []
        assert len(s1_list) == len(s2_list) == len(labels)
        for i in range(len(s1_list)):
            example_line = "\t".join([s1_list[i], s2_list[i], labels[i]])  # label has '\n'
            examples.append(example_line)  # no need to encode in utf-8 anymore, seg produces utf-8

        logging.info("data list generated")

    serial_numbers = range(len(examples))
    random.shuffle(serial_numbers)

    train_numbers = serial_numbers[:int(np.rint(len(examples) * split_proportions['train']))]
    valid_numbers = serial_numbers[
                    int(np.rint(len(examples) * split_proportions['train'])): \
                        int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid'])))]
    test_numbers = serial_numbers[
                   int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid']))):]

    print(
        "train/valid/test number of examples: {}/{}/{}".format(len(train_numbers), len(valid_numbers),
                                                               len(test_numbers)))

    train, valid, test = [], [], []

    for tn in train_numbers:
        train.append(examples[tn])
    for tn in valid_numbers:
        valid.append(examples[tn])
    for tn in test_numbers:
        test.append(examples[tn])

    # Note that under default setting, corpus is already appended
    write_to_tsv(train, pjoin(args.data_dir, args.out_prefix + "_train.tsv"))
    write_to_tsv(valid, pjoin(args.data_dir, args.out_prefix + "_valid.tsv"))
    write_to_tsv(test, pjoin(args.data_dir, args.out_prefix + "_test.tsv"))
