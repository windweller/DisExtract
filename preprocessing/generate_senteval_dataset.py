"""
This file loads in processed corpus
and produce a SentEval compatible data
"""

import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
from itertools import izip

import numpy as np

from model.data import get_dis
from preprocessing.cfg import EN_FIVE_DISCOURSE_MARKERS, \
    EN_EIGHT_DISCOURSE_MARKERS, EN_DISCOURSE_MARKERS, EN_OLD_FIVE_DISCOURSE_MARKERS

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--corpus", type=str, default='books_5',
                    help="books_5|books_old_5|books_8|books_all|gw_5|gw_8")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--senteval", type=str, default='~/SentEval/data/senteval_data',
                    help="point to senteval data directory")
parser.add_argument("--merge", action='store_false', help="by default, we merge test and dev")
parser.add_argument("--train_size", default=0.9, type=float)
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

np.random.seed(params.seed)

split_proportions = {
    "train": params.train_size,
    "valid": (1 - params.train_size) / 2,
    "test": (1 - params.train_size) / 2
}

"""
Default json file loading
"""
with open(params.hypes, 'rb') as f:
    json_config = json.load(f)

DIS_dir = pjoin(params.senteval, "DIS")


def write_to_file(file_name, contents, assignments, label_list=None):
    with open(pjoin(DIS_dir, file_name), 'w') as f:
        if label_list is None:
            for a in assignments:
                f.write(contents[a] + '\n')
        else:
            # if it's label, we map it to original label
            contents = map(lambda l: label_list[l], contents)
            for a in assignments:
                f.write(contents[a] + '\n')


def write_to_senteval_format(valid, test):
    if params.corpus == "books_5":
        dis_list = EN_FIVE_DISCOURSE_MARKERS
    elif params.corpus == "books_8":
        dis_list = EN_EIGHT_DISCOURSE_MARKERS
    elif params.corpus == "books_all":
        dis_list = EN_DISCOURSE_MARKERS
    elif params.corpus == "books_old_5":
        dis_list = EN_OLD_FIVE_DISCOURSE_MARKERS
    else:
        raise Exception("Corpus/Discourse Tag Set {} not found".format(params.corpus))

    if not os.path.exists(DIS_dir):
        os.makedirs(DIS_dir)

    if params.merge:
        merged = {'s1': valid['s1'] + test['s1'], 's2': valid['s2'] + test['s2'],
                  'label': valid['label'].tolist() + test['label'].tolist()}
    else:
        valid['label'] = valid['label'].tolist()
        merged = valid

    num_examples = len(merged['s1'])
    assignments = range(num_examples)
    np.random.shuffle(assignments)

    train_numbers = assignments[:int(np.rint(num_examples * split_proportions['train']))]
    valid_numbers = assignments[int(np.rint(num_examples * split_proportions['train'])): int(
        np.rint(num_examples * (split_proportions['train'] + split_proportions['valid'])))]
    test_numbers = assignments[int(np.rint(num_examples * (split_proportions['train'] + split_proportions['valid']))):]

    write_to_file('s1.train', merged['s1'], train_numbers)
    write_to_file('s2.train', merged['s2'], train_numbers)
    write_to_file('labels.train', merged['label'], train_numbers, dis_list)

    write_to_file('s1.dev', merged['s1'], valid_numbers)
    write_to_file('s2.dev', merged['s2'], valid_numbers)
    write_to_file('labels.dev', merged['label'], valid_numbers, dis_list)

    write_to_file('s1.test', merged['s1'], test_numbers)
    write_to_file('s2.test', merged['s2'], test_numbers)
    write_to_file('labels.test', merged['label'], test_numbers, dis_list)


if __name__ == '__main__':
    data_dir = json_config['data_dir']
    prefix = json_config[params.corpus]

    _, valid, test = get_dis(data_dir, prefix, params.corpus)

    write_to_senteval_format(valid, test)
