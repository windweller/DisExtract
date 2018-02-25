"""
Load in data, and interactively filter sentences to give what we want
"""

import json
import argparse
from os.path import join as pjoin
from itertools import izip
from preprocessing.cfg import EN_DISCOURSE_MARKERS
from data import get_dis
import itertools

import numpy as np
import IPython

parser = argparse.ArgumentParser(description='DAT data generation')
parser.add_argument("--corpus", type=str, default='books_5',
                    help="books_5|books_old_5|books_8|books_all|gw_5|gw_8")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--senteval", type=str, default='~/SentEval/data/senteval_data',
                    help="point to senteval data directory")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# load in the corpus
params, _ = parser.parse_known_args()

"""
Default json file loading
"""
with open(params.hypes, 'rb') as f:
    json_config = json.load(f)

data_dir = json_config['data_dir']
prefix = json_config[params.corpus]

"""
Data
"""
marker_dict = get_dis(data_dir, prefix, params.corpus)

"""
Search
"""

def display_marker():
    print EN_DISCOURSE_MARKERS

def get_sent(marker, idx=-1, rand=True):
    if idx != -1:
        rand = False

    if rand:
        nums = range(len(marker_dict[marker]))
        np.random.shuffle(nums)
        ex_num = nums[0]

        return marker_dict[marker][ex_num]
    else:
        return marker_dict[marker][idx]


def get_sents(marker, st=0, en=10, rand=False):
    if rand:
        nums = range(len(marker_dict[marker]))
        np.random.shuffle(nums)
        choices = nums[st:en]
        selected = [marker_dict[marker][ch] for ch in choices]
    else:
        selected = marker_dict[marker][st:en]

    return selected


if __name__ == '__main__':
    # call in console
    IPython.embed()
