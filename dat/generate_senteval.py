"""
We can generate a 3-class classification dataset
for SentEval analysis, so we can compare models

Note that SentEval cannot take more than 300k pairs
(Memory issue)
"""
import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
from itertools import izip
from preprocessing.cfg import EN_DISCOURSE_MARKERS
from data import get_dis
import itertools

import numpy as np

parser = argparse.ArgumentParser(description='DAT data generation')
parser.add_argument("--corpus", type=str, default='books_all',
                    help="books_5|books_old_5|books_8|books_all|gw_5|gw_8")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--senteval", type=str, default='~/SentEval/data/senteval_data',
                    help="point to senteval data directory")
parser.add_argument("--train_size", default=0.9, type=float)
parser.add_argument("--cutoff", default=300000, type=int, help="-1 means no cutoff, the default is 300k")
parser.add_argument("--seed", type=int, default=1234, help="seed")

parser.add_argument("--include", type=str, default='causal_contrast_temporal_sequence_misc',
                    help="take one out to not generate; this will be part of corpus tag as well")

# load in the corpus
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

data_dir = json_config['data_dir']
prefix = json_config[params.corpus]

"""
Data
"""
marker_dict = get_dis(data_dir, prefix, params.corpus)

# Note 1: we are ignoring commonly confusable markers like "while", which can mean
# at the same time, or in contrast

groups_of_discourse_markers = {
    "causal": ['because', 'so'],
    "contrast": ['although', 'though', 'but'],
    "temporal": ['before', 'after'],
    "sequence": ['and', 'also', 'then'],
    "misc": ['if', 'while', 'as', 'when']
    # "temporal": ['while', 'as', 'when'] # but "while" can also signal contrast
}

# Note 2: we exclude marker like 'still' that looks weird...

order_invar_list = ['but', 'and', 'also', 'while', 'as', 'when']
order_dep_list = ['because', 'so', 'if', 'although',
                  'though', 'before', 'after', 'then']

# only called by order sequence
def flip(S1, C, S2):
    return S2 + C + S1

def swap(S1, C1, S2, C2):
    return S1 + C2 + S2

def flip_swap(S1, C1, S2, C2):
    return S2 + C2 + S1

# define instruction manual
transformation_pipelines = {}

# (C) -> ([flip], 'entail')
for C in order_invar_list:
    transformation_pipelines[C].append(([flip], 'entail'))
for C in order_dep_list:
    transformation_pipelines[C].append(([flip], 'contrast'))

# (C1, C2) -> [([swap | flip_swap], 'entail'), ([swap | flip_swap], 'contrast')]
# Define 'neutral': C for C in discourse_markers - groups_of_discourse_markers['misc']
# make sure 'while' and 'as' swap don't appear as 'neutral'
transformation_pipelines[('because', 'so')] = [(flip_swap, 'entail'),
                                               (swap, 'contrast')]
# "result because cause" entails "cause so result"

transformation_pipelines[('although', 'though')] = [(swap, 'entail'),
                                                    (flip_swap, 'entail')]
# Although it's raining, I want to go for a walk; It's raining, though I want to go for a walk.

# Note 3: certain groups don't have "contrast"...an inherent unbalanced problem
# Note 3: there's some feature-attachment here, "although" and "though" are order-invariant
# Note 3: so flip order does not affect them

transformation_pipelines[('although', 'but')] = [(swap, 'entail'),
                                                 (flip_swap, 'contrast')]
# It's raining, although I want to go for a walk; It's raining, but I want to go for a walk.
# It's raining, although I want to go for a walk; I want to go for a walk, but it's raining.

transformation_pipelines[('though', 'but')] = [(swap, 'entail'),
                                               (flip_swap, 'contrast')]

if __name__ == '__main__':
    # dataset = []  # (s1+C+s2, s1+C+s2, label)
    # included_sets = params.include.split("_")
    # included_markers = []
    # for included_set in included_sets:
    #     included_markers += groups_of_discourse_markers[included_set]
    #
    # for c1, c2 in itertools.product(included_markers, included_markers[1:]):
    #     if (c1, c2) in transformation_pipelines:
    #         ops = transformation_pipelines[(c1, c2)]
    #         for op in ops:
    #             funcs, label = op

    dataset = []
