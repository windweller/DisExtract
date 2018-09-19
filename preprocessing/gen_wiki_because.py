"""
For now, we are only generating evaluation from wiki
no training
"""

import argparse
import random
import numpy as np
from os.path import join as pjoin
from os.path import dirname, abspath

import logging
import sys

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

np.random.seed(123)
random.seed(123)


parser = argparse.ArgumentParser(description='Because Producer')

parser.add_argument("--corpus", type=str, default='because',
                    help="books|gigaword_ch|gigaword_es|ptb|wikitext|because")
parser.add_argument("--train_size", default=0.9, type=float)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--max_ratio", default=5.0, type=float)
parser.add_argument("--data_dir", type=str, default='default', help="the path for the data file")
parser.add_argument("--out_prefix", type=str, required=True,
                    help="Prefix the produced files, normally timestamp!")

args, _ = parser.parse_known_args()
args.min_ratio = 1 / args.max_ratio  # auto-generate min-ratio


# ======== Data Path =========
root_dir = dirname(dirname(abspath(__file__)))
data_dir = pjoin(root_dir, "data", args.corpus)

def write_to_opennmt(data, out_prefix, split_name):
    with open(pjoin(data_dir, '{}-src-{}.txt'.format(out_prefix, split_name)), 'w') as src:
        with open(pjoin(data_dir, '{}-tgt-{}.txt'.format(out_prefix, split_name)), 'w') as tgt:
            for line in data:
                s1, s2, label = line.strip().split('\t')  # need to remove '\n'
                src.write(s1 + '\n')
                tgt.write(s2 + '\n')

number_of_filtered_examples = 0

examples = []
with open(pjoin(data_dir, 'wikipedia_en_because.txt'), 'r') as f:
    for i, line in enumerate(f):
        s1, s2, label = line.strip().split('\t')
        # simple filtering!
        s1_len = len(s1.split())
        s2_len = len(s2.split())

        ratio = float(s1_len) / max(s2_len, 0.0001)

        if s1_len < args.min_seq_len or args.max_seq_len < s1_len:
            continue
        elif s2_len < args.min_seq_len or args.max_seq_len < s2_len:
            continue
        elif ratio < args.min_ratio or args.max_ratio < ratio:
            continue
        else:
            # example_line = "\t".join([s1, s2, label]) + "\n"
            examples.append(line)
            number_of_filtered_examples += 1

print("original number: {}, filtered out number: {}".format(i, i - number_of_filtered_examples))

write_to_opennmt(examples, args.out_prefix, 'test')
