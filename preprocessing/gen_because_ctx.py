"""
This merges and check conflicts

This should produce data file that can be consumed by OpenNMT
"""

import argparse
import random
import numpy as np
from os.path import join as pjoin
from os.path import dirname, abspath

import logging
import sys
import string

printable = set(string.printable)

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

np.random.seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description='Because Producer')

parser.add_argument("--corpus", type=str, default='because_ctx',
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

# ======== Split =========

assert (args.train_size < 1 and args.train_size > 0)
split_proportions = {
    "train": args.train_size,
    "valid": (1 - args.train_size) / 2,
    "test": (1 - args.train_size) / 2
}
assert (sum([split_proportions[split] for split in split_proportions]) == 1)

print("the data split is: {}".format(split_proportions))

def _str(s):
    """ Convert PTB tokens to normal tokens """
    if s.lower() == '-lrb-':
        s = '('
    elif s.lower() == '-rrb-':
        s = ')'
    elif s.lower() == '-lsb-':
        s = '['
    elif s.lower() == '-rsb-':
        s = ']'
    elif s.lower() == '-lcb-':
        s = '{'
    elif s.lower() == '-rcb-':
        s = '}'
    return s

def fix_tok(s):
    return ' '.join(_str(w) for w in s.split())

def fix_str(sent):
    if " n ' t " in sent:
        sent = sent.replace(" n ' t ", "n't")
    if " ' m " in sent:
        sent = sent.replace(" ' m ", "'m")
    return sent

# ======== Data Path =========
if args.data_dir == "default":
    root_dir = dirname(dirname(abspath(__file__)))
    args.data_dir = pjoin(root_dir, "data", args.corpus)


def write_to_file(data, file_name):
    with open(file_name, 'wb') as f:
        for line in data:
            f.write(line)


def write_to_opennmt(data, out_prefix, split_name):
    with open(pjoin(args.data_dir, '{}-src-{}.txt'.format(out_prefix, split_name)), 'w') as src:
        with open(pjoin(args.data_dir, '{}-tgt-{}.txt'.format(out_prefix, split_name)), 'w') as tgt:
            for line in data:
                ctx, s1, s2, label = line.strip().split('\t')  # need to remove '\n'

                ctx = filter(lambda x: x in printable, ctx)
                s1 = filter(lambda x: x in printable, s1)
                s2 = filter(lambda x: x in printable, s2)

                src.write(fix_str(" ".join(ctx)) + " || " + ' <Q> ' + fix_tok(s1) + '\n')
                tgt.write(fix_tok(s2) + '\n')


if __name__ == '__main__':

    datafiles = ['gigaword_en_because_ctx.txt', 'news_crawl_ordered_because_ctx.txt']

    check_repeat = set()
    examples = []

    total_num = 0.
    for data_file in datafiles:
        with open(pjoin(args.data_dir, data_file), 'rb') as f:
            for line in f:
                total_num += 1
                if line not in check_repeat:
                    check_repeat.add(line)
                    examples.append(line)

    print("original {}, found repeat {}".format(len(examples), total_num - len(examples)))

    del check_repeat  # release memory

    number_of_filtered_examples = 0
    new_examples = []
    for i, ex in enumerate(examples):
        ctx_s, s1, s2, label = ex[:-1].split('\t')

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
            new_examples.append(ex)
            number_of_filtered_examples += 1

    print("original number: {}, filtered out number: {}".format(len(examples), len(examples) - number_of_filtered_examples))

    assert number_of_filtered_examples != 0

    del examples
    examples = new_examples

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

    # now it's all set
    write_to_opennmt(train, args.out_prefix, 'train')
    write_to_opennmt(valid, args.out_prefix, 'valid')
    write_to_opennmt(test, args.out_prefix, 'test')
