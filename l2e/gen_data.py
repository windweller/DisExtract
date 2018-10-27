"""
Generate LM data
two modes:
1. Unfiltered corpus to train LM on (does not learn sentence boundary) (vs. dependency parsing).
2. Because_CTX (with LM)
3. Because_NMT (with LM) (power of dependency + power of conditional LM) (need to add <G> token)

No GloVE embedidng. Glorot initialization.
"""
import os
import json
import argparse
from os.path import join as pjoin
from os.path import dirname, abspath

import numpy as np
import random

np.random.seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description='Language Model Producer')

parser.add_argument("--corpus", type=str, default='because_ctx',
                    help="because|because_nmt|because_ctx; because is the raw dataset without "
                         "dependency parsing")
parser.add_argument("--data_dir", type=str, default='default', help="the path for the data file")
parser.add_argument("--out_prefix", type=str, required=True,
                    help="Prefix the produced files, normally timestamp!")
# this is only used for filtered but unparsed corpus!
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--train_size", default=0.9, type=float)

args, _ = parser.parse_known_args()

# ======== Split =========

assert (args.train_size < 1 and args.train_size > 0)
split_proportions = {
    "train": args.train_size,
    "valid": (1 - args.train_size) / 2,
    "test": (1 - args.train_size) / 2
}
assert (sum([split_proportions[split] for split in split_proportions]) == 1)

corpus_dir = {
    # we focus on un-ordered data
    "because": "/home/anie/DisExtract/preprocessing/corpus/news_crawl/markers_BECAUSE/sentences/BECAUSE.json",
    "because_nmt": "/home/anie/OpenNMT-py/data/",
    "because_ctx": "/home/anie/OpenNMT-py/data/because_ctx"
}

file_list = {
    "because_nmt": {
        "train": ("src-train.txt", "tgt-train.txt"),
        "val": ("src-val.txt", "tgt-val.txt"),
        "test": ("src-test.txt", "tgt-test.txt")
    },
    "because_ctx": {
        "train": ("gigaword_newscrawl_ordered_ctx_2018oct11-src-train.txt",
                  "gigaword_newscrawl_ordered_ctx_2018oct11-tgt-train.txt"),
        "val": ("gigaword_newscrawl_ordered_ctx_2018oct11-src-valid.txt",
                "gigaword_newscrawl_ordered_ctx_2018oct11-tgt-valid.txt"),
        "test": ("gigaword_newscrawl_ordered_ctx_2018oct11-src-test.txt",
                 "gigaword_newscrawl_ordered_ctx_2018oct11-tgt-test.txt")
    }
}

# ======== Data Path =========
if args.data_dir == "default":
    root_dir = dirname(dirname(abspath(__file__)))
    args.data_dir = pjoin(root_dir, "data", args.corpus + '_lm')


def write_to_file(data, split):
    with open(pjoin(args.data_dir, args.out_prefix + "_" + split + '.txt'), 'w') as f:
        for line in data:
            f.write(line)


"""
For context, format is ".... || <Q> ... <G> ..."
For NMT, format is "... <G> ..."
for normal LM, no format is used.
"""


def produce_src_tgt(corpus):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    for split in ['train', 'val', 'test']:
        with open(pjoin(corpus_dir[corpus], file_list[corpus][split][0])) as src_f, \
                open(pjoin(corpus_dir[corpus], file_list[corpus][split][1])) as tgt_f, \
                open(pjoin(args.data_dir, args.out_prefix + "_" + split + '.txt'), 'w') as out_f:
            for src, tgt in zip(src_f, tgt_f):
                line = src.strip() + " <G> " + tgt.strip()
                out_f.write(line + "\n")


def produce_unparsed(corpus):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # load in json
    sentences = json.load(open(corpus_dir[corpus]))

    print("total sentences: {}".format(len(sentences["because"]["sentence"])))
    # we ignore the sentences["because"]["previous"] sentences

    filtered_sentences = []

    for sentence in sentences["because"]["sentence"]:
        sent = sentence.split()
        if len(sent) > args.max_seq_len:
            continue
        elif len(sent) < args.min_seq_len:
            continue
        else:
            filtered_sentences.append(sentence)

    del sentences

    examples = filtered_sentences
    serial_numbers = list(range(len(examples)))
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

    write_to_file(train, 'train')
    write_to_file(valid, 'valid')
    write_to_file(test, 'test')

if __name__ == '__main__':
    # most LM programs train on line by line file
    # we merge source and target file
    # this LM should learn sentence boundaries

    if args.corpus == 'because':
        produce_unparsed(args.corpus)
    else:
        produce_src_tgt(args.corpus)
