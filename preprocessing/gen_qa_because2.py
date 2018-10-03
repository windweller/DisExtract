"""
We get 2 DB
We read them in, store in list (because our memory is big enough)
iterate through both
"""

from os.path import join as pjoin
from os.path import dirname, abspath
import numpy as np
import random

import argparse
import json
import logging
from tqdm import tqdm
import sklearn
import editdistance

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

np.random.seed(123)
random.seed(123)

gigaword_db = []
with open('/home/anie/DisExtract/preprocessing/corpus/because/gigaword_db.txt') as f:
    for line in f:
        gigaword_db.append(line.strip())

newscrawl_db = []
with open('/home/anie/DisExtract/preprocessing/corpus/because/newscrawl_db.txt') as f:
    for line in f:
        newscrawl_db.append(line.strip())

gigaword_because = []
with open('/home/anie/DisExtract/data/because_qa/gigaword_en_because.txt') as f:
    for line in f:
        gigaword_because.append(line.strip())

newscrawl_because = []
with open('/home/anie/DisExtract/data/because_qa/news_crawl_ordered_because.txt') as f:
    for line in f:
        newscrawl_because.append(line.strip())


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


data_dir = "/home/anie/DisExtract/data/because_qa/"


def write_to_opennmt(data, out_prefix, split_name):
    with open(pjoin(data_dir, '{}-src-{}.txt'.format(out_prefix, split_name)), 'w') as src:
        with open(pjoin(data_dir, '{}-tgt-{}.txt'.format(out_prefix, split_name)), 'w') as tgt:
            for tup in data:
                s1, s2 = line  # need to remove '\n'
                src.write(s1 + '\n')
                tgt.write(s2 + '\n')


def sent_similarity(s1_list, s2_list):
    s1_set = set(s1_list)
    s2_set = set(s2_list)
    return len(s1_set.intersection(s2_set)) / float(len(s1_set.union(s2_set)))


def match(because_sents, dataset_sents):
    context_pairs = []
    misses = 0

    prev_matched_idx = 0  # update this, we never look back
    for i, line in enumerate(tqdm(because_sents)):
        # search in dataset_sents
        s1, s2, marker = line.strip().split('\t')
        s1_list = [_str(w) for w in s1.split()]
        s2_list = [_str(w) for w in s2.split()]

        query = s1_list + ['because'] + s2_list
        for i_2 in range(prev_matched_idx, len(dataset_sents)):
            if 'because' not in dataset_sents[i_2]:
                continue
            cand_sent_list = dataset_sents[i_2].split()
            if sent_similarity(query, cand_sent_list) > 0.6:
                # matched
                prev_matched_idx = i_2 + 1
                break
            else:
                continue

        if i_2 != len(dataset_sents) - 1:

            # obtained i_2, now retrieve context, non-inclusive
            context = dataset_sents[i_2 - 3:i_2] + dataset_sents[i_2 + 1:i_2 + 3 + 1]
            context = ' '.join(context)

            full_str = context + ' ||'
            full_str += ' <Q> ' + ' '.join(s1_list)

            context_pairs.append((full_str, s2))
        else:
            misses += 1

    logger.info("number of unmatched {}".format(misses))

    return context_pairs


giga_context_pairs = match(gigaword_because, gigaword_db)
write_to_opennmt(giga_context_pairs, "gigaword_ctx_s1_s2_2018oct2", 'all')

newscrawl_context_pairs = match(newscrawl_because, newscrawl_db)
write_to_opennmt(newscrawl_context_pairs, "newscrawl_ctx_s1_s2_2018oct2", "all")
