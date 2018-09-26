"""
We can try to generate some context
(in unsupervised fashion)
for Why-type questions!

This requires loading in "because" sentence pairs from `./data` directory

We provide two types of generations:
turn the S1 into a question, or NOT.

Can train 2 models and compare difference.
Also, we use the same special token from CoQA.

Also, if there aren't any context that matches our criteria, we will exclude that entry from the dataset
"""

import argparse
import json
import logging
from tqdm import tqdm
from drqa import retriever  # this is installed globally

from os.path import join as pjoin
from os.path import dirname, abspath

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, default=None)
parser.add_argument('--data_json', type=str, required=True, default=None)
parser.add_argument('--inclusion_threshold', type=float, default=0.35,
                    help="the amount of query not in reference; (1 - threshold)% "
                         "is how much of query overlap with reference")
parser.add_argument('--score_threshold', type=int, default=200,
                    help="score computed by DrQA")
parser.add_argument("--data_dir", type=str, default='default', required=True,
                    help="the path for the data file")
parser.add_argument("--data_prefix", type=str, required=True,
                    help="Prefix the files that we need to load in")
parser.add_argument("--out_prefix", type=str, required=True,
                    help="Prefix the files that we are going to store")

args = parser.parse_args()

# ======== Data Path =========
if args.data_dir == "default":
    root_dir = dirname(dirname(abspath(__file__)))
    args.data_dir = pjoin(root_dir, "data", args.corpus)

id_to_text = {}
with open(args.data_json, 'r') as f:
    for line in tqdm(f):
        dict = json.loads(line.strip())
        id_to_text[dict['id']] = dict['text']

logger.info("initialize model")
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)


def inclusion_match(query, reference, silent=True):
    reference_set = set(reference.split())
    query_set = set(query.split())
    if not silent:
        print(len(query_set - reference_set) / float(len(query_set)))

    # this number should be small is most words are matched, this means 65% of words matched
    if len(query_set - reference_set) / float(len(query_set)) < args.inclusion_threshold:  # 0.35:
        # True meaning query and reference match, and we should reject
        return True
    else:
        return False

# ======== load in src/tgt ========



if __name__ == '__main__':
    pass