"""
We can try to generate some context
(in unsupervised fashion)
for Why-type questions!

This requires loading in "because" sentence pairs from `./data` directory

We provide two types of generations:
turn the S1 into a question, or NOT.

Can train 2 models and compare difference.
Also, we use the same special token from CoQA.

We split this file apart from gen_because so that it's easier to use
and this file takes longer to load relevant documents.
"""

import argparse
import json
import logging
from tqdm import tqdm
from drqa import retriever  # this is installed globally

from os.path import join as pjoin
from os.path import dirname, abspath
import numpy as np
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

np.random.seed(123)
random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, default=None)
parser.add_argument("--corpus", type=str, default='because_qa',
                    help="books|gigaword_ch|gigaword_es|ptb|wikitext|because_qa")
parser.add_argument('--data_json', type=str, required=True, default=None)
parser.add_argument("--data_dir", type=str, default='default', help="the path for the data file")
parser.add_argument("--train_size", default=0.9, type=float)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--max_ratio", default=5.0, type=float)
parser.add_argument("--context_len", default=6, type=int)  # 6x5=30 - 6x50 = 300 (30-300 length, similar to CoQA)
parser.add_argument("--out_prefix", type=str, required=True,
                    help="Prefix the files that we are going to store")
parser.add_argument("--s1_to_q", action='store_true', help="This will invoke AllenNLP SRL.")

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

# ======== Data Path =========
if args.data_dir == "default":
    root_dir = dirname(dirname(abspath(__file__)))
    args.data_dir = pjoin(root_dir, "data", args.corpus)

id_to_text = {}
with open(args.data_json, 'r') as f:
    for line in tqdm(f, total=49469914):
        dict = json.loads(line.strip())
        id_to_text[dict['id']] = dict['text']

logger.info("initialize model")
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)


def find(query, k=1):
    # we find the location of the query in the original document
    doc_names, doc_scores = ranker.closest_docs(query, k)

    assert len(doc_names) != 0

    for i in range(len(doc_names)):
        result = [i + 1, doc_names[i], '%.5g' % doc_scores[i], id_to_text[doc_names[i]]]

    return result


# no need to worry about lower-casing one word
# should still be highly matchable!
def s1_s2_to_query(s1, s2, marker='because'):
    return s1 + marker + s2


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


"""
{"id": "newscrawl_157004655", "text": "Lancaster was frustrated that International Rugby Board regulations forced him to send Morgan back to the Scarlets for Friday night 's RaboDirect PRO12 game against Connacht ."}
{"id": "newscrawl_157004656", "text": "But Lancaster voluntarily released 17 others , including fly - halves Charlie Hodgson and Toby Flood who are competing for a place in England 's 22 to face France a week on Sunday ."}
"""
def retrieve_context(doc_loc):
    source_str = doc_loc.split("_")[0]
    loc = int(doc_loc.split("_")[1])
    assert args.context_len % 2 == 0
    before = int(args.context_len / 2)
    after = int(args.context_len / 2)

    misses = 0.  # nearby context that is not available

    sents = []
    for inc in range(before):
        idx = loc - (before - inc)
        id_str = source_str + "_" + str(idx)
        if id_str in id_to_text:
            sents.append(id_to_text[id_str])
        else:
            misses += 1

    for inc in range(2, after + 1 + 1):
        idx = loc + inc
        id_str = source_str + "_" + str(idx)
        if id_str in id_to_text:
            sents.append(id_to_text[id_str])
        else:
            misses += 1

    return sents, misses


def write_to_file(data, file_name):
    with open(file_name, 'wb') as f:
        for line in data:
            f.write(line)

def fix_tok(s):
    return ' '.join(_str(w) for w in s.split())


def write_to_opennmt(data, out_prefix, split_name):
    total_misses = 0.
    with open(pjoin(args.data_dir, '{}-src-{}.txt'.format(out_prefix, split_name)), 'w') as src:
        with open(pjoin(args.data_dir, '{}-tgt-{}.txt'.format(out_prefix, split_name)), 'w') as tgt:
            for line in tqdm(data):
                s1, s2, label = line.strip().split('\t')  # need to remove '\n'

                # s1 and s2 from Stanford pipeline has tokenization issue, fix them
                s1 = fix_tok(s1)
                s2 = fix_tok(s2)

                # we retrieve context here!
                query = s1_s2_to_query(s1, s2, label)
                doc_loc = find(query)[1]  # doc_name indicates location

                context, misses = retrieve_context(doc_loc)
                total_misses += misses
                context = " ".join(context)  # concatenate them; no tokenization issue

                full_str = context + ' ||'
                full_str += ' <Q> ' + s1  # TODO: map S1 to question here

                # src.write(s1 + '\n')
                src.write(full_str + '\n')
                tgt.write(s2 + '\n')


if __name__ == '__main__':

    datafiles = ['gigaword_en_because.txt', 'news_crawl_because.txt']

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
        s1, s2, label = ex[:-1].split('\t')

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

    print(
    "original number: {}, filtered out number: {}".format(len(examples), len(examples) - number_of_filtered_examples))

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
