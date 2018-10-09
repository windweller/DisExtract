# -*- coding: utf-8 -*-

"""
Preprocess gigaword english

This file only deals with the flattened gigaword file.
The file is already sentence-tokenized, also word-tokenized.

"""

import os
import re
import io
import sys
import json
import gzip
from copy import copy
import argparse

import logging
from util import rephrase
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from cfg import DISCOURSE_MARKER_SET_TAG, EN_BECAUSE_MARKER, EN_DISCOURSE_MARKERS  # we only get "because", this will save a lot of parsing time


parser = argparse.ArgumentParser(description='DisExtract Gigaword English')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")

parser.add_argument("--filter", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--filter_because", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence) that has because")
parser.add_argument("--filter_print_every", default=10000, type=int)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--context_len", default=5, type=int, help="we are storing this number of sentences previous to context")

parser.add_argument("--parse", action='store_true',
                    help="Stage 3: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--exclude_list", action='store_true', help="use exclusion list defined in this file")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")

args, _ = parser.parse_known_args()

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

with open(args.json, 'rb') as f:
    json_config = json.load(f)

gigaword_en_dir = json_config['gigaword_en_dir']
gigaword_en_file = 'gigaword_en_flattened.txt'


def collect_raw_sentences(source_dir, filenames, marker_set_tag, discourse_markers):
    """
    This function needs to be implemented differently for each corpus
    since it contains crucial corpus-specific functions, though
    the main logic remains the same

    :param source_dir:
    :param filenames:
    :param marker_set_tag:
    :param discourse_markers:
    :return:
    """
    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    output_dir = pjoin(markers_dir, "sentences")

    if not os.path.exists(markers_dir):
        os.makedirs(markers_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sentences = {marker: {"sentence": [], "previous": [],
                          "before": []} for marker in discourse_markers}

    for filename in filenames:
        logger.info("reading {}".format(filename))
        file_path = pjoin(source_dir, filename)

        previous_sentence = ""
        previous_sentence_split = None
        FIRST = True

        before_list = []

        with io.open(file_path, 'rU', encoding="utf-8") as f:
            for i, sentence in enumerate(f):
                words = rephrase(sentence).split()  # replace "for example"
                for marker in discourse_markers:
                    if marker == "for example":
                        proxy_marker = "for_example"
                    else:
                        proxy_marker = marker

                    # [min_len, max_len) like [5, 10)
                    if len(words) >= args.max_seq_len or len(words) < args.min_seq_len:
                        continue

                    # length-based filtering
                    if not FIRST:
                        # this part might be uncalled for...
                        len2 = len(previous_sentence_split)
                        ratio = float(len2) / len(words)

                        if ratio <= args.min_ratio or ratio >= args.max_ratio:
                            continue

                    # all bookcorpus text are lower case
                    if proxy_marker in words:
                        sentences[marker]["sentence"].append(sentence)
                        sentences[marker]["previous"].append(previous_sentence)
                        sentences[marker]["before"].append(copy(before_list))

                # current methods won't allow us to capture "after" easily!
                if len(before_list) == args.context_len:
                    before_list.pop(0)
                    before_list.append(sentence)
                else:
                    before_list.append(sentence)

                previous_sentence = sentence
                previous_sentence_split = words

                if i % args.filter_print_every == 0:
                    logger.info("processed {}".format(i))

        logger.info("{} file finished".format(filename))

    logger.info('writing files')

    # bad idea! should use tsv instead!
    with open(pjoin(output_dir, "{}.json".format(marker_set_tag)), 'wb') as f:
        json.dump(sentences, f)

    logger.info('file writing complete')

    statistics_lines = []
    for marker in sentences:
        n_sentences = len(sentences[marker]["sentence"])
        statistics_lines.append("{}\t{}".format(marker, n_sentences))

    statistics_report = "\n".join(statistics_lines)
    with open(pjoin(markers_dir, "VERSION.txt"), "wb") as f:
        f.write(
            "commit: \n\ncommand: \n\nmarkers:\n" + statistics_report + '\n'
        )

def parse_filtered_sentences(source_dir, marker_set_tag):
    """
    This function can be the same for each corpus

    :param source_dir:
    :param marker_set_tag:
    :return:
    """

    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    input_dir = pjoin(markers_dir, "sentences")
    input_file_path = pjoin(input_dir, "{}.json".format(marker_set_tag))
    output_dir = pjoin(markers_dir, "parsed_sentence_pairs")

    if not os.path.exists(markers_dir):
        raise Exception("{} does not exist".format(markers_dir))
    if not os.path.exists(input_dir):
        raise Exception("{} does not exist".format(input_dir))
    if not os.path.exists(input_file_path):
        raise Exception("{} does not exist".format(input_file_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("setting up parser (actually just testing atm)")
    setup_corenlp()

    # parsed_sentence_pairs = {marker: {"s1": [], "s2": []} for marker in discourse_markers}
    with open(pjoin(output_dir, "{}_parsed_sentence_pairs.txt".format(marker_set_tag)), 'a') as w:

        with open(input_file_path, 'rb') as f:
            logger.info("reading {}".format(input_file_path))
            sentences = json.load(f)
            logger.info("total sentences: {}".format(
                sum([len(sentences[marker]["sentence"]) for marker in sentences])
            ))
            for marker, slists in sentences.iteritems():
                i = 0
                # the set will remove the same row
                for sentence, previous, ctx in set(zip(slists["sentence"], slists["previous"], slists["before"])):
                    i += 1
                    if True:
                        parsed_output = dependency_parsing(sentence, previous, marker)
                        if parsed_output:
                            s1, s2 = parsed_output

                            ctx_s = " ".join(ctx).replace('\n', '')

                            # parsed_sentence_pairs[marker]["s1"].append(s1)
                            # parsed_sentence_pairs[marker]["s2"].append(s2)
                            line_to_print = "{}\t{}\t{}\t{}\n".format(ctx_s, s1, s2, marker)
                            w.write(line_to_print)

                        if i % args.filter_print_every == 0:
                            logger.info("processed {}".format(i))

    logger.info('file writing complete')


def dependency_parsing(sentence, previous_sentence, marker):
    return depparse_ssplit(sentence, previous_sentence, marker, lang='en')


if __name__ == '__main__':
    if args.filter_because:
        collect_raw_sentences(gigaword_en_dir, [gigaword_en_file], "BECAUSE", EN_BECAUSE_MARKER)
    elif args.filter:
        collect_raw_sentences(gigaword_en_dir, [gigaword_en_file], DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS)
    elif args.parse:
        setup_corenlp("en")
        parse_filtered_sentences(gigaword_en_dir, "BECAUSE")
