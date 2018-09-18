# -*- coding: utf-8 -*-

"""
This processes Wikipedia dump (wikimedia)
Needs to first process the xml with: Wikiextractor (https://github.com/attardi/wikiextractor)

python WikiExtractor.py --no-templates -o /home/anie/wikimedia/20180801/extracted_en_wiki.txt --filter_disambig_pages \
                        --processes 6  /home/anie/wikimedia/20180801/enwiki-latest-pages-articles.xml
(takes about 1.5 hour)

This file is a bit different. We filter at one directory, and store in one file
We then parse in a different place...
"""

import os
import re
import io
import sys
import json
import gzip
import string
import argparse

import logging
from util import rephrase
import nltk
import spacy
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from cfg import DISCOURSE_MARKER_SET_TAG, EN_BECAUSE_MARKER, \
    EN_DISCOURSE_MARKERS  # we only get "because", this will save a lot of parsing time

parser = argparse.ArgumentParser(description='DisExtract Gigaword English')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")

parser.add_argument('--extract', action='store_true',
                    help="Stage 1: compress all files into one flattened file")
parser.add_argument('--extract_out', type=str, default='flattened.txt', help="file name of extracted file")
parser.add_argument("--filter", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--filter_because", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence) that has because")
parser.add_argument("--filter_print_every", default=10000, type=int)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)

parser.add_argument("--parse", action='store_true',
                    help="Stage 3: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--exclude_list", action='store_true', help="use exclusion list defined in this file")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")

args, _ = parser.parse_known_args()

printable = set(string.printable)

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

with open(args.json, 'rb') as f:
    json_config = json.load(f)

wikipedia_dir = json_config['wikipedia_dir']
wiki_en_dir = json_config['wiki_en_dir']
wiki_en_file = 'flattened_tokenized.txt'

wiki_files_path = []

wiki_file_dirs = os.listdir(wikipedia_dir)
for wiki_file_dir in wiki_file_dirs:
    if not os.path.isdir(pjoin(wikipedia_dir, wiki_file_dir)):
        continue
    wiki_files = os.listdir(pjoin(wikipedia_dir, wiki_file_dir))
    for w_f in wiki_files:
        wiki_files_path.append(pjoin(wikipedia_dir, wiki_file_dir, w_f))

print "total number of wikipedia files: ", len(wiki_files_path)


# if we want to extract context, we should re-process these files
def flatten_files():
    all_sentences = []
    for f_i, f_path in enumerate(wiki_files_path):
        with open(f_path, 'r') as f:
            title_mark = False
            for line in f:
                if '<doc id=' in line:
                    title_mark = True
                    continue
                elif '</doc>' in line:
                    continue
                elif line.strip() == '':  # empty line
                    continue
                elif title_mark is True:  # so no titles
                    title_mark = False
                    continue

                processed = filter(lambda x: x in printable, line).strip()
                if len(processed.split()) < 5:  # take out non-ascii non-utf-8, if the sent is too short, we throw out
                    continue
                sentences = nltk.sent_tokenize(processed)
                all_sentences.extend(sentences)

        if f_i % 50 == 0:
            logger.info("processing {}".format(f_i))

    logger.info("writing to file...")
    with open(pjoin(wiki_en_dir, args.extract_out), 'w') as f:
        for s in all_sentences:
            f.write(s + '\n')


def tokenize_files():
    en_nlp = spacy.load("en_core_web_sm")
    all_sentences = []
    with open(pjoin(wiki_en_dir, args.extract_out), 'r') as f:
        for i, line in enumerate(f):
            tokens = en_nlp.tokenizer(unicode(line.strip()))
            words = [str(token) for token in tokens if not str(token).isspace()]
            if len(words) < 5:
                continue
            all_sentences.append(' '.join(words))
            if i % 10000 == 0:
                logger.info("processed {} sentences".format(i))
    with open(pjoin(wiki_en_dir, 'flattened_tokenized.txt'), 'w') as f:
        for s in all_sentences:
            f.write(s + '\n')


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

    sentences = {marker: {"sentence": [], "previous": []} for marker in discourse_markers}

    for filename in filenames:
        logger.info("reading {}".format(filename))
        file_path = pjoin(source_dir, filename)

        previous_sentence = ""
        previous_sentence_split = None
        FIRST = True
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
                for sentence, previous in set(zip(slists["sentence"], slists["previous"])):
                    i += 1
                    if True:
                        parsed_output = dependency_parsing(sentence, previous, marker)
                        if parsed_output:
                            s1, s2 = parsed_output

                            # parsed_sentence_pairs[marker]["s1"].append(s1)
                            # parsed_sentence_pairs[marker]["s2"].append(s2)
                            line_to_print = "{}\t{}\t{}\n".format(s1, s2, marker)
                            w.write(line_to_print)

                        if i % args.filter_print_every == 0:
                            logger.info("processed {}".format(i))

    logger.info('file writing complete')


def dependency_parsing(sentence, previous_sentence, marker):
    return depparse_ssplit(sentence, previous_sentence, marker, lang='en')


if __name__ == '__main__':
    if args.extract:
        flatten_files()
        tokenize_files()
    elif args.filter_because:
        collect_raw_sentences(wiki_en_dir, [wiki_en_file], "BECAUSE", EN_BECAUSE_MARKER)
    elif args.parse:
        setup_corenlp("en")
        parse_filtered_sentences(wiki_en_dir, "BECAUSE")
