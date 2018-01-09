# -*- coding: utf-8 -*-

"""
Preprocess gigaword chinese 5th edition

We are only process text type of "story", and ignore the rest.
"story" is the most frequent type in this corpus.
"""

import os
import io
import sys
import json
import gzip
import argparse
import re
import nltk.data

import logging
from util import rephrase
from os.path import join as pjoin

import xml.etree.ElementTree as ET

from parser import depparse_ssplit, setup_corenlp
from cfg import SP_DISCOURSE_MARKERS


"""
Stats:
Raw gigaword es file has: ???? (??M) sentences

Unlike bookcorpus.py, we are not filtering anything (due to difficulty in tokenization for raw string)
"""

parser = argparse.ArgumentParser(description='DisExtract Gigaword Spanish')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")
parser.add_argument("--extract", action='store_true')

parser.add_argument("--filter", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--filter_print_every", default=10000, type=int)

parser.add_argument("--parse", action='store_true',
                    help="Stage 3: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")
parser.add_argument("--marker_set_tag", default="ALL", type=str, help="ALL|FIVE|EIGHT")

args, _ = parser.parse_known_args()

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

with open(args.json, 'rb') as f:
    json_config = json.load(f)

gigaword_es_dir = json_config['gigaword_es_dir']
gigaword_es_file = 'gigaword_es.txt'

def process_sent(sent, lang="es"):
    sent = re.sub(r"\(.+\)", "", sent)  # get rid of parentheses (many content inside are English/other languages)

    sent = sent.replace("&amp;gt;", "")

    # HTML entities
    sent = sent.replace("&lt;", '<')
    sent = sent.replace("&gt;", '>')
    sent = sent.replace("&amp;", '&')
    sent = sent.replace("&apos;", '\'')
    sent = sent.replace("&quot;", '"')

    if lang == "ch":
        sent = re.sub(r'[A-Z a-z.]+', "", sent)  # get rid of English characters
        # and all spaces in the sentence. This will only work in Chinese
        sent = re.sub(r'[0-9]+', "", sent)

    sent = re.sub(r"\(", "", sent)
    sent = re.sub(r"\)", "", sent)

    # # resolve weird 「 symbol
    # sent = sent.replace("「", '“')
    # sent = sent.replace("」", "”")

    return sent


def extract_stories(lines):
    # pass in all lines from a gigaword xml file
    sentences = []

    story_doc = False
    paragraph = False
    paragraph_text = []
    for line in lines:
        if 'DOC' in line and 'type="story"' in line:
            story_doc = True
        if '<P>' in line and story_doc:
            paragraph = True
            continue
        if '</P>' in line and story_doc:
            paragraph = False
            sentence = "".join(paragraph_text).strip()
            # preprocess the sentence
            sentence = process_sent(sentence)
            sentences.append(sentence)
            paragraph_text = []
        if '</DOC>' in line and story_doc:
            story_doc = False

        if paragraph:
            paragraph_text.append(line)

    return sentences


def extrat_raw_gigaword():
    news_sources = os.listdir(pjoin(gigaword_es_dir, 'data'))
    articles_processed = 0
    sentences = []
    for news_source in news_sources:
        files = os.listdir(pjoin(gigaword_es_dir, 'data', news_source))
        files = filter(lambda s: '.gz' in s, files)
        for file in files:
            with gzip.open(pjoin(gigaword_es_dir, 'data', news_source, file), 'rb') as f:
                file_content = f.read()
                lines = file_content.split('\n')
                sents = extract_stories(lines)
                sentences.extend(sents)
            articles_processed += 1
            if articles_processed % 20 == 0:
                print("processed {} articles".format(articles_processed))
                print("{} sentences are collected".format(len(sentences)))

    with open(pjoin(gigaword_es_dir, gigaword_es_file), 'wb') as f:
        for sent in sentences:
            f.write(sent + '\n')

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

    spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

    for filename in filenames:
        logger.info("reading {}".format(filename))
        file_path = pjoin(source_dir, filename)
        i = 0

        previous_sentence = ""
        with io.open(file_path, 'rU', encoding="utf-8") as f:
            for raw_section in f:

                # # TODO: use
                section = raw_section.replace("\n", "")
                sentences_in_section = spanish_tokenizer.tokenize(section)
                for sentence in sentences_in_section:
                    i+=1

                    for marker in discourse_markers:

                        # all bookcorpus text are lower case
                        marker_at_start = marker.capitalize()
                        marker_in_middle = marker

                        if marker == "y":
                            marker_at_start = "Y "
                            marker_in_middle = " y "

                        if marker == "si":
                            marker_at_start = "Si "
                            marker_in_middle = " si "

                        if marker == "pero":
                            marker_at_start = "Pero "
                            marker_in_middle = " pero "

                        if marker_at_start in sentence or marker_in_middle in sentence:
                            sentences[marker]["sentence"].append(sentence)
                            sentences[marker]["previous"].append(previous_sentence)

                    previous_sentence = sentence

                    if i % args.filter_print_every == 0:
                        logger.info("processed {}".format(i))

        logger.info("{} file finished".format(filename))

    logger.info('writing files')

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
            "commit: \n\ncommand: \n\nmarkers:\n" + statistics_report
        )

def parse_filtered_sentences(source_dir, input_marker_set_tag, output_marker_set_tag):
    """
    This function can be the same for each corpus

    :param source_dir:
    :param marker_set_tag:
    :return:
    """

    markers_dir = pjoin(source_dir, "markers_" + input_marker_set_tag)
    input_dir = pjoin(markers_dir, "sentences")
    input_file_path = pjoin(input_dir, "{}.json".format(input_marker_set_tag))
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
    with open(pjoin(output_dir, "{}_parsed_sentence_pairs.txt".format(output_marker_set_tag)), 'a') as w:
        # header = "{}\t{}\t{}\n".format("s1", "s2", "marker")
        # w.write(header)

        with open(input_file_path, 'rb') as f:
            logger.info("reading {}".format(input_file_path))
            sentences = json.load(f)
            logger.info("total sentences: {}".format(
                sum([len(sentences[marker]["sentence"]) for marker in sentences])
            ))
            i = 0
            for marker, slists in sentences.iteritems():
                for sentence, previous in zip(slists["sentence"], slists["previous"]):
                    i += 1
                    if i > 0:
                        parsed_output = dependency_parsing(sentence, previous, marker)
                        if parsed_output:
                            s1, s2 = parsed_output

                            # parsed_sentence_pairs[marker]["s1"].append(s1)
                            # parsed_sentence_pairs[marker]["s2"].append(s2)
                            line_to_print = "{}\t{}\t{}\n".format(s1, s2, marker)
                            w.write(line_to_print)

                        if i % args.filter_print_every == 0:
                            logger.info("processed {}".format(i))

    # logger.info('writing files')

    # with open(pjoin(output_dir, "{}_parsed_sentence_pairs.json".format(marker_set_tag)), 'wb') as f:
    #     json.dump(parsed_sentence_pairs, f)

    logger.info('file writing complete')

def dependency_parsing(sentence, previous_sentence, marker):
    return depparse_ssplit(sentence, previous_sentence, marker, lang='sp')

if __name__ == '__main__':
    if args.extract:
        extrat_raw_gigaword()
    elif args.filter:
        collect_raw_sentences(gigaword_es_dir, [gigaword_es_file], "ALL14", SP_DISCOURSE_MARKERS)
    elif args.parse:
        setup_corenlp("sp")
        if args.marker_set_tag=="ALL":
            parse_filtered_sentences(gigaword_es_dir, "ALL14", "ALL14")
        elif args.marker_set_tag=="FIVE":
            parse_filtered_sentences(gigaword_es_dir, "ALL14", "FIVE")
        elif args.marker_set_tag=="EIGHT":
            parse_filtered_sentences(gigaword_es_dir, "ALL14", "EIGHT")


