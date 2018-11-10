"""
We are not training on this dataset, so no splitting

this file will do:
filtering,
parsing,
producing

three in one
"""

import os
import re
import io
import sys
import json
import gzip
import argparse
import spacy

import logging
import string
from copy import copy
from util import rephrase
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from os.path import dirname, abspath

printable = set(string.printable)

# from cfg import DISCOURSE_MARKER_SET_TAG, EN_BECAUSE_MARKER, \
#     EN_DISCOURSE_MARKERS

parser = argparse.ArgumentParser(description='DisExtract WMT News Crawl 2007-2017')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")

parser.add_argument("--filter", action='store_true',
                    help="Stage 1: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--filter_because", action='store_true',
                    help="Stage 1: run filtering on the corpus, collect sentence pairs (sentence and previous sentence) that has because")
parser.add_argument("--filter_print_every", default=10000, type=int)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--context_len", default=5, type=int,
                    help="we are storing this number of sentences previous to context")

parser.add_argument("--parse", action='store_true',
                    help="Stage 2: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--tag", type=str, default="BECAUSE",
                    help="Discourse tag / also folder name / generated during filter")
parser.add_argument("--exclude_list", action='store_true', help="use exclusion list defined in this file")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")
parser.add_argument("--delay_print", action='store_true', help="not caching dependency parsed result")

# needs to add tokenization to this
parser.add_argument("--produce", action='store_true',
                    help="Stage 3: producing src/tgt files, with and without context")
parser.add_argument("--out_prefix", type=str)
parser.add_argument("--max_ratio", default=5.0, type=float)
parser.add_argument("--data_dir", type=str, default='default', help="the path for the data file")
parser.add_argument("--corpus", type=str, default='news_commentary')

args, _ = parser.parse_known_args()
en_nlp = spacy.load("en_core_web_sm")

args.min_ratio = 1 / args.max_ratio  # auto-generate min-ratio

# ======== Data Path =========
if args.data_dir == "default":
    root_dir = dirname(dirname(abspath(__file__)))
    args.data_dir = pjoin(root_dir, "data", args.corpus)

if not os.path.isdir(args.data_dir):
    os.mkdir(args.data_dir)

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

with open(args.json, 'rb') as f:
    json_config = json.load(f)

news_commentary_en_dir = './corpus/news_commentary/'
news_commentary_en_file = 'news-commentary-v13.en'


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

    # "before" is where we grab context
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
                        sentences[marker]["before"].append(copy(before_list))  # add list of context

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

            stored_parsed_res = []
            for marker, slists in sentences.iteritems():
                i = 0
                # the set will remove the same row
                for sentence, previous, ctx in zip(slists["sentence"], slists["previous"], slists["before"]):
                    i += 1

                    if not args.delay_print:
                        parsed_output = dependency_parsing(sentence, previous, marker)
                        if parsed_output:
                            s1, s2 = parsed_output
                            ctx_s = " ".join(ctx).replace('\n', '')

                            # parsed_sentence_pairs[marker]["s1"].append(s1)
                            # parsed_sentence_pairs[marker]["s2"].append(s2)
                            line_to_print = "{}\t{}\t{}\t{}\n".format(ctx_s, s1, s2, marker)
                            w.write(line_to_print)
                    else:
                        parsed_output = dependency_parsing(sentence, previous, marker)
                        if parsed_output:
                            s1, s2 = parsed_output
                            ctx_s = " ".join(ctx).replace('\n', '')

                            stored_parsed_res.append((ctx_s, s1, s2, marker))

                    if i % args.filter_print_every == 0:
                        logger.info("processed {}".format(i))

            logger.info("start writing to file")
            for tup in stored_parsed_res:
                ctx_s, s1, s2, marker = tup
                line_to_print = "{}\t{}\t{}\t{}\n".format(ctx_s, s1, s2, marker)
                w.write(line_to_print)

    logger.info('file writing complete')


def dependency_parsing(sentence, previous_sentence, marker):
    return depparse_ssplit(sentence, previous_sentence, marker, lang='en')


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


def write_to_opennmt(data, out_prefix, split_name):
    with open(pjoin(args.data_dir, '{}-ctx-src-{}.txt'.format(out_prefix, split_name)), 'w') as ctx_src, \
            open(pjoin(args.data_dir, '{}-src-{}.txt'.format(out_prefix, split_name)), 'w') as src, \
            open(pjoin(args.data_dir, '{}-tgt-{}.txt'.format(out_prefix, split_name)), 'w') as tgt:
        for line in data:
            ctx, s1, s2, label = line.strip().split('\t')  # need to remove '\n'

            ctx = filter(lambda x: x in printable, ctx)
            s1 = filter(lambda x: x in printable, s1)
            s2 = filter(lambda x: x in printable, s2)

            ctx_src.write(ctx + " || " + ' <Q> ' + fix_tok(s1) + '\n')
            src.write(fix_tok(s1) + '\n')  # we have one-to-one map now
            tgt.write(fix_tok(s2) + '\n')


def tokenize(sentence):
    tokens = en_nlp.tokenizer(unicode(sentence.strip()))
    words = [str(token) for token in tokens if not str(token).isspace()]

    return ' '.join(words)


def produce(source_dir, marker_set_tag):
    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    output_dir = pjoin(markers_dir, "parsed_sentence_pairs")
    #

    examples = []
    total_num = 0.
    check_repeat = set()

    with open(pjoin(output_dir, "{}_parsed_sentence_pairs.txt".format(marker_set_tag)), 'r') as f:
        for line in f:
            total_num += 1
            if line not in check_repeat:
                check_repeat.add(line)
                examples.append(line)

    print("original {}, found repeat {}".format(len(examples), total_num - len(examples)))

    del check_repeat  # release memory

    number_of_filtered_examples = 0
    new_examples = []

    with open(pjoin(output_dir, "{}_parsed_sentence_pairs.txt".format(marker_set_tag)), 'r') as f:
        for ex in f:
            if len(ex[:-1].split('\t')) > 4:
                print("tab sep was not cleaned in the filtering...")
                continue

            ctx_s, s1, s2, label = ex[:-1].split('\t')

            s1 = tokenize(s1)
            s2 = tokenize(s2)

            ctx_s = tokenize(ctx_s)

            s1_len = len(s1.split())
            s2_len = len(s2.split())

            ex = '\t'.join([ctx_s, s1, s2, label])

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
        "original number: {}, filtered out number: {}".format(len(examples),
                                                              len(examples) - number_of_filtered_examples))

    write_to_opennmt(new_examples, args.out_prefix, 'test')

    print("test number of examples: {}".format(len(new_examples)))


if __name__ == '__main__':
    if args.filter_because:
        collect_raw_sentences(news_commentary_en_dir, [news_commentary_en_file], "BECAUSE", ['because'])
    # elif args.filter:
    #     collect_raw_sentences(newscrawl_en_dir, [newscrawl_en_file], DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS)
    elif args.parse:
        setup_corenlp("en")
        parse_filtered_sentences(news_commentary_en_dir, args.tag)  # "BECAUSE"
    elif args.produce:
        produce(news_commentary_en_dir, args.tag)
