"""
We generate why corpus based on
News Commentary dataset (6k)

This dataset looks better than L2E

Author: Erin
Modified: Allen
"""

import os
import json
import argparse
from os.path import join as pjoin
from os.path import dirname, abspath

import numpy as np
import random

import logging

from preprocessing.parser import depparse_ssplit, setup_corenlp, get_parse, Sentence
import spacy

parser = argparse.ArgumentParser(description='Generate Why Pairs')
parser.add_argument("--data_file", type=str, help="should be the file line by line with S1")
parser.add_argument("--output_file", type=str, help="full path and name")
parser.add_argument("--dry", action="store_true", help="no output to a file, but will generate indices")
parser.add_argument("--gen_indices", type=str, help="full path to a file where it's a list of which "
                                                    "line is degenerate")

args, _ = parser.parse_known_args()

setup_corenlp("en")
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_do(subj_is_plural, past_tense):
    if past_tense == True:
        return "did"
    elif subj_is_plural == True:
        return "do"
    else:
        return "does"


def why(original_sentence):
    original_sentence = original_sentence.lower()

    # replace "'s" with "is" where appropriate
    s = Sentence(get_parse(original_sentence), original_sentence, "en")
    indices_to_replace = [t["index"] for t in s.tokens if t["word"] == "'s" and t["pos"][0] == "V"]
    for i in indices_to_replace:
        s.tokens[i - 1]["word"] = "is"
        s.tokens[i - 2]["after"] = " "
    original_sentence = "".join(t["word"] + t["after"] for t in s.tokens).replace("  ", " ")
    if s.tokens[0]["word"].lower() == "or":
        return ("ERROR: can't transform: starts with 'or'")

    s = Sentence(get_parse(original_sentence), original_sentence, "en")
    s.new_tokens = s.tokens
    doc = nlp(original_sentence)
    lemmas = {token.text.lower(): token.lemma_ for token in doc}
    past_tense = {token.text: token.tag_ == "VBD" for token in doc}

    # 1. Take the root of sentence.
    sentence_indices = [t["index"] for t in s.tokens]
    root_index = [i for i in sentence_indices if "ROOT" in s.find_dep_types(i)][0]
    if s.token(root_index)["pos"][0] == "V":
        root_verb_index = root_index
        # Find where the subject NP starts.
        # First, find any children of the root where the dep is nsubj or nsubjpass
        # and grap the corresponding dependent.
        subj_head_indices = s.find_children(root_index, filter_types=["nsubj", "nsubjpass"])
        if len(subj_head_indices) > 0:
            subj_head_index = subj_head_indices[0]
        else:
            return ("ERROR: can't transform: missing nsubj")
    else:
        # assume the first verb is root (this is a hack)
        verb_indices = [t["index"] for t in s.tokens if t["pos"][0] == "V"]
        if len(verb_indices) > 0:
            root_verb_index = min(verb_indices)
        else:
            return ("ERROR: can't transform: weird root")

        if s.token(root_index)["pos"][0] == "N":
            subj_head_index = root_index
        else:
            # assume the first noun is subj (this is a hack)
            noun_indices = [t["index"] for t in s.tokens if t["pos"][0] == "N"]
            if len(noun_indices) > 0:
                subj_head_index = min(noun_indices)
            else:
                return ("ERROR: can't transform: weird root")

    # Then, get the index where this phrase begins.
    subj_start_index = min(s.get_subordinate_indices([subj_head_index], [subj_head_index]))
    subj_head = s.token(subj_head_index)["word"]
    subj_is_plural = (subj_head.lower() != lemmas[subj_head.lower()])

    if subj_head.lower() == "who":
        s.tokens[subj_head_index - 1]["word"] = "they"
        s.new_tokens = s.tokens

    # 2. Look whether ther's an aux or cop or auxpass dependent of the root,
    aux_types = ["aux", "cop", "auxpass"]
    aux_indices = s.find_children(root_verb_index, filter_types=aux_types)
    if (len(aux_indices) == 0):
        aux_indices = [s.find_children(i, filter_types=aux_types) for i in sentence_indices]
        aux_indices = [item for sublist in aux_indices for item in sublist]
    has_aux = len(aux_indices) > 0
    # If so, take the first of these, and move it before the subject `nsubjpass` or `nsubj`.
    if has_aux:
        first_aux_index = min(aux_indices)
        s.move(first_aux_index, subj_start_index)
    # 3. Otherwise put "do" or "does" or "did" in front of the subject,
    # check root verb's lemma, and past tense -> lemma.
    else:
        # find root verb
        # make it a lemma
        root_verb = s.token(root_verb_index)["word"]
        if (root_verb == "is"):
            s.move(root_verb_index, 1)
        else:
            s.token(root_verb_index)["word"] = lemmas[root_verb.lower()]
            # figre out which of "do" "does" or "did" to use
            s.add({"word": get_do(subj_is_plural, past_tense[root_verb]),
                   "index": subj_start_index, "after": " "})

    # 4. Remove any `RB` (adverbs) in front of the subject (`nsubj` or `nsubjpass`)
    for t in s.new_tokens:
        if "pos" in t.keys() and t["pos"] == "RB":
            s.cut(t["index"])

    # Add "why" at the beginning.
    s.add({"word": "Why", "index": 1, "after": " "})

    # Also, add a question mark.
    last_word = s.new_tokens[-1]["word"]
    if (last_word in ["."]):
        last_word = "?"
    else:
        s.add({"word": "?", "index": len(s.new_tokens) + 1, "after": ""})

    return "".join([x["word"] + x["after"] for x in s.new_tokens])


def process_q(q):
    if q[-1] == ".":
        q = q[:-1]
    if ", ," in q:
        q = q.replace(", ,", ",")

    return q


def transform(s1):
    q = why(s1)
    if "ERROR:" in q:
        return None
    else:
        return process_q(q) + '?'


if __name__ == '__main__':
    skipped_list = []  # 1 corresponds to added; 0 corresponds to not picked
    with open(args.data_file) as f, \
            open(args.output_file, 'w') as f_out:
        for i, line in enumerate(f):
            q = transform(line.strip())
            if q is not None:
                f_out.write(q + '\n')
                skipped_list.append(1)
            else:
                skipped_list.append(0)

            if i % 100 == 0:
                logger.info(i)
