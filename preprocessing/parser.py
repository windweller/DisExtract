#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Will read all files from a directory, map to discourse markers
initiate dependency parsing
This creates (s1, s2, marker) by each marker
"""

import numpy as np
import argparse
import io
import nltk
import pickle
import requests
import re

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

import json
from itertools import izip

from copy import deepcopy as cp

np.random.seed(123)

DISCOURSE_MARKERS = [
    "after",
    "also",
    "although",
    "and",
    "as",
    "because",
    "before",
    "but",
    "for example",
    "however",
    "if",
    "meanwhile",
    "so",
    "still",
    "then",
    "though",
    "when",
    "while"
]
DISCOURSE_MARKER_SET_TAG = "ALL18"

# patterns = {
#     "because": ("IN", "mark", "advcl"),
# }

def setup_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def sentence_to_token_ids(sentence, vocabulary):
    return [vocabulary.get(w, UNK_ID) for w in sentence]

def undo_rephrase(lst):
    return " ".join(lst).replace("for_example", "for example").split()

def rephrase(str):
    return str.replace("for example", "for_example")

def cleanup(str):
    return str.replace(" @-@ ", "-")

# # not sure if i'll use this actually...
# dangerous_dependencies = ["mark", "advcl", "acl"]

dependency_patterns = {
  "after": {
    "POS": "IN",
    "S2": "mark", # S2 head (full S head) ---> connective
    "S1": ["advcl", "acl"]
  },
  "also": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["advcl"]
  },
  "although": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "and": {
    "POS": "CC",
    "S2": "cc",
    "S1": ["conj"]
  },
  "as": {
    "POS": "IN",
    "S2": "mark",
    "S1": "advcl"
  },
  "before": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "so": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "still": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["parataxis", "dep"]
  },
  "though": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "because": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "however": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["dep", "parataxis"]
  },
  "if": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "while": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  }
}

# search for pattern:
# "[discourse marker] S2, S1" (needs dependency parse)
def search_for_reverse_pattern_pair(sent, marker, words, previous_sentence):
    parse_string = get_parse(sent, depparse=True)

    # book corpus maybe has carriage returns and new lines and other things?
    try: 
        parse = json.loads(parse_string.replace('\r\n', ''))
    except ValueError:
        parse = json.loads(re.sub("[^A-z0-9.,!:?\"'*&/\{\}\[\]()=+-]", "", parse_string))        
    sentence = Sentence(parse["sentences"][0], sent)
    return sentence.find_pair(marker, "s2 discourse_marker s1", previous_sentence)


def is_verb_tag(tag):
    return tag[0] == "V" and not tag[-2:] in ["BG", "BN"]

"""
POS-tag string as if it's a sentence
and see if it has a verb that could plausibly be the predicate.
"""
def has_verb(string):
    parse = get_parse(string, depparse=False)
    tokens = json.loads(parse)["sentences"][0]["tokens"]
    return any([is_verb_tag(t["pos"]) for t in tokens])


"""
using the depparse, look for the desired pattern, in any order
"""
def search_for_dep_pattern(marker, current_sentence, previous_sentence):  
    parse_string = get_parse(current_sentence, depparse=True)

    # book corpus maybe has carriage returns and new lines and other things?
    try: 
        parse = json.loads(parse_string.replace('\r\n', ''))
    except ValueError:
        parse = json.loads(re.sub("[^A-z0-9.,!?:\"'*&/\{\}\[\]()=+-]", "", parse_string))

    sentence = Sentence(parse["sentences"][0], current_sentence)
    return sentence.find_pair(marker, "any", previous_sentence)


# https://stackoverflow.com/a/18669080
def get_indices(lst, element, case="sensitive"):
  result = []
  starting_index = -1
  while True:
    try:
        found_index = lst.index(element, starting_index+1)
        starting_index = found_index
    except ValueError:
        return result
    result.append(found_index)

def get_nearest(lst, element):
    distances = [abs(e-element) for e in lst]
    return lst[np.argmin(distances)]


def separate_at_signs(lst):
    s = " ".join(lst)
    separated_s = re.sub(" @([^ ]+)@ ", " @ \1 @ ", s)
    return separated_s.split()


"""
parsed tokenization is different from original tokenization.
try to re-align and extract the correct words given the
extraction_indices (which are 1-indexed into parsed_words)

fix me to catch more cases?
"""
def extract_subphrase(orig_words, parsed_words, extraction_indices):
    extraction_indices = [i-1 for i in extraction_indices]

    orig_words = separate_at_signs(orig_words)

    if len(orig_words) == len(parsed_words):
        return " ".join([orig_words[i] for i in extraction_indices])
    else:
        first_parse_index = extraction_indices[0]
        first_word_indices = get_indices(orig_words, parsed_words[first_parse_index])

        last_parse_index = extraction_indices[-1]

        last_word_indices = get_indices(orig_words, parsed_words[last_parse_index])

        if len(first_word_indices)>0 and len(last_word_indices)>0:
            first_orig_index = get_nearest(first_word_indices, first_parse_index)
            last_orig_index = get_nearest(last_word_indices, last_parse_index)
            if last_orig_index-first_orig_index == last_parse_index-first_parse_index:
                # maybe it's just shifted
                shift = first_orig_index - first_parse_index
                extraction_indices = [i+shift for i in extraction_indices]
                return " ".join([orig_words[i] for i in extraction_indices])
            else:
                # or maybe there's funny stuff happening inside the subphrase
                # in which case T-T
                return None
        else:
            if len(first_word_indices)>0 and abs(last_parse_index-len(parsed_words))<3:
                # the end of the sentence is always weird. assume it's aligned

                # grab the start of the subphrase
                first_orig_index = get_nearest(first_word_indices, first_parse_index)
                # shift if necessary
                shift = first_orig_index - first_parse_index
                extraction_indices = [i+shift for i in extraction_indices]

                if len(orig_words) > extraction_indices[-1]:
                    # extend to the end of the sentence if we're not already there
                    extraction_indices += range(extraction_indices[-1]+1, len(orig_words))
                else:
                    extraction_indices = [i for i in extraction_indices if i<len(orig_words)]

                return " ".join([orig_words[i] for i in extraction_indices])

            else:
                # or maybe the first and/or last words have been transformed,
                # in which case T-T
                return None
        


"""
use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
to parse sentences: tokens, dependency parse
"""
def get_parse(sentence, depparse=True):
    sentence = sentence.replace("'t ", " 't ")
    if depparse:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    else:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos'}"
    data = sentence
    parse_string = requests.post(url, data=data).text
    return json.loads(parse_string)["sentences"][0]

class Sentence():
    def __init__(self, json_sentence, original_sentence):
        self.json = json_sentence
        self.dependencies = json_sentence["basicDependencies"]
        self.tokens = json_sentence["tokens"]
        self.original_sentence = original_sentence
    def indices(self, word):
        if len(word.split(" ")) > 1:
            words = word.split(" ")
            indices = [i for lst in [self.indices(w) for w in words] for i in lst]
            return indices
        else:
            return [i+1 for i in get_indices([t["word"].lower() for t in self.tokens], word)]
    def token(self, index):
        return self.tokens[int(index)-1]
    def word(self, index):
        return self.token(index)["word"]

    def find_parents(self, index, filter_types=False, needs_verb=False):
        deps = self.find_deps(index, dir="parents", filter_types=filter_types)

        if needs_verb:
            deps = [d for d in deps if self.gov_is_verb(d) or self.dep_is_verb(d)]

        return [d["governor"] for d in deps]

    def find_children(self, index, filter_types=False, exclude_types=False, needs_verb=False):
        deps = self.find_deps(
            index,
            dir="children",
            filter_types=filter_types,
            exclude_types=exclude_types
        )

        if needs_verb:
            deps = [d for d in deps if self.dep_is_verb(d)]

        # print(deps)
        return [d["dependent"] for d in deps]

    def is_verb(self, index):
        pos = self.token(index)["pos"]
        if pos[0] == "V":
            return True
        else:
            cop_relations = self.find_deps(index, dir="children", filter_types="cop")
            has_cop_relation = len(cop_relations)>0
            if has_cop_relation:
                return True
            else:
                return False

    def gov_is_verb(self, d):
        index = d["governor"]
        return self.is_verb(index)

    def dep_is_verb(self, d):
        index = d["dependent"]
        return self.is_verb(index)

    def find_deps(self, index, dir=None, filter_types=False, exclude_types=False):
        deps = []
        if dir=="parents" or dir==None:
            deps += [d for d in self.dependencies if d['dependent']==index]
        if dir=="children" or dir==None:
            deps += [d for d in self.dependencies if d['governor']==index]

        if filter_types:
            deps = [d for d in deps if d["dep"] in filter_types]
        if exclude_types:
            deps = [d for d in deps if not d["dep"] in exclude_types]

        return deps

    def find_dep_types(self, index, dir=None, filter_types=False):
        deps = self.find_deps(index, dir=dir, filter_types=filter_types)
        return [d["dep"] for d in deps]

    def __str__(self):
        return " ".join([t["word"] for t in self.tokens])

    def get_subordinate_indices(self, acc, explore, depth=0, exclude_indices=[], exclude_types=[]):
        # print("acc: {}\nexplore: {}\ndepth: {}\nexclude_indices: {}".format(acc, explore, depth, exclude_indices))
        exclude_indices.sort()
        acc.sort()
        explore.sort()
        # print("exclude: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in exclude_indices]))
        # print("acc: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in acc]))
        # print("explore: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in explore]))
        # print("*****")

        children = [c for i in explore for c in self.find_children(i, exclude_types=exclude_types) if not c in exclude_indices]
        if len(children)==0:
            return acc
        else:
            return self.get_subordinate_indices(
                acc=acc + children,
                explore=children,
                depth=depth+1,
                exclude_indices=exclude_indices,
                exclude_types=exclude_types
            )

    def get_phrase_from_head(self, head_index, exclude_indices=[], exclude_types=[]):
        # given an index,
        # grab every index that's a child of it in the dependency graph
        subordinate_indices = self.get_subordinate_indices(
            acc=[head_index],
            explore=[head_index],
            exclude_indices=exclude_indices,
            exclude_types=exclude_types
        )
        if not subordinate_indices:
            return None
        subordinate_indices.sort()
        
        # make string of subordinate phrase from parse
        parse_subordinate_string = " ".join([self.word(i) for i in subordinate_indices])

        # correct subordinate phrase from parsed version to wikitext version
        # (tokenization systems are different)
        orig_words = self.original_sentence.split()
        parsed_words = [t["word"] for t in self.tokens]

        subordinate_phrase = extract_subphrase(orig_words, parsed_words, subordinate_indices)

        # make a string from this to return
        return subordinate_phrase

    def get_valid_marker_indices(self, marker):
        pos = dependency_patterns[marker]["POS"]
        return [i for i in self.indices(marker) if pos == self.token(i)["pos"] ]

    def get_candidate_S2_indices(self, marker, marker_index, needs_verb=False):
        connection_type = dependency_patterns[marker]["S2"]
        # Look for S2
        return self.find_parents(marker_index, filter_types=[connection_type], needs_verb=needs_verb)

    def get_candidate_S1_indices(self, marker, s2_head_index, needs_verb=False):
        valid_connection_types = dependency_patterns[marker]["S1"]
        return self.find_parents(
            s2_head_index,
            filter_types=valid_connection_types,
            needs_verb=needs_verb
        ) + self.find_children(
            s2_head_index,
            filter_types=valid_connection_types,
            needs_verb=needs_verb
        )

    def find_pair(self, marker, order, previous_sentence):
        assert(order in ["s2 discourse_marker s1", "any"])
        # fix me
        # (this won't quite work if there are multiple matching connections)
        # (which maybe never happens)
        S1 = None
        S2 = None
        s1_ind = 1000
        s2_ind = 0

        for marker_index in self.get_valid_marker_indices(marker):

            for s2_head_index in self.get_candidate_S2_indices(marker, marker_index, needs_verb=True):
                s2_ind = s2_head_index

                possible_S1s = []

                for s1_head_index in self.get_candidate_S1_indices(marker, s2_head_index, needs_verb=True):
                    # store S1 if we have one
                    S1 = self.get_phrase_from_head(
                        s1_head_index,
                        exclude_indices=[s2_head_index]
                    )
                    # we'll lose some stuff here because of alignment between
                    # wikitext tokenization and corenlp tokenization.
                    # if we can't get a phrase, reject this pair
                    if not S1:
                        break

                    # if we are only checking for the "reverse" order, reject anything else
                    if order=="s2 discourse_marker s1":
                        if s1_ind < s2_ind:
                            break

                    possible_S1s.append((s1_head_index, S1))

                # to do: fix this. it is wrong. we're just grabbing the first if there are multiple matches for the S1 pattern rather than choosing in a principled way
                if len(possible_S1s) > 0:
                    s1_ind, S1 = possible_S1s[0]

                # store S2 if we have one
                S2 = self.get_phrase_from_head(
                    s2_head_index,
                    exclude_indices=[marker_index, s1_ind],
                    # exclude_types=dependency_patterns[marker]["S1"]
                )
                # we'll lose some stuff here because of alignment between
                # wikitext tokenization and corenlp tokenization.
                # if we can't get a phrase, reject this pair
                # update: we fixed some of these with the @ correction
                if not S2:
                    return None

        # if S2 is the whole sentence *and* we're missing S1, let S1 be the previous sentence
        if S2 and not S1:
            words_in_sentence = len(self.tokens)
            words_in_s2 = len(S2.split())
            if words_in_sentence - 1 == words_in_s2:
                S1 = previous_sentence

        if S1 and S2:
            return S1, S2
        else:
            return None

def depparse_ssplit(sentence, previous_sentence, marker):
    sentence = cleanup(sentence)
    parse = get_parse(sentence)
    # print(json.dumps(parse, indent=4))
    sentence = Sentence(parse, sentence)
    return(sentence.find_pair(marker, "any", previous_sentence))

if __name__ == '__main__':
    args = setup_args()
    depparse_ssplit()



