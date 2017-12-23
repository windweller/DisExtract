# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

EN_DISCOURSE_MARKERS = [
    "after",
    "also",
    "although",
    "and",
    "as",
    "because",
    "before",
    "but",
    # "for example",
    # "however",
    "if",
    # "meanwhile",
    "so",
    "still",
    "then",
    "though",
    "when",
    "while"
]
DISCOURSE_MARKER_SET_TAG = "ALL18"  # ALL15 now

EN_FIVE_DISCOURSE_MARKERS = [
    "and",
    "because",
    "but",
    "if",
    "when"
]

EN_EIGHT_DISCOURSE_MARKERS = [
    "and"
    "because"
    "but",
    "if",
    "when",
    "so",
    "though",
    "before"
]

# corenlp server port on localhost
EN_PORT = 12345
CH_PORT = 12346
SP_PORT = 12347

_PAD = b"<pad>" # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

CH_DISCOURSE_MARKERS = [
  "并且",
  "而且",
  "因为",
  "之前",
  "但是",
  "可是",
  "不过",
  "但",
  "如果",
  "因此",
  "所以",
  # "然后": [{"POS": "AD", "S1": "conj", "S2": "advmod"}, {"POS": "AD", "S1": "dep", "S2": "advmod"}],
  "虽然",
  "尽管",
  "当"
]

SP_DISCOURSE_MARKERS = [
    "porque"
]