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

EN_OLD_FIVE_DISCOURSE_MARKERS = [
    "so",
    "but",
    "because",
    "if",
    "when"
]

EN_EIGHT_DISCOURSE_MARKERS = [
    "and",
    "because",
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
  # "然后",
  "虽然",
  "尽管",
  "当"
]

CH_FIVE_DISCOURSE_MARKERS = [
    "and",
    "because",
    "but",
    "if",
    "when"
]

CH_EIGHT_DISCOURSE_MARKERS = [
    "and",
    "because",
    "but",
    "if",
    "when"
]

SP_FIVE_DISCOURSE_MARKERS = [
    "y",
    "porque",
    "pero",
    "si",
    "cuando"
]

SP_EIGHT_DISCOURSE_MARKERS = [
    "y",
    "porque",
    "pero",
    "si",
    "cuando",
    ### SO
    # "entonces",
    # "por eso",
    # "por lo cual",
    ### THOUGH
    "aunque",
    ### BEFORE
    "antes"
]


SP_DISCOURSE_MARKERS = [
  "y",
  "pero",
  "porque",
  "si",
  "cuando",
  "entonces",
  "por eso",
  "por lo cual",
  "aunque",
  "antes",
  "después",
  "mientras",
  "también",
  "por ejemplo",
  "además"
  # "luego",
  # # "a pesar de que",
  # # "ya que",
  # # "dado que",
  # "puesto que",
  # "sin embargo",
  # "por lo tanto",
  # # "en consecuencia",
  # "todavía"
]
