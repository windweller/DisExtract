EN_DISCOURSE_MARKERS = [
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

EN_FIVE_DISCOURSE_MARKERS = [
    "and",
    "because",
    "but",
    "if",
    "when"
]

_PAD = b"<pad>" # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1