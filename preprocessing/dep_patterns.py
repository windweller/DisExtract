# -*- coding: utf-8 -*-

# grab test cases
ch_dependency_patterns = {
  "然后": [
    { # then
      "POS": "AD",
      "S1": "conj",  # the dep tag S1 -> S2
      "S2": "advmod"  # the dep tag S2 -> marker
    },
    { # then
      "POS": "AD",
      "S1": "dep",  # the dep tag S1 -> S2
      "S2": "advmod"  # the dep tag S2 -> marker
    }
  ],
  "但是": [
    { # but
      "POS": "AD",
      "S1": "conj",  # nsubj
      "S2": "advmod"
    },
  ],
  "也": [
    { # also
      "POS": "AD",
      "S1": "conj",
      "S2": "advmod"
    },
  ],
  "因为": [
    { # because
      "POS": "P",
      "S1": "nmod:prep",
      "S2": "case"
    }
  ]
}

sp_dependency_patterns = {
  "luego": [{"S1": "parataxis", "S2": "advmod", "POS": "rg"}],
  "después": [{"S1": "advcl", "S2": "advmod", "POS": "rg"}],
  "también": [{"S1": "parataxis", "S2": "advmod", "POS": "rg"}],
  "además": [{"S1": "parataxis", "S2": "advmod", "POS": "rg"}],
  "aunque": [{"S1": "advcl", "S2": "mark", "POS": "cs"}],
  # "a pesar de que": [{"S1": "", "S2": "", "POS": ""}],
  "y": [{"S1": "conj", "S2": "cc", "POS": "cc", "flip": True}],
  "porque": [{"S1": "advcl", "S2": "mark", "POS": "cs"}],
  "ya que": [{"S1": "advcl", "S2": "mark", "POS": "rg", "head": "ya"}],
  # "dado que": [{"S1": "", "S2": "", "POS": "", "head": ""}],
  "puesto que": [{"S1": "advcl", "S2": "mark", "POS": "nc0s000", "head": "puesto"}],
  "antes": [{"S1": "advcl", "S2": "advmod", "POS": "rg"}],
  "pero": [{"S1": "conj", "S2": "cc", "POS": "cc", "flip": True}],
  "por ejemplo": [{"S1": "advcl", "S2": "nmod", "POS": "ejemplo"}],
  "sin embargo": [{"S1": "conj", "S2": "cc", "POS": "sp000", "head": "sin"}],
  "si": [{"S1": "advcl", "S2": "mark", "POS": "cs"}],
  "mientras": [{"S1": "advcl", "S2": "mark", "POS": "cs", "alternative": "mientras tanto"}],
  "entonces": [{"S1": "parataxis", "S2": "advmod", "POS": "rg"}],
  "por lo tanto": [{"S1": "parataxis", "S2": "nmod", "POS": "vmis000", "head": "tanto"}],
  "por eso": [{"S1": "advcl", "S2": "nmod", "POS": "pd000000", "head": "eso"}],
  "por lo cual": [{"S1": "parataxis", "S2": "nmod", "POS": "pr000000", "head": "cual"}],
  "en consecuencia": [{"S1": "advcl", "S2": "nmod", "POS": "nc0s000", "head": "consecuencia"}],
  "todavía": [{"S1": "parataxis", "S2": "advmod", "POS": "rg"}],
  "cuando": [{"S1": "advcl", "S2": "mark", "POS": "cs"}]
}

en_dependency_patterns = {
  # S2 ~ S2 head (full S head) ---> connective
  "after": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
    {"S1": "acl", "S2": "mark", "POS": "IN"},
  ],
  "also": [
    {"S1": "advcl", "S2": "advmod", "POS": "RB"},
  ],
  "although": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "and": [
    {"S1": "conj", "S2": "cc", "POS": "CC", "flip": True},
  ],
  "as": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "before": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "but": [
    {"S1": "conj", "S2": "cc", "POS": "CC", "flip": True},
  ],
  "so": [
    # {"S1": "parataxis", "S2": "dep", "POS": "IN", "flip": True},
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
    {"S1": "advcl", "S2": "mark", "POS": "RB"},
  ],
  "still": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
    {"S1": "dep", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
  ],
  "though": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "because": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "however": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB"},
    # {"S1": "ccomp", "S2": "advmod", "POS": "RB"}, ## rejecting in favor of high precision
  ],
  "if": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "meanwhile": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB"},
  ],
  "while": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "for example": [
    {"S1": "parataxis", "S2": "nmod", "POS": "NN", "head": "example"},
  ],
  "then": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
  ],
  "when": [
    {"S1": "advcl", "S2": "advmod", "POS": "WRB"},
  ],
}