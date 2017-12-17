# -*- coding: utf-8 -*-

# grab test cases
cn_dependency_patterns = {
  "然后": [{ # then
    "POS": ["AD"],
    "S1": ["conj"],  # the dep tag S1 -> S2
    "S2": ["advmod"]  # the dep tag S2 -> marker
  },
  { # then
    "POS": ["AD"],
    "S1": ["dep"],  # the dep tag S1 -> S2
    "S2": ["advmod"]  # the dep tag S2 -> marker
  }
  ],
  "但是": { # but
    "POS": ["AD"],
    "S1": ["conj"],  # nsubj
    "S2": ["advmod"]
  },
  "也": { # also
    "POS": ["AD"],
    "S1": ["conj"],
    "S2": ["advmod"]
  },
  "因为": { # because
    "POS": ["P"],
    "S1": ["nmod:prep"],
    "S2": ["case"]
  }
}

en_dependency_patterns = {
  "after": {
    "POS": ["IN"],
    "S2": ["mark"], # S2 head (full S head) ---> connective
    "S1": ["advcl", "acl"]
  },
  # "although" : [{"POS": "IN", "S2": "mark", "S1": "advcl"}, {"POS": "IN", "S2": "dep", "S1": "parataxis"}]
  "also": {
    "POS": ["RB"],
    "S2": ["advmod"],
    "S1": ["advcl"]
  },
  "although": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "and": {
    "POS": ["CC"],
    "S2": ["cc"],
    "S1": ["conj"],
    "flip": True
  },
  "as": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "before": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "but": {
    "POS": ["CC"],
    "S2": ["cc"],
    "S1": ["conj"],
    "flip": True
  },
  # "so": {
  #   "POS": "IN",
  #   "S2": "dep",
  #   "S1": ["parataxis"],
  #   "flip": True
  # },
  "so": {
    "POS": ["IN", "RB"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "still": {
    "POS": ["RB"],
    "S2": ["advmod"],
    "S1": ["parataxis", "dep"],
    "acceptable_order": "S1 S2"
  },
  "though": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "because": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "however": {
    "POS": ["RB"],
    "S2": ["advmod"],
    "S1": [
        "parataxis",
        # "ccomp" ## rejecting in favor of high precision
    ]
  },
  "if": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "meanwhile": {
    "POS": ["RB"],
    "S2": ["advmod"],
    "S1": ["parataxis"]
  },
  "while": {
    "POS": ["IN"],
    "S2": ["mark"],
    "S1": ["advcl"]
  },
  "for example": {
    "POS": ["NN"],
    "S2": ["nmod"],
    "S1": ["parataxis"],
    "head": "example"
  },
  "then": {
    "POS": ["RB"],
    "S2": ["advmod"],
    "S1": ["parataxis"],
    "acceptable_order": "S1 S2"
  },
  "when": {
    "POS": ["WRB"],
    "S2": ["advmod"],
    "S1": ["advcl"]
  }
}