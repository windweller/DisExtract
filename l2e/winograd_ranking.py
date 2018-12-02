"""
Use Winograd LM ranker to rank
"""

import sys
PATH_TO_TF = '/home/anie/models/research/lm_commonsense/'
sys.path.insert(0, PATH_TO_TF)
PATH_TO_TF = '/home/anie/models/research/'
sys.path.insert(0, PATH_TO_TF)

import lm_commonsense.eval as lm_eval
import lm_commonsense.utils as utils

# L2E needs further processing, but LM1B does not (only need splitting)

# Finish this today

# The interface we need is to get a SCORE (fixed score) for each sentence (candidate) that we send in

# Result Table will look like:
# Percentage of times Chosen among other candidates

