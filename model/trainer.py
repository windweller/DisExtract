"""
Code adapted from https://github.com/facebookresearch/InferSent/blob/master/train_nli.py
"""

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet

