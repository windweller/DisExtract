"""
Code adapted from
https://github.com/facebookresearch/InferSent/blob/master/train_nli.py

with minor modifications
"""

import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
from itertools import izip

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_dis, get_batch, build_vocab
from dissent import DisSent
from util import get_optimizer, get_labels

import logging

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--corpus", type=str, default='books_5', help="books_5|books_old_5|books_8|books_all|gw_cn_5|gw_cn_all|gw_es_5|dat")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')

# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--cur_epochs", type=int, default=1)
parser.add_argument("--cur_lr", type=float, default=0.1)
parser.add_argument("--cur_valid", type=float, default=-1e10, help="must set this otherwise resumed model will be saved by default")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_emb", type=float, default=0., help="embedding dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--tied_weights", action='store_true', help="RNN would share weights on both directions")
parser.add_argument("--reload_val", action='store_true', help="Reload the previous best epoch on validation, should be used with tied weights")
parser.add_argument("--char", action='store_true', help="for Chinese we can train on char-level model")
parser.add_argument("--s1", action='store_true', help="training only on S1")
parser.add_argument("--s2", action='store_true', help="training only on S2")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(params.outputdir):
    os.makedirs(params.outputdir)
file_handler = logging.FileHandler("{0}/log.txt".format(params.outputdir))
logging.getLogger().addHandler(file_handler)

# print parameters passed, and all parameters
logger.info('\ntogrep : {0}\n'.format(sys.argv[1:]))
logger.info(params)

"""
Default json file loading
"""
with open(params.hypes, 'rb') as f:
    json_config = json.load(f)

data_dir = json_config['data_dir']
prefix = json_config[params.corpus]
glove_path = json_config['glove_path']

"""
DATA
1. build vocab through BPE
"""
train, valid, test = get_dis(data_dir, prefix, params.corpus)  # this stays the same
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], glove_path)
# batching function needs to be different:
# 1). return x_bpe, y_bpe, y triple

# unknown words instead of map to <unk>, this directly takes them out
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
                                           [word for word in sent.split() if word in word_vec] +
                                           ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300

dis_labels = get_labels(params.corpus)
label_size = len(dis_labels)
