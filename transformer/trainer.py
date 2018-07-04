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
import random

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_dis, data_gen, pad_batch, build_vocab
from util import get_labels, TextEncoder, ResultLogger

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
parser.add_argument("--dpout", type=float, default=0.1, help="residual, embedding, attention dropout") # 3 dropouts
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--maxlr", type=float, default=2.5e-4, help="minimum lr")
parser.add_argument("--l2", type=float, default=0.01, help="on non-bias non-gain weights")
parser.add_argument("--max_norm", type=float, default=2., help="max norm (grad clipping). Original paper uses 1.")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument("--train_emb", action='store_true', help="Initialize embedding randomly, and then learn it, default to False")
# for now we fix non-linearity to whatever PyTorch provides...could be SELU

# model
parser.add_argument("--d_ff", type=int, default=3072, help="decoder nhid dimension")
parser.add_argument("--d_model", type=int, default=768, help="decoder nhid dimension")
parser.add_argument("--n_heads", type=int, default=12, help="number of attention heads")
parser.add_argument("--n_layers", type=int, default=8, help="decoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--pool_type", type=str, default='max', help="flag if we do max pooling, which hasn't been done before")
parser.add_argument("--reload_val", action='store_true', help="Reload the previous best epoch on validation, should be used with tied weights")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

"""
SEED
"""
random.seed(params.seed)
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
bpe_encoder_path = json_config['bpe_encoder_path']
bpe_vocab_path = json_config['bpe_vocab_path']
params_path = json_config['params_path']

"""
BPE encoder
"""
text_encoder = TextEncoder(bpe_encoder_path, bpe_vocab_path)
encoder = text_encoder.encoder

# add special token
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)

"""
DATA
1. build vocab through BPE
"""
train, valid, test = get_dis(data_dir, prefix, params.corpus)  # this stays the same

# word_vec = build_vocab(train['s1'] + train['s2'] +
#                        valid['s1'] + valid['s2'] +
#                        test['s1'] + test['s2'], glove_path)
# batching function needs to be different:
# 1). return s1, s2, y_label

# If this is slow...we can speed it up
# Numericalization; No padding here
# Also, Batch class from OpenNMT will take care of target generation
max_len = 0.
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        num_sents = []
        y_sents = []
        for sent in eval(data_type)[split]:
            num_sent = text_encoder.encode([sent], verbose=False, lazy=True)[0]
            num_sents.append([encoder['_start_']] + num_sent + [encoder['_end_']])
            # y_sents.append(num_sent + [encoder['_end_']])
            max_len = max_len if max_len > len(num_sent) + 1 else len(num_sent) + 1
        eval(data_type)[split] = np.array(num_sents)
        # eval(data_type)['y_'+split] = np.array(y_sents)

"""
Params
2. Load in parameters (word embeddings)
"""

shapes = json.load(open(pjoin(params_path, 'params_shapes.json')))
offsets = np.cumsum([np.prod(shape) for shape in shapes])
init_params = [np.load(pjoin(params_path, 'params_{}.npy'.format(n))) for n in range(3)]
init_params = np.split(np.concatenate(init_params, 0), offsets[:2])[:-1]
init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes[:2])]

n_special = 3  # <s>, </s>, <pad>
n_ctx = 512
n_ctx = min(max_len, n_ctx)

init_params[0] = init_params[0][:n_ctx]
word_embeddings = np.concatenate([init_params[1],
                                   np.zeros((1, params.n_embed), np.float32), # pad, zero-value!
                                  (np.random.randn(n_special-1, params.n_embed)*0.02).astype(np.float32)], 0)
ctx_embeddings = init_params[0]
del init_params[1]


dis_labels = get_labels(params.corpus)
label_size = len(dis_labels)

"""
MODEL
"""
# model config
config_dis_model = {
    'n_words': len(encoder),
    'd_model': params.d_model, # same as word embedding size
    'd_ff': params.d_ff, # this is the bottleneck blowup dimension
    'n_layers': params.n_layers,
    'dpout': params.dpout,
    'dpout_fc': params.dpout_fc,
    'fc_dim': params.fc_dim,
    'bsize': params.batch_size,
    'n_classes': label_size,
    'pool_type': params.pool_type,
    'n_heads': params.n_heads,
    'use_cuda': True,
    'train_emb': params.train_emb
}

# TODO: shuffling data happens inside train_epoch

