"""
Evaluate on PDTB
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

from data import get_pdtb, get_batch, build_vocab
from util import get_labels

import logging

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--corpus", type=str, default='books_5', help="books_5|books_old_5|books_8|books_all|gw_cn_5|gw_cn_all|gw_es_5|dat")
parser.add_argument("--hypes", type=str, default='hypes/pdtb.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--modeldir", type=str, default='sandbox/', help="Output directory")
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

if params.char and params.corpus == "gw_cn_5":
    prefix = prefix.replace('discourse', 'discourse_char')

"""
DATA
"""
test = get_pdtb(data_dir, prefix, params.corpus)
word_vec = build_vocab(test['s1'] + test['s2'], glove_path)

# unknown words instead of map to <unk>, this directly takes them out
for split in ['s1', 's2']:
    for data_type in ['test']:
        eval(data_type)[split] = np.array([['<s>'] +
                                           [word for word in sent.split() if word in word_vec] +
                                           ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300

dis_labels = get_labels(params.corpus)
label_size = len(dis_labels)

"""
MODEL
"""
# model config
config_dis_model = {
    'n_words': len(word_vec),
    'word_emb_dim': params.word_emb_dim,
    'enc_lstm_dim': params.enc_lstm_dim,
    'n_enc_layers': params.n_enc_layers,
    'dpout_emb': params.dpout_emb,
    'dpout_model': params.dpout_model,
    'dpout_fc': params.dpout_fc,
    'fc_dim': params.fc_dim,
    'bsize': params.batch_size,
    'n_classes': label_size,
    'pool_type': params.pool_type,
    'encoder_type': params.encoder_type,
    'tied_weights': params.tied_weights,
    'use_cuda': True,
}

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn.size_average = False

def get_multiclass_recall(preds, y_label):
    # preds: (label_size), y_label; (label_size)
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of the correct labels
        idx_of_cat = y_label == la
        cat_preds = preds[idx_of_cat]
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def get_multiclass_prec(preds, y_label):
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of predictions
        idx_of_cat = preds == la
        cat_preds = y_label[idx_of_cat]  # ground truth
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def evaluate(epoch, eval_type='test', final_eval=False, save_confusion=False):

    dis_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        logger.info('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = test['s1']
    s2 = test['s2']
    target = test['label']

    valid_preds, valid_labels = [], []

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # we collect samples
        labels = target[i:i + params.batch_size]
        preds = pred.cpu().numpy()

        valid_preds.extend(preds.tolist())
        valid_labels.extend(labels.tolist())

    mean_multi_recall = get_multiclass_recall(np.array(valid_preds), np.array(valid_labels))
    mean_multi_prec = get_multiclass_prec(np.array(valid_preds), np.array(valid_labels))

    multiclass_recall_msg = 'Multiclass Recall - '
    for k, v in mean_multi_recall.iteritems():
        multiclass_recall_msg += dis_labels[k] + ": " + str(v[0]) + " "

    multiclass_prec_msg = 'Multiclass Precision - '
    for k, v in mean_multi_prec.iteritems():
        if len(v) == 0:
            v = [0.]
        multiclass_prec_msg += dis_labels[k] + ": " + str(v[0]) + " "

    logger.info(multiclass_recall_msg)
    logger.info(multiclass_prec_msg)

    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        logger.info('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        logger.info('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
        # print out multi-class recall and precision

    if save_confusion:
        with open(pjoin(params.outputdir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in izip(valid_preds, valid_labels):
                writer.writerow({'preds': pair[0], 'labels': pair[1]})

    return eval_acc


"""
Evaluate model on different tasks
"""
if __name__ == '__main__':

    map_locations = {}
    for d in range(4):
        if d != params.gpu_id:
            map_locations['cuda:{}'.format(d)] = "cuda:{}".format(params.gpu_id)

    if params.cur_epochs != 1:
        model_path = pjoin(params.modeldir, params.outputmodelname + "-{}".format(params.cur_epochs) + ".pickle")  # this is the best model
        dis_net = torch.load(model_path, map_location=map_locations)
    else:
        # this loads in the final model, last epoch
        dis_net = torch.load(os.path.join(params.modeldir, params.outputmodelname + ".pickle"))

    # cuda by default
    dis_net.cuda()
    loss_fn.cuda()

    logger.info('\nEvaluating on {} : Last Epoch'.format(params.hypes[6:-5])) # corpus name
    # evaluate(1e6, 'valid', True)
    evaluate(0, 'test', True, True)  # save confusion results on test data