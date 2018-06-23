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

from data import get_merged_data, get_batch, build_vocab, get_dis
from util import get_labels, get_optimizer

import logging

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--corpus", type=str, default='books_5',
                    help="books_5|books_old_5|books_8|books_all|gw_cn_5|gw_cn_all|gw_es_5|dat")
parser.add_argument("--hypes", type=str, default='hypes/pdtb.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--modeldir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')

# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--cur_epochs", type=int, default=1)
parser.add_argument("--cur_lr", type=float, default=0.1)
parser.add_argument("--cur_valid", type=float, default=-1e10,
                    help="must set this otherwise resumed model will be saved by default")

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
parser.add_argument("--retrain", action='store_true', help="Retrain the last classifier/decoder on new corpora")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--tied_weights", action='store_true', help="RNN would share weights on both directions")
parser.add_argument("--reload_val", action='store_true',
                    help="Reload the previous best epoch on validation, should be used with tied weights")
parser.add_argument("--char", action='store_true', help="for Chinese we can train on char-level model")
parser.add_argument("--distance", type=str, default='l2', help="{l2|l2_norm|dot|cos|nonlinear}"
                                                               "l2: distance is unnormalized euclidean distance; negative distance|"
                                                               "l2_norm: normalized L2 distance |"
                                                               "dot: use unnormalized dot product similarity|"
                                                               "cos: distance is negative of 1 - normalized dot product similarity|"
                                                               "nonlinear: use a relation network, unknown distance metric")

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


def get_target_within_batch(target):
    labels_in_batch = np.unique(target).tolist()
    target_to_label_map = dict(izip(range(len(labels_in_batch)), labels_in_batch))
    label_to_target_map = dict(izip(labels_in_batch, range(len(labels_in_batch))))
    # fastest approach: https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
    vf = np.vectorize(label_to_target_map.get)
    mapped_target = vf(target)  # faster for the entire dataset
    return mapped_target, len(labels_in_batch), target_to_label_map


def get_target_to_batch_idx(mapped_target, num_uniq_tgt):
    # return: {0: [1, 5, 6], 1: [2, 3], ...}
    # map label index to data point indices inside a batch
    label_to_batch_idx = {}
    for i in xrange(num_uniq_tgt):
        label_to_batch_idx[i] = np.where(mapped_target == i)[0]  # numpy. PyTorch advanced indexing is fine
    return label_to_batch_idx


"""
DATA
"""
if not params.retrain:
    test = get_merged_data(data_dir, prefix, params.corpus)
    word_vec = build_vocab(test['s1'] + test['s2'], glove_path)

    # unknown words instead of map to <unk>, this directly takes them out
    for split in ['s1', 's2']:
        for data_type in ['test']:
            eval(data_type)[split] = np.array([['<s>'] +
                                               [word for word in sent.split() if word in word_vec] +
                                               ['</s>'] for sent in eval(data_type)[split]])
else:
    train, valid, test = get_dis(data_dir, prefix, params.corpus)
    word_vec = build_vocab(train['s1'] + train['s2'] +
                           valid['s1'] + valid['s2'] +
                           test['s1'] + test['s2'], glove_path)

    # unknown words instead of map to <unk>, this directly takes them out
    for split in ['s1', 's2']:
        for data_type in ['train', 'valid', 'test']:
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
    'distance': params.distance
}

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        # classifier
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']

        self.inputdim = 5 * 2 * self.enc_lstm_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, self.fc_dim),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.n_classes)
        )

    def forward(self, features):
        output = self.classifier(features)
        return output


"""
TRAIN
"""
val_acc_best = -1e10 if params.cur_epochs == 1 else params.cur_valid
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    logger.info('\nTRAINING : Epoch ' + str(epoch))
    dis_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    label = train['label'][permutation]
    # target = train['label'][permutation]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1 \
                                                                                        and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']
    logger.info('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        batch_size = len(s1[stidx:stidx + params.batch_size])

        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        k = s1_batch.size(1)  # actual batch size

        target, num_uniq_tgts, _ = get_target_within_batch(label[stidx:stidx + params.batch_size])
        tgt_batch = Variable(torch.LongTensor(target)).cuda()

        target_to_batch_idx = get_target_to_batch_idx(target, num_uniq_tgts)

        # model forward
        # u = dis_net.encoder((s1_batch, s1_len))
        # v = dis_net.encoder((s2_batch, s2_len))
        #
        # features = torch.cat((u, v, u - v, u * v, (u + v) / 2.), 1).detach()
        #
        # output = classifier(features)

        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len), target_to_batch_idx, num_uniq_tgts, batch_size)

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in dis_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr']  # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == params.log_interval:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                stidx, round(np.mean(all_costs), 2),
                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                int(words_count * 1.0 / (time.time() - last_time)),
                round(100. * correct / (stidx + k), 2)))
            logger.info(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct / len(s1), 2)
    logger.info('results : epoch {0} ; mean accuracy train : {1}'
                .format(epoch, train_acc))
    return train_acc


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

    # it will only be "valid" during retraining (fine-tuning)
    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    tgt_label = valid['label'] if eval_type == 'valid' else test['label']

    # target = valid['label'] if eval_type == 'valid' else test['label']

    valid_preds, valid_labels = [], []

    for i in range(0, len(s1), params.batch_size):

        batch_size = len(s1[i:i + params.batch_size])

        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())

        # model forward
        # output = dis_net((s1_batch, s1_len), (s2_batch, s2_len))

        target, num_uniq_tgts, target_to_label_map = get_target_within_batch(tgt_label[i:i + params.batch_size])
        tgt_batch = Variable(torch.LongTensor(target)).cuda()

        target_to_batch_idx = get_target_to_batch_idx(target, num_uniq_tgts)

        # model forward
        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len), target_to_batch_idx, num_uniq_tgts, batch_size)

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # we collect samples
        labels = tgt_label[i:i + params.batch_size]  # actual real bona-fide labels
        tgt_preds = pred.cpu().numpy()
        preds = map(target_to_label_map.get, tgt_preds)  # map back

        valid_preds.extend(preds)
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

    # save model, anneal learning rate, etc.
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            logger.info('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(dis_net, os.path.join(params.outputdir,
                                             params.outputmodelname + ".pickle"))
            # monitor memory usage
            try:
                torch.save(dis_net, os.path.join(params.outputdir,
                                                 params.outputmodelname + "-" + str(epoch) + ".pickle"))
            except:
                print("saving by epoch error, maybe due to disk space limit")

            val_acc_best = eval_acc
        else:
            # can reload previous best model
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                logger.info('Shrinking lr by : {0}. New lr = {1}'
                            .format(params.lrshrink,
                                    optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True

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
        model_path = pjoin(params.modeldir, params.outputmodelname + "-{}".format(
            params.cur_epochs) + ".pickle")  # this is the best model
        dis_net = torch.load(model_path, map_location=map_locations)
    else:
        # this loads in the final model, last epoch
        dis_net = torch.load(os.path.join(params.modeldir, params.outputmodelname + ".pickle"))

    if params.retrain:
        # freeze dis_net encoder params..hopefully this works
        for p in dis_net.encoder.parameters():
            p.requires_grad = False

        optimizer = optim_fn(dis_net.classifier.parameters(), **optim_params)

    # cuda by default
    dis_net.cuda()
    loss_fn.cuda()

    if not params.retrain:
        logger.info('\nEvaluating on {} : Last Epoch'.format(params.hypes[6:-5]))  # corpus name
        # evaluate(1e6, 'valid', True)
        evaluate(0, 'test', True, True)  # save confusion results on test data
    else:
        epoch = params.cur_epochs  # start at 1

        while not stop_training and epoch <= params.n_epochs:
            train_acc = trainepoch(epoch)
            eval_acc = evaluate(epoch, 'valid')
            epoch += 1

        # Run best model on test set.
        del dis_net
        dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))

        evaluate(1e6, 'valid', True)
        logger.info('\nTEST : Epoch {0}'.format(epoch))
        evaluate(0, 'test', True, True)
        # the last model is already saved, saving encoder means nothing
        # once retrain is done, we can just call normal evaluate.py to evaluate full model