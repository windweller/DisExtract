"""
Uses NSP and MLM training
"""

import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
from itertools import izip
from sklearn import metrics

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_data, get_mlm_batch, build_vocab, get_nsp_data
from dissent import DisMLM
from util import get_optimizer

import logging

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--hypes", type=str, default='hypes/mlm.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')

# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--cur_epochs", type=int, default=1)
parser.add_argument("--cur_lr", type=float, default=0.1)
parser.add_argument("--cur_valid", type=float, default=-1e10,
                    help="must set this otherwise resumed model will be saved by default")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_predictions_per_seq", type=int, default=20)  # drop 20 words at most
parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="prob of noising")
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_emb", type=float, default=0., help="embedding dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")
parser.add_argument("--run_checks", action='store_true', help="Run assertion checks")

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
parser.add_argument("--s1", action='store_true', help="training only on S1")
parser.add_argument("--s2", action='store_true', help="training only on S2")
parser.add_argument("--train_markers", type=str, default='', help="allow to only select a subset of markers "
                                                                  "can select books_5, books_8")

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
glove_path = json_config['glove_path']

"""
DATA
"""
# we load data like this to build vocab, we will not use this data for training
train, valid, test = get_data(data_dir)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], glove_path)

params.word_emb_dim = 300

# this will become the output/label matrix
vocab_emb_np = np.zeros((len(word_vec), params.word_emb_dim))
vocab_list = []
vocab_dict = {}
for word, emb in word_vec.iteritems():
    vocab_emb_np[len(vocab_list), :] = emb
    vocab_dict[word] = len(vocab_list)
    vocab_list.append(word)

# unknown words instead of map to <unk>, this directly takes them out
# for split in ['s1', 's2']:
#     for data_type in ['train', 'valid', 'test']:
#         eval(data_type)[split] = np.array([['<s>'] +
#                                            [word for word in sent.split() if word in word_vec] +
#                                            ['</s>'] for sent in eval(data_type)[split]])

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
    'n_classes': 2,
    'pool_type': params.pool_type,
    'encoder_type': params.encoder_type,
    'tied_weights': params.tied_weights,
    'use_cuda': True,
    's1_only': params.s1,
    's2_only': params.s2
}

if params.cur_epochs == 1:
    dis_net = DisMLM(config_dis_model, vocab_emb_np)
    logger.info(dis_net)
else:
    # if starting epoch is not 1, we resume training
    # 1. load in model
    # 2. resume with the previous learning rate
    model_path = pjoin(params.outputdir, params.outputmodelname + ".pickle")  # this is the best model
    # this might have conflicts with gpu_idx...
    dis_net = torch.load(model_path)

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn.size_average = False

mlm_loss_fn = torch.nn.AdaptiveLogSoftmaxWithLoss(config_dis_model['enc_lstm_dim'] * 2, len(word_vec),
                                                  [int(round(len(word_vec) / 15)), int(3 * round(len(word_vec) / 15))])

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn([p for p in dis_net.parameters() if p.requires_grad], **optim_params)

if params.cur_epochs != 1:
    optimizer.param_groups[0]['lr'] = params.cur_lr

# cuda by default
dis_net.cuda()
loss_fn.cuda()
mlm_loss_fn.cuda()

"""
TRAIN
"""
val_acc_best = -1e10 if params.cur_epochs == 1 else params.cur_valid
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


# TODO: 1). MLM loss; 2). NSP loss
def trainepoch(epoch, train):
    logger.info('\nTRAINING : Epoch ' + str(epoch))
    dis_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    mlm_correct = 0.

    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1 \
                                                                                        and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']
    logger.info('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch, prep MLM
        s1_batch, s1_len, s1_masked_pos, s1_lm_targets = get_mlm_batch(s1[stidx:stidx + params.batch_size],
                                                                       word_vec, params, vocab_list, vocab_dict,
                                                                       flatten_targets=True)
        s2_batch, s2_len, s2_masked_pos, s2_lm_targets = get_mlm_batch(s2[stidx:stidx + params.batch_size],
                                                                       word_vec, params, vocab_list, vocab_dict,
                                                                       flatten_targets=True)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())

        # s1_masked_pos: [batch_size, num_predictions]
        # s1_lm_targets: flattened

        # MLM
        s1_lm_targets = Variable(torch.LongTensor(s1_lm_targets)).cuda()
        s2_lm_targets = Variable(torch.LongTensor(s2_lm_targets)).cuda()

        # NSP
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()

        k = s1_batch.size(1)  # actual batch size

        # model forward
        output, s1_mlm_feats, s2_mlm_feats = dis_net((s1_batch, s1_len), (s2_batch, s2_len), s1_masked_pos,
                                                     s2_masked_pos)  # , s1_lm_targets, s2_lm_targets
        # [batch_size, 2], [bath_size * num_preds, |Vocab|]

        s1_mlm_res = mlm_loss_fn(s1_mlm_feats, s1_lm_targets)
        s2_mlm_res = mlm_loss_fn(s2_mlm_feats, s2_lm_targets)

        # MLM correct
        s1_mlm_pred = mlm_loss_fn.predict(s1_mlm_feats)  # s1_mlm_logits.data.max(1)[1] # s1_mlm_logits.data.max(1)[1]
        s2_mlm_pred = mlm_loss_fn.predict(s2_mlm_feats)  # s2_mlm_logits.data.max(1)[1] # s2_mlm_logits.data.max(1)[1]

        mlm_correct += 0.5 * (s1_mlm_pred.long().eq(s1_lm_targets.data.long()).cpu().sum() + \
                              s2_mlm_pred.long().eq(s2_lm_targets.data.long()).cpu().sum())

        # mlm_correct += 0.5 * (s1_mlm_pred.long().eq(s1_new_targets.data.long()).cpu().sum() + \
        #                       s2_mlm_pred.long().eq(s2_new_targets.data.long()).cpu().sum())

        # NSP correct
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # if params.run_checks:
        #     assert s1_mlm_logits.size(0) == s1_lm_targets.size(0) and \
        #            s2_mlm_logits.size(0) == s2_lm_targets.size(0)

        # MLM loss
        # mlm_loss = loss_fn(s1_mlm_logits, s1_new_targets) + loss_fn(s2_mlm_logits, s2_new_targets)
        mlm_loss = s1_mlm_res.loss + s2_mlm_res.loss

        # NSP loss
        nsp_loss = loss_fn(output, tgt_batch)

        loss = mlm_loss + nsp_loss

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
            logs.append(
                '{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; NSP accuracy train : {4}; MLM accuracy train : {5}'.format(
                    stidx, round(np.mean(all_costs), 2),
                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)),
                    round(100. * correct / (stidx + k), 2), round(100. * mlm_correct / (stidx + k), 2))
            )
            logger.info(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_nsp_acc = round(100 * correct / len(s1), 2)
    train_mlm_acc = round(100 * mlm_correct / len(s1), 4)

    logger.info('results : epoch {0} ; mean NSP accuracy train : {1}; mean MLM accuracy train : {2}'
                .format(epoch, train_nsp_acc, train_mlm_acc))

    return train_nsp_acc, train_mlm_acc


def evaluate(epoch, eval_data, is_test=False, final_eval=False, save_confusion=False):
    # this function also saves model and/or reloads model

    global dis_net

    dis_net.eval()
    correct = 0.
    mlm_correct = 0.

    global val_acc_best, lr, stop_training, adam_stop

    eval_type = "VALIDATION" if not is_test else "TEST"

    logger.info('\n {0} : Epoch {1}'.format(eval_type, epoch))

    s1 = eval_data['s1']
    s2 = eval_data['s2']
    target = eval_data['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        # prepare batch, prep MLM
        s1_batch, s1_len, s1_masked_pos, s1_lm_targets = get_mlm_batch(s1[i:i + params.batch_size],
                                                                       word_vec, params, vocab_list, vocab_dict,
                                                                       flatten_targets=True)
        s2_batch, s2_len, s2_masked_pos, s2_lm_targets = get_mlm_batch(s2[i:i + params.batch_size],
                                                                       word_vec, params, vocab_list, vocab_dict,
                                                                       flatten_targets=True)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())

        # MLM
        s1_lm_targets = Variable(torch.LongTensor(s1_lm_targets)).cuda()
        s2_lm_targets = Variable(torch.LongTensor(s2_lm_targets)).cuda()

        # NSP
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        # output, s1_mlm_logits, s2_mlm_logits = dis_net((s1_batch, s1_len), (s2_batch, s2_len), s1_masked_pos,
        #                                                s2_masked_pos)
        output, s1_mlm_feats, s2_mlm_feats = dis_net((s1_batch, s1_len), (s2_batch, s2_len), s1_masked_pos,
                                                       s2_masked_pos)

        # MLM correct
        # s1_mlm_pred = s1_mlm_logits.data.max(1)[1]
        # s2_mlm_pred = s2_mlm_logits.data.max(1)[1]

        s1_mlm_pred = mlm_loss_fn.predict(s1_mlm_feats)
        s2_mlm_pred = mlm_loss_fn.predict(s2_mlm_feats)

        mlm_correct += 0.5 * (s1_mlm_pred.long().eq(s1_lm_targets.data.long()).cpu().sum() +
                              s2_mlm_pred.long().eq(s2_lm_targets.data.long()).cpu().sum())

        # NSP correct
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = np.round(100 * (correct / float(len(s1))), 2)
    mlm_eval_acc = np.round(100 * (mlm_correct / float(len(s1))), 4)
    # eval_acc = np.round(metrics.accuracy_score(valid_labels, valid_preds) * 100, 2)
    if final_eval:
        logger.info('finalgrep {0} : NSP accuracy : {1}; MLM accuracy : {2}'.format(eval_type, eval_acc, mlm_eval_acc))
    else:
        logger.info('togrep : results : epoch {0}; {1} ; mean NSP accuracy :'
                    '{2} ; mean MLM accuracy : {3}'.format(epoch, eval_type, eval_acc, mlm_eval_acc))
        # print out multi-class recall and precision

    # there is no re-loading of previous best model btw
    # we want to anneal based on MLM, which is harder
    if eval_type == 'VALIDATION' and epoch <= params.n_epochs:
        if mlm_eval_acc > val_acc_best:
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

            val_acc_best = mlm_eval_acc
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

            # now we finished annealing, we can reload
            if params.reload_val:
                del dis_net
                dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))
                logger.info("Load in previous best epoch")
    return eval_acc, mlm_eval_acc


"""
Train model on Discourse Classification task
"""
if __name__ == '__main__':
    epoch = params.cur_epochs  # start at 1

    while not stop_training and epoch <= params.n_epochs:
        train, valid, test = get_nsp_data(data_dir, params)

        # turn them into numpy array
        for split in ['s1', 's2']:
            for data_type in ['train', 'valid', 'test']:
                eval(data_type)[split] = np.array([['<s>'] +
                                                   [word for word in sent.split() if word in word_vec] +
                                                   ['</s>'] for sent in eval(data_type)[split]])

        train_acc = trainepoch(epoch, train)
        eval_acc = evaluate(epoch, valid)
        epoch += 1

    # Run best model on test set.
    del dis_net
    dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))

    logger.info('\nTEST : Epoch {0}'.format(epoch))
    evaluate(1e6, valid, True)
    evaluate(0, test, True, True)  # save confusion results on test data

    # Save encoder instead of full model
    torch.save(dis_net.encoder,
               os.path.join(params.outputdir, params.outputmodelname + ".pickle" + '.encoder'))
