"""
Generate visualization for the model
methods accept data passed in from outside
"""
import numpy as np
import torch
from data import get_batch
from torch.autograd import Variable

def evaluate(dis_net, data, word_vec, filter_target, batch_size=32):
    """
    :param dis_net: Model
    :param data: Either valid or test or combined, should be a dictionary
    :param word_vec: obtained after executing `build_vocab()` method
    :return: (type1_list, type2_list)
    """
    dis_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    # it will only be "valid" during retraining (fine-tuning)
    s1 = data['s1']
    s2 = data['s2'] # if eval_type == 'valid' else test['s2']
    target = data['label']

    valid_preds, valid_labels = [], []

    for i in range(0, len(s1), batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + batch_size])).cuda()

        # model forward
        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # we collect samples
        labels = target[i:i + batch_size]
        preds = pred.cpu().numpy()

        valid_preds.extend(preds.tolist())
        valid_labels.extend(labels.tolist())

    return