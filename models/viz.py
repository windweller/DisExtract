"""
Generate visualization for the model
methods accept data passed in from outside
"""
import numpy as np
import itertools
import torch
from data import get_batch
from torch.autograd import Variable


def collect_type_errors(dis_net, data, word_vec, target_marker_id, batch_size=512):
    """
    :param dis_net: Model
    :param data: Either valid or test or combined, should be a dictionary
    :param word_vec: obtained after executing `build_vocab()` method
    :return: (type1_list, type2_list)
    """
    dis_net.eval()

    # it will only be "valid" during retraining (fine-tuning)
    s1 = data['s1']
    s2 = data['s2']  # if eval_type == 'valid' else test['s2']
    target = data['label']

    # valid_preds, valid_labels = [], []
    type_one_list, type_two_list = [], []
    num_pred_made = 0.
    num_target_marker = 0.

    for i in range(0, len(s1), batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + batch_size])).cuda()

        # model forward
        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        # correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # we collect samples
        labels = target[i:i + batch_size]
        preds = pred.cpu().numpy()

        # analyze and collect Type I and Type II
        counter = 0
        for p, l in itertools.izip(preds.tolist(), labels.tolist()):
            # false positive, Type I error
            if p == target_marker_id and l != target_marker_id:
                type_one_list.append([s1[i + counter], s2[i + counter], p, l])
            elif p != target_marker_id and l == target_marker_id:
                type_two_list.append([s1[i + counter], s2[i + counter], p, l])
            counter += 1

            if p == target_marker_id:
                num_pred_made += 1
            if l == target_marker_id:
                num_target_marker += 1

        if i % 100 == 0:
            print("processed {}".format(i))

    return type_one_list, type_two_list, num_pred_made, num_target_marker


def visualize(dis_net, sentence):
    pass
