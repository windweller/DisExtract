"""
Generate visualization for the model
methods accept data passed in from outside
"""
import numpy as np
import itertools
import torch
from scipy.special import expit as sigmoid
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
    correct_list, type_one_list, type_two_list = [], [], []
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
            elif p == l:
                correct_list.append([s1[i + counter], s2[i + counter], p, l])
            counter += 1

            if p == target_marker_id:
                num_pred_made += 1
            if l == target_marker_id:
                num_target_marker += 1

        if i % 100 == 0:
            print("processed {}".format(i))

    return correct_list, type_one_list, type_two_list, num_pred_made, num_target_marker


# propagate a three-part
def propagate_three(a, b, c, activation):
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


# propagate tanh nonlinearity
def propagate_tanh_two(a, b):
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))


# adapted from github acd
# start=0, stop=len(sent) will work

class CDLSTM(object):
    def __init__(self, model):
        self.model = model.encoder
        weights = model.encoder.enc_lstm.state_dict()

        self.hidden_dim = model.encoder.enc_lstm_dim

        self.W_ii, self.W_if, self.W_ig, self.W_io = np.split(weights['weight_ih_l0'], 4, 0)
        self.W_hi, self.W_hf, self.W_hg, self.W_ho = np.split(weights['weight_hh_l0'], 4, 0)
        self.b_i, self.b_f, self.b_g, self.b_o = np.split(weights['bias_ih_l0'].numpy() + weights['bias_hh_l0'].numpy(), 4)

        # TODO: pre-multiply these matrices together first!!!
        self.W_out = model.classifier.weight.data

    def cd_text(self, batch, start, stop):
        word_vecs = self.model.embed(batch.text)[:, 0].data
        T = word_vecs.size(0)
        relevant = np.zeros((T, self.hidden_dim))
        irrelevant = np.zeros((T, self.hidden_dim))
        relevant_h = np.zeros((T, self.hidden_dim))
        irrelevant_h = np.zeros((T, self.hidden_dim))
        for i in range(T):
            if i > 0:
                prev_rel_h = relevant_h[i - 1]
                prev_irrel_h = irrelevant_h[i - 1]
            else:
                prev_rel_h = np.zeros(self.hidden_dim)
                prev_irrel_h = np.zeros(self.hidden_dim)

            rel_i = np.dot(self.W_hi, prev_rel_h)
            rel_g = np.dot(self.W_hg, prev_rel_h)
            rel_f = np.dot(self.W_hf, prev_rel_h)
            rel_o = np.dot(self.W_ho, prev_rel_h)
            irrel_i = np.dot(self.W_hi, prev_irrel_h)
            irrel_g = np.dot(self.W_hg, prev_irrel_h)
            irrel_f = np.dot(self.W_hf, prev_irrel_h)
            irrel_o = np.dot(self.W_ho, prev_irrel_h)

            if i >= start and i <= stop:
                rel_i = rel_i + np.dot(self.W_ii, word_vecs[i])
                rel_g = rel_g + np.dot(self.W_ig, word_vecs[i])
                rel_f = rel_f + np.dot(self.W_if, word_vecs[i])
                rel_o = rel_o + np.dot(self.W_io, word_vecs[i])
            else:
                irrel_i = irrel_i + np.dot(self.W_ii, word_vecs[i])
                irrel_g = irrel_g + np.dot(self.W_ig, word_vecs[i])
                irrel_f = irrel_f + np.dot(self.W_if, word_vecs[i])
                irrel_o = irrel_o + np.dot(self.W_io, word_vecs[i])

            rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, self.b_i, sigmoid)
            rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, self.b_g, np.tanh)

            relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
            irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

            if i >= start and i < stop:
                relevant[i] += bias_contrib_i * bias_contrib_g
            else:
                irrelevant[i] += bias_contrib_i * bias_contrib_g

            if i > 0:
                rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, self.b_f, sigmoid)
                relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
                irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[
                    i - 1] + irrel_contrib_f * \
                             relevant[i - 1]

            o = sigmoid(np.dot(self.W_io, word_vecs[i]) + np.dot(self.W_ho, prev_rel_h + prev_irrel_h) + self.b_o)
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(rel_o, irrel_o, self.b_o, sigmoid)
            new_rel_h, new_irrel_h = propagate_tanh_two(relevant[i], irrelevant[i])
            # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
            # irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
            relevant_h[i] = o * new_rel_h
            irrelevant_h[i] = o * new_irrel_h

        # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
        # we actually apply to all the linear layers to get the final influence
        scores = np.dot(self.W_out, relevant_h[T - 1])
        irrel_scores = np.dot(self.W_out, irrelevant_h[T - 1])

        # (T, num_classes)
        return scores
