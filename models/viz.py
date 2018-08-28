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

import matplotlib.pyplot as plt
import math
import matplotlib.colors as colors


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
        s1_batch, s2_batch = Variable(
            s1_batch.cuda()), Variable(s2_batch.cuda())
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

    # this processes and embeds text


class BaseLSTM(object):
    def __init__(self, model, glove_path, bilstm=False):
        self.model = model
        weights = model.encoder.enc_lstm.state_dict()

        self.model.encoder.set_glove_path(glove_path)

        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1)

        self.hidden_dim = model.encoder.enc_lstm_dim

        self.W_ii, self.W_if, self.W_ig, self.W_io = np.split(
            weights['weight_ih_l0'], 4, 0)
        self.W_hi, self.W_hf, self.W_hg, self.W_ho = np.split(
            weights['weight_hh_l0'], 4, 0)
        self.b_i, self.b_f, self.b_g, self.b_o = np.split(
            weights['bias_ih_l0'].numpy() + weights['bias_hh_l0'].numpy(),
            4)

        if bilstm:
            self.rev_W_ii, self.rev_W_if, self.rev_W_ig, self.rev_W_io = np.split(
                weights['weight_ih_l0_reverse'], 4, 0)
            self.rev_W_hi, self.rev_W_hf, self.rev_W_hg, self.rev_W_ho = np.split(
                weights['weight_hh_l0_reverse'], 4, 0)
            self.rev_b_i, self.rev_b_f, self.rev_b_g, self.rev_b_o = np.split(
                weights['bias_ih_l0_reverse'].numpy() + weights['bias_hh_l0_reverse'].numpy(),
                4)

        self.word_emb_dim = 300
        self.glove_path = glove_path

        if not bilstm:
            self.classifiers = [
                (self.model.classifier[0].weight.data.numpy()[:, :20480],
                 self.model.classifier[0].bias.data.numpy())
            ]
        else:
            self.classifiers = [
                (self.model.classifier[0].weight.data.numpy(),
                 self.model.classifier[0].bias.data.numpy())
            ]

        skip = True
        for c in self.model.classifier:
            if skip:
                skip = False
                continue
            self.classifiers.append(
                (c.weight.data.numpy(), c.bias.data.numpy()))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def classify(self, u, v):
        # note that u, v could be positional!! don't mix the two
        final_res = np.concatenate([u, v, u - v, u * v, (u + v) / 2.])
        for c in self.classifiers:
            w, b = c
            final_res = np.dot(w, final_res) + b
        return final_res

    def get_word_dict(self, sentences, tokenize=True, already_split=False):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        if not already_split:
            sentences = [s.split() if not tokenize else word_tokenize(s)
                         for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        return word_dict

    def get_glove(self, word_dict):
        assert hasattr(self, 'glove_path'), \
            'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def build_vocab(self, sentences, tokenize=True, already_split=False):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize, already_split)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

        if already_split:
            self.model.encoder.build_vocab([' '.join(s) for s in sentences], tokenize=False)
        else:
            self.model.encoder.build_vocab(sentences, tokenize=False)

    def prepare_samples(self, sentences, tokenize, verbose, no_sort=False, already_split=False):
        if tokenize:
            from nltk.tokenize import word_tokenize
        if not already_split:
            sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
                         ['<s>'] + word_tokenize(s) + ['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                               Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(
                n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        if no_sort:
            # technically "forward" method is already sorting
            return sentences, lengths

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, return_len=False):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((max_len, len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        # (T, bsize, word_dim)
        if not return_len:
            return embed
        else:
            return torch.from_numpy(embed).float(), lengths


# ======== Global Max-pooling based interpretation ========
class GMPModel(BaseLSTM):
    def get_context_scores(self):
        # note: not word-level scores
        # return: (time_step, label_size)
        pass


# ======== CD-based interpretation =========
# (currently broken because it doesn't extend to global max-pooling)



# propagate a three-part
def propagate_three(a, b, c, activation):
    a_contrib = 0.5 * (activation(a + c) - activation(c) +
                       activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) +
                       activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


# propagate tanh nonlinearity
def propagate_tanh_two(a, b):
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))


def tiles_to_cd(texts):
    starts, stops = [], []
    tiles = texts
    L = tiles.shape[0]
    for c in range(tiles.shape[1]):
        text = tiles[:, c]
        start = 0
        stop = L - 1
        while text[start] == 0:
            start += 1
        while text[stop] == 0:
            stop -= 1
        starts.append(start)
        stops.append(stop)
    return starts, stops


# pytorch needs to return each input as a column
# return batch_size x L tensor


def gen_tiles(text, fill=0,
              method='cd', prev_text=None, sweep_dim=1):
    L = text.shape[0]
    texts = np.zeros((L - sweep_dim + 1, L), dtype=np.int)
    for start in range(L - sweep_dim + 1):
        end = start + sweep_dim
        if method == 'occlusion':
            text_new = np.copy(text).flatten()
            text_new[start:end] = fill
        elif method == 'build_up' or method == 'cd':
            text_new = np.zeros(L)
            text_new[start:end] = text[start:end]
        texts[start] = np.copy(text_new)
    return texts


def correct_propagate_max_two(a, b, d=0):
    return 0.5 * (np.max(a, axis=d) + (np.max(a + b, axis=d) - np.max(b, axis=d))), \
            0.5 * (np.max(b, axis=d) + (np.max(a + b, axis=d) - np.max(a, axis=d)))

class CDLSTM(BaseLSTM):
    # this implementation is wrong...no consideration of max pooling
    def get_word_level_scores(self, sentA, sentB, skip_A=False, skip_B=False):
        """
        :param sentence: ['a', 'b', 'c', ...]
        :return:
        """
        # texts = gen_tiles(text_orig, method='cd', sweep_dim=1).transpose()
        # starts, stops = tiles_to_cd(texts)
        # [0, 1, 2,...], [0, 1, 2,...]

        sent_A, _, _ = self.prepare_samples(
            [sentA], tokenize=False, verbose=True, already_split=True)
        sent_B, _, _ = self.prepare_samples(
            [sentB], tokenize=False, verbose=True, already_split=True)

        tup0, tup1 = self.cd_text(sent_A, start=0, stop=len(sentA) - 1)
        h_A = tup0 + tup1
        tup0, tup1 = self.cd_text(sent_B, start=0, stop=len(sentB) - 1)
        h_B = tup0 + tup1

        # compute A, treat B as fixed
        scores_A = None
        if not skip_A:
            starts, stops = range(len(sentA)), range(len(sentA))
            scores_A = np.array([self.classify(self.cd_text(sent_A, start=starts[i], stop=stops[i])[0], h_B)
                                 for i in range(len(starts))])

        # compute B, treat A as fixed
        scores_B = None
        if not skip_B:
            starts, stops = range(len(sentB)), range(len(sentB))
            scores_B = np.array([self.classify(h_A, self.cd_text(sent_B, start=starts[i], stop=stops[i])[0])
                                 for i in range(len(starts))])

        # (sent_len, num_label)
        return scores_A, scores_B

    def cd_text(self, sentences, start, stop):

        # word_vecs = self.model.embed(batch.text)[:, 0].data
        word_vecs = self.get_batch(sentences).squeeze()

        T = word_vecs.shape[0]
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

            rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(
                rel_i, irrel_i, self.b_i, sigmoid)
            rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(
                rel_g, irrel_g, self.b_g, np.tanh)

            relevant[i] = rel_contrib_i * \
                          (rel_contrib_g + bias_contrib_g) + \
                          bias_contrib_i * rel_contrib_g
            irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (
                                                                                                       rel_contrib_i + bias_contrib_i) * irrel_contrib_g

            if i >= start and i < stop:
                relevant[i] += bias_contrib_i * bias_contrib_g
            else:
                irrelevant[i] += bias_contrib_i * bias_contrib_g

            if i > 0:
                rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(
                    rel_f, irrel_f, self.b_f, sigmoid)
                relevant[i] += (rel_contrib_f +
                                bias_contrib_f) * relevant[i - 1]
                irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[
                    i - 1] + irrel_contrib_f * \
                             relevant[i - 1]

            o = sigmoid(np.dot(
                self.W_io, word_vecs[i]) + np.dot(self.W_ho, prev_rel_h + prev_irrel_h) + self.b_o)
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(
                rel_o, irrel_o, self.b_o, sigmoid)
            new_rel_h, new_irrel_h = propagate_tanh_two(
                relevant[i], irrelevant[i])
            # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
            # irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
            relevant_h[i] = o * new_rel_h
            irrelevant_h[i] = o * new_irrel_h

        return relevant_h[T - 1], irrelevant_h[T - 1]

        # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
        # we actually apply to all the linear layers to get the final influence

        # scores = np.dot(self.W_out, relevant_h[T - 1])
        # irrel_scores = np.dot(self.W_out, irrelevant_h[T - 1])

        # scores = self.classify(relevant_h[T - 1])
        # irrel_scores = self.classify(irrelevant_h[T - 1])

        # (num_classes)
        # return scores


def propagate_max_two(a, b, d=0):
    # need to return a, b with the same shape...
    indices = np.argmax(a + b, axis=d)
    a_mask = np.zeros_like(a)
    a_mask[indices, np.arange(a.shape[1])] = 1
    a = a * a_mask

    b_mask = np.zeros_like(b)
    b_mask[indices, np.arange(b.shape[1])] = 1
    b = b * b_mask

    return a, b

class MaxPoolingCDBiLSTM(BaseLSTM):
    def cell(self, prev_h, prev_c, x_i):
        # x_i = word_vecs[i]
        rel_i = np.dot(self.W_hi, prev_h)
        rel_g = np.dot(self.W_hg, prev_h)
        rel_f = np.dot(self.W_hf, prev_h)
        rel_o = np.dot(self.W_ho, prev_h)

        rel_i = sigmoid(rel_i + np.dot(self.W_ii, x_i) + self.b_i)
        rel_g = np.tanh(rel_g + np.dot(self.W_ig, x_i) + self.b_g)
        rel_f = sigmoid(rel_f + np.dot(self.W_if, x_i) + self.b_f)
        rel_o = sigmoid(rel_o + np.dot(self.W_io, x_i) + self.b_o)

        c_t = rel_f * prev_c + rel_i * rel_g
        h_t = rel_o * np.tanh(c_t)

        return h_t, c_t

    def rev_cell(self, prev_h, prev_c, x_i):
        # x_i = word_vecs[i]
        rel_i = np.dot(self.rev_W_hi, prev_h)
        rel_g = np.dot(self.rev_W_hg, prev_h)
        rel_f = np.dot(self.rev_W_hf, prev_h)
        rel_o = np.dot(self.rev_W_ho, prev_h)

        rel_i = sigmoid(rel_i + np.dot(self.rev_W_ii, x_i) + self.rev_b_i)
        rel_g = np.tanh(rel_g + np.dot(self.rev_W_ig, x_i) + self.rev_b_g)
        rel_f = sigmoid(rel_f + np.dot(self.rev_W_if, x_i) + self.rev_b_f)
        rel_o = sigmoid(rel_o + np.dot(self.rev_W_io, x_i) + self.rev_b_o)

        c_t = rel_f * prev_c + rel_i * rel_g
        h_t = rel_o * np.tanh(c_t)

        return h_t, c_t

    def run_bi_lstm(self, sentences):
        # this is used as validation
        word_vecs = self.get_batch(sentences).squeeze()

        T = word_vecs.shape[0]

        hidden_states = np.zeros((T, self.hidden_dim))
        rev_hidden_states = np.zeros((T, self.hidden_dim))

        cell_states = np.zeros((T, self.hidden_dim))
        rev_cell_states = np.zeros((T, self.hidden_dim))

        for i in range(T):
            if i > 0:
                prev_h = hidden_states[i - 1]  # this is just the prev hidden state
                prev_c = cell_states[i - 1]
            else:
                prev_h = np.zeros(self.hidden_dim)
                prev_c = np.zeros(self.hidden_dim)

            new_h, new_c = self.cell(prev_h, prev_c, word_vecs[i])

            hidden_states[i] = new_h
            cell_states[i] = new_c

        for i in reversed(range(T)):
            # 20, 19, 18, 17, ...
            if i < T - 1:
                prev_h = rev_hidden_states[i + 1]  # this is just the prev hidden state
                prev_c = rev_cell_states[i + 1]
            else:
                prev_h = np.zeros(self.hidden_dim)
                prev_c = np.zeros(self.hidden_dim)

            new_h, new_c = self.rev_cell(prev_h, prev_c, word_vecs[i])

            rev_hidden_states[i] = new_h
            rev_cell_states[i] = new_c

        # stack second dimension
        return np.hstack([hidden_states, rev_hidden_states]), np.hstack([cell_states, rev_cell_states])

    # this is a fast leaf-level implementation for global max pooling
    def get_word_level_scores(self, sentA, sentB, skip_A=False, skip_B=False):
        """
        :param sentence: ['a', 'b', 'c', ...]
        :return:
        """
        # texts = gen_tiles(text_orig, method='cd', sweep_dim=1).transpose()
        # starts, stops = tiles_to_cd(texts)
        # [0, 1, 2,...], [0, 1, 2,...]

        sent_A, _, _ = self.prepare_samples(
            [sentA], tokenize=False, verbose=True, already_split=True)
        sent_B, _, _ = self.prepare_samples(
            [sentB], tokenize=False, verbose=True, already_split=True)

        rel_A, irrel_A = self.cd_encode(sent_A)  # already masked
        rel_B, irrel_B = self.cd_encode(sent_B)

        # now we actually fire up the encoder, and get gradients w.r.t. hidden states
        s1_batch, s1_len = self.get_batch([sent_A], return_len=True)
        s2_batch, s2_len = self.get_batch([sent_B], return_len=True)

        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        hid_states_A = self.model.encoder.get_hidden_states((s1_batch, s1_len))
        hid_states_B = self.model.encoder.get_hidden_states((s2_batch, s2_len))

        u, v = torch.max(hid_states_A, 0)[0], torch.max(hid_states_B, 0)[0]
        features = torch.cat((u, v, u - v, u * v, (u + v) / 2.), 1)
        output = self.model.classifier(features).squeeze()

        label_id = torch.max(output, 0)[1]

        # compute A score
        output[label_id].backward()
        scores_A = hid_states_A.grad.squeeze() * torch.from_numpy(rel_A)

        # compute B, treat A as fixed
        scores_B = hid_states_B.grad.squeeze() * torch.from_numpy(rel_B)

        # (sent_len, num_label)
        return scores_A, scores_B

    def cd_encode(self, sentences):
        rel_h, irrel_h, _ = self.flat_cd_text(sentences)
        rev_rel_h, rev_irrel_h, _ = self.flat_cd_text(sentences, reverse=True)
        rel = np.hstack([rel_h, rev_rel_h])  # T, 2*d
        irrel = np.hstack([irrel_h, rev_irrel_h])  #T, 2*d
        # again, hidden-states = rel + irrel

        # we mask both
        rel_masked, irrel_masked = propagate_max_two(rel, irrel)

        return rel_masked, irrel_masked  # (2*d), actual sentence representation


    def flat_cd_text(self, sentences, reverse=False):
        # collects relevance for word 0 to sent_length
        # not considering interactions between words; merely collecting word contribution

        # word_vecs = self.model.embed(batch.text)[:, 0].data
        word_vecs = self.get_batch(sentences).squeeze()

        T = word_vecs.shape[0]

        # so prev_h is always irrelevant
        # there's no rel_h because we only look at each time step individually

        # relevant cell states, irrelevant cell states
        relevant = np.zeros((T, self.hidden_dim))
        irrelevant = np.zeros((T, self.hidden_dim))

        relevant_h = np.zeros((T, self.hidden_dim))
        irrelevant_h = np.zeros((T, self.hidden_dim))  # keep track of the entire hidden state

        hidden_states = np.zeros((T, self.hidden_dim))
        cell_states = np.zeros((T, self.hidden_dim))

        if not reverse:
            W_ii, W_if, W_ig, W_io = self.W_ii, self.W_if, self.W_ig, self.W_io
            W_hi, W_hf, W_hg, W_ho = self.W_hi, self.W_hf, self.W_hg, self.W_ho
            b_i, b_f, b_g, b_o = self.b_i, self.b_f, self.b_g, self.b_o
        else:
            W_ii, W_if, W_ig, W_io = self.rev_W_ii, self.rev_W_if, self.rev_W_ig, self.rev_W_io
            W_hi, W_hf, W_hg, W_ho = self.rev_W_hi, self.rev_W_hf, self.rev_W_hg, self.rev_W_ho
            b_i, b_f, b_g, b_o = self.rev_b_i, self.rev_b_f, self.rev_b_g, self.rev_b_o

        # strategy: keep using prev_h as irrel_h
        # every time, make sure h = irrel + rel, then prev_h = h

        indices = range(T) if not reverse else reversed(range(T))
        for i in indices:
            first_cond = i > 0 if not reverse else i < T - 1
            if first_cond:
                ret_idx = i - 1 if not reverse else i + 1
                prev_c = cell_states[ret_idx]
                prev_h = hidden_states[ret_idx]
            else:
                prev_c = np.zeros(self.hidden_dim)
                prev_h = np.zeros(self.hidden_dim)

            irrel_i = np.dot(W_hi, prev_h)
            irrel_g = np.dot(W_hg, prev_h)
            irrel_f = np.dot(W_hf, prev_h)
            irrel_o = np.dot(W_ho, prev_h)

            rel_i = np.dot(W_ii, word_vecs[i])
            rel_g = np.dot(W_ig, word_vecs[i])
            rel_f = np.dot(W_if, word_vecs[i])
            rel_o = np.dot(W_io, word_vecs[i])

            # this remains unchanged
            rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(
                rel_i, irrel_i, b_i, sigmoid)
            rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(
                rel_g, irrel_g, b_g, np.tanh)

            relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + \
                          bias_contrib_i * rel_contrib_g
            irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + \
                            (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

            relevant[i] += bias_contrib_i * bias_contrib_g
            # if i >= start and i < stop:
            #     relevant[i] += bias_contrib_i * bias_contrib_g
            # else:
            #     irrelevant[i] += bias_contrib_i * bias_contrib_g

            cond = i > 0 if not reverse else i < T - 1
            if cond:
                rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(
                    rel_f, irrel_f, b_f, sigmoid)
                # previous relevent[i-1] should be 0!!
                # relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
                # irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[
                #     i - 1] + irrel_contrib_f * relevant[i - 1]

                # not sure if this is completely correct
                irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * prev_c

            # recompute o-gate
            o = sigmoid(rel_o + irrel_o + b_o)
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(
                rel_o, irrel_o, b_o, sigmoid)
            # from current cell state
            new_rel_h, new_irrel_h = propagate_tanh_two(relevant[i], irrelevant[i])
            # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
            # irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
            relevant_h[i] = o * new_rel_h
            irrelevant_h[i] = o * new_irrel_h

            hidden_states[i] = relevant_h[i] + irrelevant_h[i]
            cell_states[i] = relevant[i] + irrelevant[i]

        return relevant_h, irrelevant_h, hidden_states


def word_heatmap(text_orig, scores, label_pred=0, data=None, fontsize=9):
    text_orig = np.array(text_orig)
    num_words = text_orig.size

    comps_list = [np.array(range(num_words))]
    num_iters = len(comps_list)

    scores_list = scores

    # populate data
    if data is None:
        data = np.empty(shape=(num_iters, num_words))
        data[:] = np.nan
        data[0, :] = scores_list

    data[np.isnan(data)] = 0  # np.nanmin(data) - 0.001
    if num_iters == 1:
        plt.figure(figsize=(16, 1), dpi=300)
    else:
        plt.figure(figsize=(16, 3), dpi=300)

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

            #     cmap = plt.get_cmap('RdBu') if label_pred == 0 else plt.get_cmap('RdBu_r')

    cmap = plt.get_cmap('RdBu')
    if label_pred == 1:
        data *= -1
    # cmap = matplotlib.cm.Greys
    # cmap.set_bad(color='black')
    #                    cmap='viridis')#'RdBu')
    abs_lim = max(abs(np.nanmax(data)), abs(np.nanmin(data)))

    c = plt.pcolor(data,
                   edgecolors='k',
                   linewidths=0,
                   norm=MidpointNormalize(vmin=abs_lim * -1, midpoint=0., vmax=abs_lim),
                   cmap=cmap)

    def show_values(pc, text_orig, data, fontsize, fmt="%s", **kw):
        val_mean = np.nanmean(data)
        val_min = np.min(data)
        pc.update_scalarmappable()
        # ax = pc.get_axes()
        ax = pc.axes

        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            # pick color for text
            if np.all(color[:3] > 0.5):  # value > val_mean: #value > val_mean: #
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            x_ind = int(math.floor(x))
            y_ind = int(math.floor(y))

            # sometimes don't display text
            if y_ind == 0 or data[y_ind, x_ind] != 0:  # > val_min:
                ax.text(x, y, fmt % text_orig[x_ind],
                        ha="center", va="center",
                        color=color, fontsize=fontsize, **kw)

    show_values(c, text_orig, data, fontsize)
    cb = plt.colorbar(c, extend='both')  # fig.colorbar(pcm, ax=ax[0], extend='both')
    cb.outline.set_visible(False)
    plt.xlim((0, num_words))
    plt.ylim((0, num_iters))
    plt.yticks([])
    plt.plot([0, num_words], [1, 1], color='black')
    plt.xticks([])

    cb.ax.set_title('CD score')
