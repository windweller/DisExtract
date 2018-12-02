# -*- coding: utf-8 -*-

"""
Data loading code adapted from
https://github.com/facebookresearch/InferSent/blob/master/data.py
"""

import os
import numpy as np
import torch
import logging
from collections import defaultdict
from os.path import join as pjoin
from preprocessing.cfg import EN_FIVE_DISCOURSE_MARKERS, \
    EN_EIGHT_DISCOURSE_MARKERS, EN_DISCOURSE_MARKERS, EN_OLD_FIVE_DISCOURSE_MARKERS, EN_DIS_FIVE, \
    CH_FIVE_DISCOURSE_MARKERS, SP_FIVE_DISCOURSE_MARKERS

import six
import random
import collections
from copy import copy


def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if batch[i][j] in word_vec:
                embed[j, i, :] = word_vec[batch[i][j]]
                # otherwise by default it's 0

    return torch.from_numpy(embed).float(), lengths


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    # tokens: ['a', 'b', 'c'] (just one sentence)

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or token == "<s>" or token == "</s>":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "<unk>"  # [MASK] is not in GloVe, so we use <unk>
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    # we use '<unk>' actually instead of '[MASK]'
    # (['this', 'is', 'fine', '.', 'this', 'is', '[MASK]', 'fine'], [6], ['not'])
    # (masked/altered_sequence, position of mask, original token)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_mlm_batch(batch, word_vec, args, vocab, vocab_dict, flatten_targets=False):
    # no candidate indices
    # we also get batch s1, s2 seperately
    # vocab: list[words]
    # vocab_dict: {word: idx}

    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))  # embed for noised sentence

    # hidden states: [batch_size, time_step, hid_dim]
    # gather all indices, turn into a flat tensor; target should also be flat tensor too

    masked_poss = []
    lm_targets = []

    for i in range(len(batch)):
        # 1. noise the text
        noised_sent, masked_pos, lm_labels = create_masked_lm_predictions(batch[i], args.masked_lm_prob,
                                                                          args.max_predictions_per_seq, vocab,
                                                                          random.Random(args.seed))

        masked_poss.append(masked_pos)
        if flatten_targets:
            lm_targets.extend([vocab_dict[t] for t in lm_labels])  # (batch_size * num_preds)
        else:
            lm_targets.append([vocab_dict[t] for t in lm_labels])  # (batch_size, num_preds)

        for j in range(len(noised_sent)):
            # embed into vectors
            if noised_sent[j] in word_vec:
                embed[j, i, :] = word_vec[noised_sent[j]]
                # otherwise by default it's 0

    # embed is time_major (need to be very careful)
    return torch.from_numpy(embed).float(), lengths, masked_poss, lm_targets


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
        len(word_vec), len(word_dict)))
    return word_vec


def list_to_map(dis_label):
    dis_map = {}
    for i, l in enumerate(dis_label):
        dis_map[l] = i
    return dis_map


def get_dis(data_dir, prefix, discourse_tag="books_5"):
    s1 = {}
    s2 = {}
    target = {}

    if discourse_tag == "books_5" or discourse_tag == 'books_5_perfectly_balanced':
        dis_map = list_to_map(EN_FIVE_DISCOURSE_MARKERS)
        markers = EN_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "books_8":
        dis_map = list_to_map(EN_EIGHT_DISCOURSE_MARKERS)
        markers = EN_EIGHT_DISCOURSE_MARKERS
    elif discourse_tag == "books_all" or discourse_tag == "books_perfectly_balanced" or discourse_tag == "books_mostly_balanced":
        dis_map = list_to_map(EN_DISCOURSE_MARKERS)
        markers = EN_DISCOURSE_MARKERS
    elif discourse_tag == "books_dis_five":
        dis_map = list_to_map(EN_DIS_FIVE)
        markers = EN_DIS_FIVE
    elif discourse_tag == "books_old_5":
        dis_map = list_to_map(EN_OLD_FIVE_DISCOURSE_MARKERS)
        markers = EN_OLD_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "gw_cn_5":
        dis_map = list_to_map(CH_FIVE_DISCOURSE_MARKERS)
        markers = CH_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "gw_es_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
        markers = SP_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "gw_es_1M_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
        markers = SP_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == 'dat':
        dis_map = list_to_map(['entail', 'contradict'])
        markers = ['entail', 'contradict']
    else:
        raise Exception("Corpus/Discourse Tag Set {} not found".format(discourse_tag))
    print(markers)

    logging.info(dis_map)
    # dis_map: {'and': 0, ...}

    for data_type in ['train', 'valid', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}

        text_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")

        s1[data_type]['sent'] = []
        s2[data_type]['sent'] = []
        target[data_type]['data'] = []

        with open(text_path, 'r') as f:
            for line in f:
                columns = line.split('\t')
                # we use this to avoid/skip lines that are empty
                if len(columns) != 3:
                    continue
                marker = columns[2].rstrip('\n')
                if marker in markers:
                    s1[data_type]['sent'].append(columns[0])
                    s2[data_type]['sent'].append(columns[1])
                    target[data_type]['data'].append(dis_map[columns[2].rstrip('\n')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
               len(target[data_type]['data'])

        target[data_type]['data'] = np.array(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['valid']['sent'], 's2': s2['valid']['sent'],
           'label': target['valid']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test


def get_filtered_dis(data_dir, prefix, chosen_marker, discourse_tag="books_5"):
    # chosen_marker: the only marker we want to collect
    s1 = {}
    s2 = {}
    target = {}

    if discourse_tag == "books_5":
        dis_map = list_to_map(EN_FIVE_DISCOURSE_MARKERS)
        markers = EN_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "books_8":
        dis_map = list_to_map(EN_EIGHT_DISCOURSE_MARKERS)
        markers = EN_EIGHT_DISCOURSE_MARKERS
    elif discourse_tag == "books_all" or discourse_tag == "books_perfectly_balanced" or discourse_tag == "books_mostly_balanced":
        dis_map = list_to_map(EN_DISCOURSE_MARKERS)
        markers = EN_DISCOURSE_MARKERS
    elif discourse_tag == "books_dis_five":
        dis_map = list_to_map(EN_DIS_FIVE)
        markers = EN_DIS_FIVE
    elif discourse_tag == "books_old_5":
        dis_map = list_to_map(EN_OLD_FIVE_DISCOURSE_MARKERS)
        markers = EN_OLD_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "gw_cn_5":
        dis_map = list_to_map(CH_FIVE_DISCOURSE_MARKERS)
        markers = CH_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "gw_es_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
        markers = SP_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == "gw_es_1M_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
        markers = SP_FIVE_DISCOURSE_MARKERS
    elif discourse_tag == 'dat':
        dis_map = list_to_map(['entail', 'contradict'])
        markers = ['entail', 'contradict']
    else:
        raise Exception("Corpus/Discourse Tag Set {} not found".format(discourse_tag))
    print(markers)

    logging.info(dis_map)
    # dis_map: {'and': 0, ...}

    assert chosen_marker in markers, "marker {} not in list of markers".format(chosen_marker)

    for data_type in ['train', 'valid', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}

        text_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")

        s1[data_type]['sent'] = []
        s2[data_type]['sent'] = []
        target[data_type]['data'] = []

        with open(text_path, 'r') as f:
            for line in f:
                columns = line.split('\t')
                # we use this to avoid/skip lines that are empty
                if len(columns) != 3:
                    continue
                marker = columns[2].rstrip('\n')
                if marker in markers and marker == chosen_marker:
                    s1[data_type]['sent'].append(columns[0])
                    s2[data_type]['sent'].append(columns[1])
                    target[data_type]['data'].append(dis_map[columns[2].rstrip('\n')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
               len(target[data_type]['data'])

        target[data_type]['data'] = np.array(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['valid']['sent'], 's2': s2['valid']['sent'],
           'label': target['valid']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test


def get_merged_data(data_dir, prefix, discourse_tag="books_5"):
    # for evaluation
    s1 = defaultdict(list)
    s2 = defaultdict(list)
    target = defaultdict(list)

    if discourse_tag == "books_5":
        dis_map = list_to_map(EN_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "books_8":
        dis_map = list_to_map(EN_EIGHT_DISCOURSE_MARKERS)
    elif discourse_tag == "books_all" or discourse_tag == "books_perfectly_balanced" or discourse_tag == "books_mostly_balanced":
        dis_map = list_to_map(EN_DISCOURSE_MARKERS)
    elif discourse_tag == "books_dis_five":
        dis_map = list_to_map(EN_DIS_FIVE)
    elif discourse_tag == "books_old_5":
        dis_map = list_to_map(EN_OLD_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "gw_cn_5":
        dis_map = list_to_map(CH_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "gw_es_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "gw_es_1M_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == 'dat':
        dis_map = list_to_map(['entail', 'contradict'])
    else:
        raise Exception("Corpus/Discourse Tag Set {} not found".format(discourse_tag))

    logging.info(dis_map)
    # dis_map: {'and': 0, ...}

    for data_type in ['train', 'valid', 'test']:
        # s1[data_type], s2[data_type], target[data_type] = {}, {}, {}

        text_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")

        # s1['sent'] = []
        # s2[data_type]['sent'] = []
        # target[data_type]['data'] = []

        with open(text_path, 'r') as f:
            for line in f:
                columns = line.split('\t')
                # we use this to avoid/skip lines that are empty
                if len(columns) != 3:
                    continue
                s1['sent'].append(columns[0])
                s2['sent'].append(columns[1])
                target['data'].append(dis_map[columns[2].rstrip('\n')])

    assert len(s1['sent']) == len(s2['sent']) == len(target['data'])

    target['data'] = np.array(target['data'])

    print('** DATA : Found {0} pairs of sentences.'.format(
        len(s1['sent'])))

    # train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
    #          'label': target['train']['data']}
    # dev = {'s1': s1['valid']['sent'], 's2': s2['valid']['sent'],
    #        'label': target['valid']['data']}

    test = {'s1': s1['sent'], 's2': s2['sent'], 'label': target['data']}
    return test


def get_data(data_dir):
    # this method will be called at each training epoch to produce

    s1 = defaultdict(list)
    s2 = defaultdict(list)

    logging.info("Loading MLM data")

    for data_type in ['train', 'valid', 'test']:
        s1[data_type], s2[data_type] = {}, {}

        text_path = pjoin(data_dir, data_type + ".txt")  # train.txt

        s1[data_type]['sent'] = []
        s2[data_type]['sent'] = []

        with open(text_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                # we use this to avoid/skip lines that are empty
                if len(columns) != 2:
                    continue
                s1[data_type]['sent'].append(columns[0])
                s2[data_type]['sent'].append(columns[1])

        assert len(s1['sent']) == len(s2['sent'])
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent']}
    dev = {'s1': s1['valid']['sent'], 's2': s2['valid']['sent']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent']}

    return train, dev, test


def get_nsp_data(data_dir, args):
    # this method will be called at each training epoch to produce

    rng = random.Random(args.seed)

    s1 = defaultdict(list)
    s2 = defaultdict(list)
    target = {}

    logging.info("Loading MLM data")

    for data_type in ['train', 'valid', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}

        text_path = pjoin(data_dir, data_type + ".txt")  # train.txt

        s1[data_type]['sent'] = []
        s2[data_type]['sent'] = []
        target[data_type]['data'] = []

        label_map = {'random': 0, 'not_random': 1}

        with open(text_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                # we use this to avoid/skip lines that are empty
                if len(columns) != 2:
                    continue
                s1[data_type]['sent'].append(columns[0])
                s2[data_type]['sent'].append(columns[1])

        assert len(s1['sent']) == len(s2['sent'])
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]['sent']), data_type))

        # now we noise them
        for idx in range(len(s1[data_type]['sent'])):
            label = 'random'
            if rng.random() < 0.5:
                label = 'not_random'

                for _ in range(10):
                    random_document_index = rng.randint(0, len(s2[data_type]['sent']) - 1)
                    if random_document_index != idx:
                        break

                s2[data_type]['sent'][idx] = copy(s2[data_type]['sent'][random_document_index])

            target[data_type]['data'].append(label_map[label])

        # turn target into numpy array
        target[data_type]['data'] = np.array(target[data_type]['data'])

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['valid']['sent'], 's2': s2['valid']['sent'],
           'label': target['valid']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}

    return train, dev, test
