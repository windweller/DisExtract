"""
Data loading code adapted from
https://github.com/facebookresearch/InferSent/blob/master/data.py
"""

import os
import numpy as np
import torch
from os.path import join as pjoin
from preprocessing.cfg import EN_FIVE_DISCOURSE_MARKERS


def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


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


def get_dis(data_dir, prefix, discourse_tag="books_5"):
    s1 = {}
    s2 = {}
    target = {}

    if discourse_tag == "books_5":
        dis_label = EN_FIVE_DISCOURSE_MARKERS
        dis_map = {}
        for i, l in enumerate(dis_label):
            dis_map[l] = i

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
                s1[data_type]['sent'].append(columns[0])
                s2[data_type]['sent'].append(columns[1])
                target[data_type]['data'].append(columns[2].rstrip('\n'))

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
               len(target[data_type]['data'])

        target[data_type]['data'] = np.array(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test
