import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format
import data_utils

from eval_lm1b import _LoadModel

pbtxt = "/mnt/fs5/anie/lm1b/tf-lm1b/graph-2016-09-10.pbtxt"
ckpt = "/mnt/fs5/anie/lm1b/tf-lm1b/ckpt-*"
vocab_file = "/mnt/fs5/anie/lm1b/tf-lm1b/vocab-2016-09-10.txt"

MAX_WORD_LEN = 50
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
vocab = data_utils.CharsVocabulary(vocab_file, MAX_WORD_LEN)

import argparse

parser = argparse.ArgumentParser(description='Sample LM1B for datasets')

parser.add_argument("--dataset", type=str, default='winograd',
                    help="winograd|copa|news_commentary|news_commentary_ctx")
parser.add_argument("--output_file", type=str, help="where to put it")
parser.add_argument("--max_seq", type=int, default=25, help="for winograd, should be 25")
parser.add_argument("--batch_size", type=int, default=1, help="sample an entire batch")
parser.add_argument("--print_sent", action='store_true', help="print each generated sentence")
parser.add_argument("--eval_ppl", action='store_true', help="print each generated sentence")

args, _ = parser.parse_known_args()

data_paths = {
    'news_commentary_ctx': "/mnt/fs5/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-ctx-src-test.txt",
    "news_commentary": "/mnt/fs5/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt",
    "winograd": "/mnt/fs5/anie/DisExtract/data/winograd/src-wsc62.txt",
    "copa": "/mnt/fs5/anie/DisExtract/data/copa/copa_s1.txt"  # TODO: get this!!! (from Jupyter Notebook)
}

import logging

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sentences():
    sents = []
    with open(data_paths[args.dataset], 'r') as f:
        for line in f:
            sents.append(line.strip())
    return sents


# np.random.seed(255943)

def _SampleSoftmax(softmax):
    return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)


# TODO: add batch sampling here!!
def batch_sample(prefix_sents, vocab, max_sample_words=50, sess=None, t=None):
    """
    This code forbids batch sampling

    :param prefix_sents: [str] (list of sentences)
    :param vocab:
    :param max_sample_words:
    :return:
    """
    current_batch_size = len(prefix_sents)
    BATCH_SIZE = current_batch_size

    max_current_seq_len = max(map(lambda x: len(x.split()), prefix_sents))
    max_sample_len = max_current_seq_len + max_sample_words

    targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)  # NUM_TIMESTEPS = 1
    weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)  # Cross-ent weights, not important

    prefix_sents = ['<S> ' + prefix_sent for prefix_sent in prefix_sents]

    prefix = []
    prefix_char_ids = []  # (batch_size, num_of_words, max_word_len=50)
    for s in prefix_sents:
        prefix.append([vocab.word_to_id(w) for w in s.split()])
        prefix_char_ids.append([vocab.word_to_char_ids(w) for w in s.split()])

    # we consumes one word / one set of chars at a time
    # this should be flexible / easy to extend
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)

    # make copies
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]

    # deocded sents
    sents = [''] * current_batch_size

    # stopped indices
    stop = [0] * current_batch_size

    # we do sample all the way, but keep a stop counter to cut
    for i in range(max_sample_len):
        # 1. load input matrix with the previous time step
        for j in range(current_batch_size):
            inputs[j, 0] = samples[j][0]
            char_ids_inputs[j, 0, :] = char_ids_samples[j][0]
            samples[j] = samples[j][:1]  # move tape forward

        # 2. run softmax
        softmax = sess.run([t['softmax_out']],
                           feed_dict={t['char_inputs_in']: char_ids_inputs,
                                      t['inputs_in']: inputs,
                                      t['targets_in']: targets,
                                      t['target_weights_in']: weights})

        sample = np.argmax(softmax, axis=1)
        # sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))

        return sample,

        # n. we check if some lists are already empty, if so, we fill with <PAD>

    # n+1. we reprocess the list and cut off anything after stopped indices

    pass


def sample(prefix_words, vocab, max_sample_words=50):
    targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    # weight for computing cross-entropy
    weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)
    if prefix_words.find('<S>') != 0:
        prefix_words = '<S> ' + prefix_words
    prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
    prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]

    # this is the code for only one sample
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]
    sent = ''

    original_length = len(prefix_words.split())
    max_sample_words = original_length + max_sample_words  # original length is 300, we sample 350

    while True:
        inputs[0, 0] = samples[0]
        char_ids_inputs[0, 0, :] = char_ids_samples[0]
        samples = samples[1:]
        char_ids_samples = char_ids_samples[1:]

        softmax = sess.run([t['softmax_out']],
                           feed_dict={t['char_inputs_in']: char_ids_inputs,
                                      t['inputs_in']: inputs,
                                      t['targets_in']: targets,
                                      t['target_weights_in']: weights})

        # sample = _SampleSoftmax(softmax[0])
        sample = np.argmax(softmax[0])
        if sample == 2:
            sample = _SampleSoftmax(softmax[0])  # when <UNK> appears, we randomly sample something
        sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))

        if not samples:
            samples = [sample]
            char_ids_samples = [sample_char_ids]
        sent += vocab.id_to_word(samples[0]) + ' '

        if (vocab.id_to_word(samples[0]) == '</S>' or
                    len(sent.split()) > max_sample_words):
            break

    return sent


if __name__ == '__main__':
    sess, t = _LoadModel(pbtxt, ckpt)
    if args.batch_size == 1:
        gen_sents = []
        with open(args.output_file, 'w') as f_out:
            for i, sent in enumerate(load_sentences()):
                add_because = " because" if args.dataset == "copa" else "because"
                gen_sent = sample(sent[:-1] + add_because, vocab, args.max_seq)
                gen_sents.append(gen_sent)

                if args.print_sent:
                    print(gen_sent)

                f_out.write(gen_sent + '\n')

                if i % 10 == 0:
                    logger.info(i)
    else:
        sents = load_sentences()
        for i in range(0, len(sents), args.batch_size):
            batched_sents = sents[i:i + args.batch_size]
            batch_sample(batched_sents, vocab, args.max_seq)
