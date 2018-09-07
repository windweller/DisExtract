# Week 2 Report

## Plans

1. Curate a sizable `because` corpus
   - Parse Gigaword English
   - Search for other existing large corpus (found NewsCrawl)
   - See if I can easily reformulate SQuAD why-questions (this could be an evaluation target)
2. Train a Seq2Seq style model
   - Train with Transformer LM decoder learned by OpenAI (BookCorpus) (train your own encoder).
   - Train Transformer Encoder and Decoder from scratch
   - Train Transformer Encoder with LSTM decoder
3. Train a ranker model (Easier task)
   - Use a pre-trained language model decoder (Transformer or LM) to generate (sample) sentences after reading through Sent A with `because` marker. 
   - Then train an encoder to jointly encode Sent A with multiple generated fake Sent B (make sure they are truly incorrect), and distinguish them with correct Sent B. (LM will give thematically similar sentence, closer to Sent A; better than randomly picking sentences from a corpora of Sent B).
4. Cool Adversarial training / Dual training?
   - Decoder keeps generating more realistic looking sentences
   - A ranker learns to differentiate them
   - The same ranker provides reward for a new decoder to learn better generation. 
5. Read up on the pun paper (https://web.stanford.edu/~ngoodman/papers/KaoLevyGoodman.pdf)
6. Could use the classifier to identify implicit "cause" relations. Verify on PDTB. Train a classifier that has a high precision on recognizing "because" sentences.

## Report

### Corpus Curation

How much data is enough for Seq2Seq?

WMT'14 English-French corpus, people use 36M sentences (for relatively successful seq2seq model to work). Test dataset is *newstest2014* dataset. (BLEU: 41.8)

WMT'14 English-German corpus, it has about 4.5M sentences. Test dataset is *newstest2016* dataset. (BLEU: 28.4)

Searching through machine translation dataset is a good way to discover large amount of sentences.

Our selections

- Gigaword English Fifth Edition
- News Crawl
- News discussion 

**Gigaword English Fifth Edition**

Parsed `because` sentences: 811,117 (0.8M)

This is already more than the BookCorpus, and these sentences look incredibly well-formatted:

| Sent A                                                       | Sent B                                                | Marker  |
| ------------------------------------------------------------ | ----------------------------------------------------- | ------- |
| However , the name of this campaign might be its biggest drawback . | It assumes that there is , in fact , a Generation X . | because |
| Such complex cases require patience .                        | Jurors are dealing with unfamiliar issues .           | because |
| Gun control is so small for Lott.                            | Gun money is so large for his Republican Party .      | because |

**News Crawl 2007 - 2017**

http://www.statmt.org/wmt18/translation-task.html

We download data from News Crawl 2007 - 2017.

The data file is about same size as Gigaword English

The corpus has 191,599,627 (**191M**) sentences. Gigaword had **116M** sentences.

After selecting `because`, we obtained 3,121,391 (**3.1M**) `because` sentences.

We have a nice yield (many `because` sentences are chosen, but also not super high -- not all `because` sentences should be chosen -- so that we know the extraction is important and valuable)

`because` is only about 1.5% of the entire language use.

**News Discussions 2014-2017**

This dataset is huge as well. The data curator crawled the comments after a news article. However, upon quick examination, this corpus is relatively noisy, full of spelling error and un-factual claims. This raises my doubt on whether this is a good dataset.

Exhibit 1: "Flat-earth theory"

```
Hi Deb can you proove Gravity or that the Sun is about 91 million miles from the Earth ?, what about all the cut outs of the Earth showing all the layers to the core when the drilling stopped in 1994, the hole was over seven miles deep (12,262 meters) see my point?
```

## Dataset Preprocessing

We preprocessed **2,595,736 (2.6M)** sentences from two datasets (Gigaword English 5th, and News Crawl 2007-2017).

Filter out **382,371 (382k)** sentence pairs because they don't satisfy our filtering criteria (s1, s2 < 50 words, s1, s2 > 5 words, ratio cannot be too much).

We finally get **1,992,028/110,669/110,668** (2213365, which is **2.2M**).

## Train a Seq2Seq style model

We follow the Seq2Seq preprocessing standards. We first share the vocabulary between source and target.

The standard limits vocabulary to 50k, and sentence length to 50 words (which coincides with our own filtering standard!)

```bash
CUDA_VISIBLE_DEVICES=3 python3.6 train.py -data data/because_nmt -save_model save/because_transformer_sep5/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -gpuid 0 -share_embeddings \
    -share_decoder_embeddings -log_file save/because_transformer_sep5/log.txt
```

This leads to

```
[2018-09-05 18:30:12,146 INFO] encoder: 44517376
[2018-09-05 18:30:12,146 INFO] decoder: 25275220
[2018-09-05 18:30:12,146 INFO] * number of parameters: 69792596
```

Configuration consideration:

1. Currently we shared embeddings. Anand Avati said not sharing embeddings could work better.
2. Maybe slash the train_steps in half...to 100000. Adjust it according to the results of the first run?

We try another configuration

```bash
CUDA_VISIBLE_DEVICES=1 python3.6 train.py -data data/because_nmt -save_model save/because_transformer_sep6_no_shared_emb/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -gpuid 0 \
    -share_decoder_embeddings -log_file save/because_transformer_sep5/log.txt
```

This leads to a larger decoder, and basically filled up our memory entirely.

```
[2018-09-06 09:08:17,684 INFO] encoder: 44517376
[2018-09-06 09:08:17,684 INFO] decoder: 50877268
[2018-09-06 09:08:17,684 INFO] * number of parameters: 95394644
```

Tasks:

1. Plot the cross-entropy loss curve, and "accuracy" curve.
2. Evaluate BLEU, Rogue score
3. Count the "epoch" by checking how many times model loads data in again.

## Language Modeling Candidate Generation

We'll try large language models.

LM-1-Billion is from WMT News-Crawl 2011 (nice overlap with our dataset).

Existing language models

* Google Tensorflow LM: [1Billion LM](https://github.com/tensorflow/models/tree/master/research/lm_1b): trained on LM-1-Billion
* Google Tensorflow [SkipThought](https://github.com/tensorflow/models/tree/master/research/skip_thoughts): trained on Gutenberg Books
* Google Tensorflow [Common Sense LM](https://github.com/tensorflow/models/tree/master/research/lm_commonsense): trained on LM-1-Billion, CommonCrawl, SQuAD, Gutenberg Books. (a ton of data) (superset of above)
* OpenAI PyTorch [Transformer Decoder LM](https://github.com/openai/finetune-transformer-lm): trained on Gutenberg Books.
* NVIDIA PyTorch [ByteLSTM LM](https://github.com/NVIDIA/sentiment-discovery#pretrained-models): trained on Amazon reviews. (the sentiment neuron paper)

Observation:

1. Currently we don't have a language model trained on Gigaword English, WikiText/Wikipedia. 1BLM is on news, so that's good.

Ideas:

1. What if, instead of the "ranking/selection" as an intermediate step, we treat it as the final task? We can construct two datasets: one dataset that just randomly samples 2 other sentences in the dataset (based on edit distance); one dataset that has adversarially generated sentences from language models.
2. As for the quality of LM, we can measure the edit distance or Jaccard distance it has with the original s2? (Just generate all of them). 
   - Then we can also measure the distance between the original s2 and 

Puzzles:

1. How to pick the best language model?

Tasks:

1. Build a `sample(vocab_dist, prev_tokens, beam_size=1)` function that takes in a probability distribution and previously generated tokens, sample beam :) Can use KN-5 LM as the default scoring function.
2. Provide a list of candidates to show Noah today.