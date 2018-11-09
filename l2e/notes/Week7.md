# Week 7 Report

## Future Plan

1. Grab "so", and check the quality of "so" -- artificially increase the amount of data.
2. Check the quality inconsistency, and start parsing "so" sentences now. (1h)
3. Finish the NMT architecture to incorporate MLM and LM objective. (3h)
4. Propose L2E as a difficult same-language generation problem with a lot of data (non-overlap, can't be solved by language modeling or normal chatbot training)
5. We can't do a super clean split (no need; use Winograd and COPA to substitute)
   - Because we want to see amazing things like "He was only half - Ivorian". Non clean split gives us chance to see these. Also we don't really measure any objective things on test set (PPL and accuracy and Hits@5).
6. Argument against LM is that it takes a significantly more time to train a full LM on such massive amount of text, but we can pick important sentence pairs to generate. Also Seq2Seq learns sentence boundary and LM currently does not.
   - In this situation, we might only need to train a Transformer LM on the dataset (and show it's not learning sentence boundary)?
   - Also this is a "conditional" language model, so we don't have to differentiate ourselves. 

## Current Plan

1. S1 -> Q, and sample different chatbots (treat data as model/framework/system)
2. NMT with LM, NMT with MLM...right now!!! Don't waste your time!! Last piece, and then move onto writing and mTurk experiment.
3. 

## Paper Structure

Intro

Related Work

- LM problem is to have too massive of data, not learning sentence boundary.
- Problem with common sense as cloze test (or for that matter: story cloze test)

Data

- Offer this as a new dataset / task

Model

- Language Model baseline
  - This shows the benefit of dependency parsing and our schema.
- L2E
- L2EC
  - LTR LM on encoder
- L2P (learn to predict)
- L2PC (learn to predict with context)
  - LTR LM on encoder

Experiment

- Objective: 
  - Common sense ranking (using common sense LM, to see if we can actually generate **even more plausible** explanation than COPA and Winograd; probably we can because we use more `generic` words...that just have high unigram probability)
  - Causal network choice
- Subjective: human experiments

Analysis

- Source of explanations (how can people offer explanations; where do they come from?)
- Model's ability to synthesize old information for the new 
- Gender / plurality without NER or explicit marking (just like other LM...not new..as Chris Manning pointed out)
- ???? More to come!!

## Models for Comparison

1. LM trained on filtered but not parsed corpus. (to show power of dependency parsing; the ability to extract and learn fully formed sentences)
2. L2E, L2EC || L2P
3. LM trained on L2EC / L2PC (ablation, to show the power of decoder, why is decoder important -- why is conditional LM better than straight-forward LM
4. Two Chatbot model ingesting our S1 -> Q questions.

## Human Evaluation Experiments



## Common Sense Ranking

Use Commonsense LM model to rank our generated response and original winograd schema response and see which one is more natural and more plausible by this model!! (Quick test!! 20 minutes)

If it is, then we get a strong experiment there.

We do two ranking:

1. True response vs. our response (can our generated explanation beat human?)
2. Fake response vs. our response (can we at least do better than fake response?)

## Pre-training Encoder (AUX)

A couple of notes. The engineering effort was quite monstrous...

1. We are not getting self-attention map (will require more change)

Because of "sharding", and that we actually do use sharding, the gradient computation in OpenNMT is very difficult to modify...even if we changed NMTLossCompute to reflect auxiliary loss...

Now everything is fixed, and through debugging, we know that auxiliary loss is correctly computed.

```bash
# this version aux_strength = 1
CUDA_VISIBLE_DEVICES=0 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_aux/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformerAuxLTR -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot -aux_strength 1.  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 16 -log_file save/because_ctx_aux/log.txt -share_embeddings -share_decoder_embeddings

# this version aux_strength = 0.5
CUDA_VISIBLE_DEVICES=1 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_aux_str05/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformerAuxLTR -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot -aux_strength 0.5  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 16 -log_file save/because_ctx_aux_str05/log.txt -share_embeddings -share_decoder_embeddings
    
# aux_strength = 0
CUDA_VISIBLE_DEVICES=0 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_aux_str0/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformerAuxLTR -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot -aux_strength 0.  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 16 -log_file save/because_ctx_aux_str0/log.txt -share_embeddings -share_decoder_embeddings
    
# original training (OpenNMT)
CUDA_VISIBLE_DEVICES=1 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 16 -log_file save/because_ctx/log.txt -share_embeddings -share_decoder_embeddings
```

Encoder loss is already pretty small. We can sweep a few parameters for the auxiliary strength.

We then plot out the training process of different process. The last two experiments is to show the change in aux, and whether it's useful or not.

Then we evaluate these models on test set.

```bash
# aux = 1
CUDA_VISIBLE_DEVICES=0 python3.6 translate.py -model save/because_ctx_aux/dissent_step_200000.pt -src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-test.txt \
    -tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-test.txt -share_vocab -output save/because_ctx_aux/dissent_step_200000_pred.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 32 -gpu 0 -beam_size 1 -fast

# OpenNMT (vanilla)
CUDA_VISIBLE_DEVICES=2 python3.6 translate.py -model save/because_ctx/dissent_step_200000.pt -src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-test.txt \
    -tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-test.txt -share_vocab -output save/because_ctx/dissent_step_200000_pred.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 32 -gpu 0 -beam_size 1 -fast
```

## Run Transformer LM as baseline 

A naive baseline on filtered but not dependency parsed sentences.

We write this into the DisExtract model, and we'll add beam search.

```bash
python3 gen_data.py --corpus because_ctx --out_prefix because_2018oct24
python3 gen_data.py --corpus because_nmt  --out_prefix because_2018oct24
python3 gen_data.py --corpus because  --out_prefix because_2018oct24
```

## Evaluating Transformer LM

Both LM and Seq2Seq can be done (to prove dependency parsing) can be evaluated on 2018 news crawl or news discussion dataset...

## Run Transformer LM on L2EC

Possible, and `||` and `<Q>` will be tokens to remind LM to swith tactics. So LM trained on L2EC can still be sampled with `|| <Q> winograd S1` to generate S2.

## Clean Split

If we want an external clean split dataset, we could take:

News Commentary  (from University of Edinburgh: The news commentary corpus is all collected from <https://www.project-syndicate.org/>, and is completely separate from the news crawl corpus.)

(We are probably going to stay with this one) (if we believe a clean split is important)

## Preprocessing `SO`

We also process `so`. We may or may not use it in the end, but it's good to process and take a look. We process both news crawl and news crawl shuffled.

```bash
python newscrawl_en.py --filter_so
python newscrawl_ordered_en.py --filter_so

python newscrawl_en.py --parse --tag SO
python newscrawl_ordered_en.py --parse --tag SO
```

We can use `SO` to exclusively train `generating phenoemenon` (S2 -> S1) and forget about `because` business. It might give us better sentences for COPA's `result generation`.

In shuffled news_crawl, after `filter`, we get 4,302,089 sentences.

In un-shuffled news_crawl, after `filter`, we get 5,136,043 sentences.

## BERT Transformer Model

We try to change the architecture and change a narrow but deep model!

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train.py -data data/because_nmt -save_model save/because_transformer_layer12_oct15/dissent \
    -layers 12 -rnn_size 768 -word_vec_size 768 -transformer_ff 3072 -heads 12  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 2048 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 10000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000  -share_embeddings -valid_batch_size 16 \
    -share_decoder_embeddings -log_file save/because_transformer_layer12_oct15/log.txt -world_size 2 -gpu_ranks 0 1
```

https://arxiv.org/pdf/1810.04805.pdf

It seems like deeper model works well to extract features -- not sure if it works equally well for encoder-decoder architecture!

Because we have 2 models, it has to be smaller than BERT.