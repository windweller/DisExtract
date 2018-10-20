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

## Pre-training Encoder (AUX)



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