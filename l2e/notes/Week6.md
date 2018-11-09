# Week 6 Report

## Outline

1. Training another model on the contextual generation
2. Make slides for Wednesday's presentation
3. Design experiments (human evaluation)

## TODO

1. Interpretation schema; identify nouns; visualize!!
2. Collect PMI for nouns, and replicate causal net.

## Plan

1. COPA is Winograd-like. If we are able to beat COPA and Winograd, instead of choosing, we directly generating, that's a huge deal too!!
2. Find good human evaluation metrics / methods.
3. Translate S1 to Q.
4. Comparative models: Dialogue agents trained on other datasets/tasks.
5. Run Twitter data model on S1 and S2.
6. Train two models -- S2 -> S1 (training now, 10h training).

## Important Parts

1. Show that generic language model cannot do this well (Commonsense LM).
2. Show that dialogue agents trained on other datasets cannot do this (make the dataset pitch).
3. Maybe look into attention alignment between src and tgt.

## Presentation Outline

1. Motivation: need to be able to offer explanations
2. Talk about DisSent and dependency pattern matching to extract S1 and S2
3. Talk about how `because` is difficult to learn even in a balanced setting.
4. Generate a large corpus of `because`, and learn a generative model p(s1|s2) -- generating phenomenon; p(s2|s1) -- generating explanation. (phenomenon because explanation)
5. Data preprocessing; collection; basic characteristics (2h)
6. Model details (both models)
7. Training, training curve; val/test result.
8. Analysis -- which one is harder? 
9. Evaluation on Winograd Schema (Plausibility test)
10. Evaluation on COPA (Plausibility test)
11. Check out some generated phrases
12. Generating with context 
    - What does context provide us?
13. Propose human evaluation schemes
14. Brainstorm time: 
    - What models to compare to?
    - Possible ways to turn S1 into Q (introducing SRL patterns)

## L2EC 

Back to the preprocessing train. We augment `newscrawl_ordered_en.py` with the capability to include context.

```bash
python newscrawl_ordered_en.py --parse
python gigaword_en.py --filter_because
python gen_because_ctx.py --out_prefix gigaword_newscrawl_ordered_ctx_2018oct11

python3 preprocess.py -train_src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-train.txt -train_tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-train.txt -valid_src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-valid.txt -valid_tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-valid.txt \
 -save_data data/because_ctx/because_ctx -share_vocab -src_seq_length 400
 
# for PGNet
 python3 preprocess.py -train_src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-train.txt -train_tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-train.txt -valid_src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-valid.txt -valid_tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-valid.txt \
 -save_data data/because_ctx/because_ctx_dym -share_vocab -src_seq_length 400 -dynamic_dict

# normal training
CUDA_VISIBLE_DEVICES=0,1 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1 -valid_batch_size 16 -log_file save/because_ctx/log.txt -share_embeddings

# Greedy decoding
CUDA_VISIBLE_DEVICES=2 python3.6 translate.py -model save/because_ctx/dissent_step_100000.pt -src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-test.txt -tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-test.txt  -share_vocab -output data/because_ctx/dissent_100000_pred.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 16 -gpu 0 -beam_size 1 -fast

# beam size = 5
CUDA_VISIBLE_DEVICES=3 python3.6 translate.py -model save/because_ctx/dissent_step_100000.pt -src data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-src-test.txt -tgt data/because_ctx/gigaword_newscrawl_ordered_ctx_2018oct11-tgt-test.txt  -share_vocab -output data/because_ctx/dissent_100000_pred_beam5.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 16 -gpu 0 -beam_size 5

# PGNet training

```

A small problem of this is that 

## Generating Causes: S2 -> S1

```bash
# preprocessing
python3 preprocess.py -train_src data/tgt-train.txt -train_tgt data/src-train.txt -valid_src data/tgt-val.txt -valid_tgt data/src-val.txt \
 -save_data data/because_s2_s1 -share_vocab

# training Transformer
CUDA_VISIBLE_DEVICES=0,1 python3 train.py -data data/because_s2_s1 -save_model save/because_transformer_s2_s1_oct9/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000  -share_embeddings \
    -share_decoder_embeddings -log_file save/because_transformer_s2_s1_oct9/log.txt -world_size 2 -gpu_ranks 0 1

# Translating test set
CUDA_VISIBLE_DEVICES=2 python3.6 translate.py -model save/because_transformer_s2_s1_oct9/dissent_step_60000.pt -src data/tgt-test.txt -tgt data/src-test.txt -share_vocab -output data/dissent_s2_s1_step_60000_pred.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 1 -fast

# training LSTM S1->S2
CUDA_VISIBLE_DEVICES=2,3 python3 train.py -data data/because_nmt -save_model save/because_transformer_s1_s2_big_lstm_oct9/dissent \
    -layers 2 -rnn_size 1024 -word_vec_size 1024 \
    -encoder_type brnn -decoder_type rnn \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.3 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim sgd  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000  -share_embeddings -share_decoder_embeddings \
    -log_file save/because_transformer_s1_s2_big_lstm_oct9/log.txt -world_size 2 -gpu_ranks 0 1 -master_port 11000

# training LSTM S2->S1
CUDA_VISIBLE_DEVICES=4,5 python3 train.py -data data/because_s2_s1 -save_model save/because_transformer_s2_s1_big_lstm_oct9/dissent \
    -layers 2 -rnn_size 1024 -word_vec_size 1024 \
    -encoder_type brnn -decoder_type rnn \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.3 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim sgd  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000  -share_embeddings -share_decoder_embeddings \
    -log_file save/because_transformer_s2_s1_big_lstm_oct9/log.txt -world_size 2 -gpu_ranks 0 1 -master_port 12000
```

For the LSTM model, we use the one CoQA suggested:

The default settings of OpenNMT: 2-layers of LSTMs with 500 hidden units for both the encoder and the decoder. The models are optimized using SGD, with an initial learning rate of 1.0 and a decay rate of 0.5. A dropout rate of 0.3 is applied to all layers.

Combined with results from [GenSent](https://openreview.net/pdf?id=B18WgG-CZ) (large setting), and results from [LM-commonsense](https://arxiv.org/abs/1806.02847) (We use two layers of LSTM [32] with 8,192 hidden units and a projection layer, we use a big embedding look up matrix with vocabulary size 800K and embedding size 1,024).

[2018-10-09 15:58:41,318 INFO] encoder: 63803392
[2018-10-09 15:58:41,318 INFO] decoder: 24183636
[2018-10-09 15:58:41,318 INFO] * number of parameters: 87987028

(This parameter size is similar to Transformer now; but it's not very satisfying...since the model is NOT bigger...)

## Evaluating on COPA

Generate sentences

```bash
# S1 -> S2 model
# greedy decode
CUDA_VISIBLE_DEVICES=0 python3.6 translate.py -model save/because_transformer_sep5/dissent_step_80000.pt -src data/copa/s1-only.txt \
     -share_vocab -output data/copa/dissent_step_170000_pred.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 1 -fast

# beam size = 5
CUDA_VISIBLE_DEVICES=0 python3.6 translate.py -model save/because_transformer_sep5/dissent_step_80000.pt -src data/copa/s1-only.txt \
     -share_vocab -output data/copa/dissent_step_170000_pred_beam5.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 5 -fast

# S2 -> S1 model
# greedy decode
CUDA_VISIBLE_DEVICES=0 python3.6 translate.py -model save/because_transformer_s2_s1_oct9/dissent_step_60000.pt -src data/copa/s2-only.txt -share_vocab -output data/copa/dissent_s2_s1_step_60000_pred.txt -replace_unk 
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 1 -fast

# beam size = 5
CUDA_VISIBLE_DEVICES=0 python3.6 translate.py -model save/because_transformer_s2_s1_oct9/dissent_step_60000.pt -src data/copa/s2-only.txt -share_vocab -output data/copa/dissent_s2_s1_step_60000_pred_beam5.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 5
```

We tokenized COPA now :)

Both CAUSE and RESULT can be seen in this [link](http://nbviewer.jupyter.org/github/windweller/DisExtract/blob/edge/models/notes/assets/week6/COPA%20Generation.ipynb).

## Possible Analysis

We can use the trained model to probe causal verbs??? Any possibility?

## S1 -> Q Movement Rule

Dependency parsing

1. Take the root of sentence.
2. Look whether there's an `aux` or `cop` or `auxpass` dependent of the root, take the first of these, and move it before the subject `nsubjpass` or `nsubj`.
3. Otherwise put "do" or "does" or "did" in front of the subject (by the NN tag),  check root verb's lemma, and past tense -> lemma.
4. Remove any `RB` (adverbs) in front of the subject (`nsubj` or `nsubjpass`)

## Related Work

**Commonsense causal reasoning between short texts**

https://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12818

Causality measured as PMI. We are not estimating causation. We also don't rely on templates to extract causal pairs.

Causal strength estimation through co-occurence of cause and effect. Automatic construction of causal network. (Maybe can extend to bio-medical world?)