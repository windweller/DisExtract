# Week 8 Report

We still have FairSeq ([link](https://fairseq.readthedocs.io/en/latest/getting_started.html#)) vs. Tensor2Tensor ([link](https://github.com/tensorflow/tensor2tensor/blob/master/docs/overview.md))

But both have Language Modeling, and both support Transformer

Both support beam decode

Unique advantages:

FairSeq supports large batch size, but not flexible loss function (we don't care about this...).

Tensor2Tensor might have better API wrapper, also BPE.

Disadvantages:

FairSeq needs seperate BPE runs.

Tensor2Tensor: TensorFlow might be hard to debug and understand. 

We re-organize and think about what is the ultimate goal/set of experiments that we want!

Let's stay with FairSeq right now. `SequenceScorer` is very useful.

**Plans for L2E/L2EC**

Novelty:

- Might potentially still beat Winograd (by training LM, and evaluate ensemble)
- Not many have looked into why-question
- Can generate sequence to question

Bad parts:

- No new model/algorithm

**TODO**

1. CTX will be done tomorrow at 7pm (on Cocoserv2)
2. Train LSTM seq2seq on L2EC
   - Just as a base experiment to see if LSTM works better than Transformer on this kind of task...(could also be context gate's help) (compare by validation accuracy, it used to be 27%, but we want 37% start)
3. Jumping on the bandwagon, could try conv net
4. Use the trained LSTM to generate, and see if you still have lots of repeatitive target.

These data are actually old data...**because_ctx** is not the full dataset!!!! Be aware!! 

```bash
# LSTM with context gate for source
# ccn-cluster (node18 pa4) (accidental delete, retrained) (lm) (val acc: 34.97)
CUDA_VISIBLE_DEVICES=0 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_lstm_src_gate_650/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 -context_gate source \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_lstm_src_gate_650/log.txt -share_embeddings -share_decoder_embeddings

# make slightly bigger (650 -> 1000) (node18 train)
CUDA_VISIBLE_DEVICES=1 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_lstm_src_gate_1000/dissent \
    -layers 2 -rnn_size 1000 -word_vec_size 1000 -context_gate source \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_lstm_src_gate_1000/log.txt -share_embeddings -share_decoder_embeddings
    
# make bigger (750), higher dropout 0.3 (node18 train)
CUDA_VISIBLE_DEVICES=1 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_lstm_src_gate_750_lr2_warmup8k_d03/dissent \
    -layers 2 -rnn_size 750 -word_vec_size 750 -context_gate source \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.3 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_lstm_src_gate_750_lr2_warmup8k_d03/log.txt -share_embeddings -share_decoder_embeddings

# higher learning rate (node18 lm) (too large, did not work even a bit!!)
CUDA_VISIBLE_DEVICES=2 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_lstm_src_gate_650_sgd_lr2/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 -context_gate source \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim sgd -adam_beta2 0.998 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 -decay_steps 20000 \
    -valid_batch_size 32 -log_file save/because_ctx_lstm_src_gate_650_sgd_lr2/log.txt -share_embeddings -share_decoder_embeddings

# higher Noam learning rate, longer warm up (node18 dodo)
# hmmm, not sure if longer warmup is ideal...maybe not??
# early sign is not so well for this? 25 -> 28
# so maybe long warmup / high lr is a bad idea?
# beat 27 -> 31 -> 33.994
# not working (longer warmup + higher lr bad idea)
CUDA_VISIBLE_DEVICES=2 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_lstm_src_gate_650_noam5_warmup30k/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 -context_gate source \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 30000 -learning_rate 5 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_lstm_src_gate_650_noam5_warmup30k/log.txt -share_embeddings -share_decoder_embeddings

# no source gate, and see what it looks like (dodo) (val acc: 35.7492) (better than with source gate)
CUDA_VISIBLE_DEVICES=2 python3 train.py -data data/because_ctx/because_ctx -save_model save/because_ctx_lstm_650/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_lstm_650/log.txt -share_embeddings -share_decoder_embeddings

# ========= These are real data/experiments that we want =======

# Let's train this on L2E and see if it's still good (lm) (as a comparison to Transformer on L2E)
# val acc: 36.1648 (worse than Transformer, which can get to 38)
CUDA_VISIBLE_DEVICES=1 python3 train.py -data data/because_nmt -save_model save/because_lstm_650/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_lstm_650/log.txt -share_embeddings -share_decoder_embeddings
```

Transformer does not have a source gate implemented in OpenNMT. Improvement??? 

## L2E and L2EC Discrepancy Investigation

The discrepancy is largely caused by the length ratio check and potentially sentence length check. (Previously we realize that the `gigaword_en_flattened.txt` was an incomplete copy, but this problem has since been resolved.)

So the data right now:

L2E: 

NewsCrawl 1,886,580

Gigaword 740,973

L2EC:

NewsCrawl 1,771,701 (about 100k difference) (2,149,120 with context)

Gigaword _____________ (27h parsing)

Now we load in correct gigaword, we get `1850422`, which is exactly the same as before...so the processing script should be ok? Small discrepancies are fine! 

Now we regenerate L2EC training data and it will be correct this time!

```bash
python2 gen_because_ctx.py --out_prefix gigaword_newscrawl_ordered_ctx_2018nov5
```

Generate training data, and train LSTM tonight.

This is mostly L2EC training data!!

```bash
python3 preprocess.py -train_src data/because_ctx_full/gigaword_newscrawl_ordered_ctx_2018nov5-src-train.txt -train_tgt data/because_ctx_full/gigaword_newscrawl_ordered_ctx_2018nov5-tgt-train.txt -valid_src data/because_ctx_full/gigaword_newscrawl_ordered_ctx_2018nov5-src-valid.txt -valid_tgt data/because_ctx_full/gigaword_newscrawl_ordered_ctx_2018nov5-tgt-valid.txt \
 -save_data data/because_ctx_full/because_ctx -share_vocab -src_seq_length 400 -shard_size 0 # if we don't have shard_size=0, it will split into 3 shards

# pa4 (val acc 35.5024) (vanilla architecture)
CUDA_VISIBLE_DEVICES=0 python3 train.py -data data/because_ctx_full/because_ctx -save_model save/because_ctx_full_lstm_650/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_full_lstm_650/log.txt -share_embeddings -share_decoder_embeddings
    
# dodo, train on full data, no sharding, longer!
# val acc: 36.0095 (this is still lower than LSTM on L2E, also about 2% lower than Transformer)
CUDA_VISIBLE_DEVICES=0 python3 train.py -data data/because_ctx_full/because_ctx -save_model save/because_ctx_full_lstm_650_300ksteps/dissent \
    -layers 2 -rnn_size 650 -word_vec_size 650 \
    -encoder_type brnn -decoder_type rnn  \
    -train_steps 300000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_full_lstm_650_300ksteps/log.txt -share_embeddings -share_decoder_embeddings
    
# in order to have numbers, we will train a Transformer on this corpus
# (pa4) (we can have the comparison)
# val acc: 27.6896
CUDA_VISIBLE_DEVICES=0 python3 train.py -data data/because_ctx_full/because_ctx -save_model save/because_ctx_full_transformer_lay6_d01_30ksteps/dissent \
   -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 300000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 16 -log_file save/because_ctx_full_transformer_lay6_d01_30ksteps/log.txt -share_embeddings -share_decoder_embeddings
    
# at last we train a hybrid!!! A BLSTM - TransformerD hybrid!!
# the learning rate is indeed almost the same :) (a monster!! chimara)
# on L2EC
# (train) val acc: 32.2267
CUDA_VISIBLE_DEVICES=1 python3 train.py -data data/because_ctx_full/because_ctx -save_model save/because_ctx_full_blstm_transformer_lay2_d02_30ksteps/dissent \
   -layers 2 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type brnn -decoder_type transformer -position_encoding \
    -train_steps 300000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_full_blstm_transformer_lay2_d02_30ksteps/log.txt -share_embeddings -share_decoder_embeddings
    
# another hybrid with 2 layers LSTM with 4 layers Transformer
# on L2EC (lm)
# val acc: 30.8773
CUDA_VISIBLE_DEVICES=2 python3 train.py -data data/because_ctx_full/because_ctx -save_model save/because_ctx_full_blstm_transformer_lay4_d02_30ksteps/dissent \
   -enc_layers 2 -dec_layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type brnn -decoder_type transformer -position_encoding \
    -train_steps 300000  -max_generator_batches 2 -dropout 0.2 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_full_blstm_transformer_lay4_d02_30ksteps/log.txt -share_embeddings -share_decoder_embeddings
    
# we could further ablate LSTM size to 650 instead...with -enc_rnn_size and -dec_rnn_size (dodo)
# Val acc: 30.5732
CUDA_VISIBLE_DEVICES=3 python3 train.py -data data/because_ctx_full/because_ctx -save_model save/because_ctx_full_blstm_transformer_lay6_d01_30ksteps/dissent \
   -enc_layers 2 -dec_layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type brnn -decoder_type transformer -position_encoding \
    -train_steps 300000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 5 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 \
    -valid_batch_size 32 -log_file save/because_ctx_full_blstm_transformer_lay6_d01_30ksteps/log.txt -share_embeddings -share_decoder_embeddings
```

## Available Models

1. L2E: LSTM, Transformer (both finished training)
2. L2EC: LSTM (finished), Transformer (still running)
3. LM: LSTM, Transformer (maybe not necessary)
4. Conv2LM (LM1B)

We only evaluate model that does best on validation set (in terms of accuracy).

**todo:** for evaluation set (news commentary), filter out anything that can't pass `why()` function. (re-generate with questions in them)

**todo:** evaluate L2EC on validation set...

**todo:** evaluate all models on news commentary to see what happens

**todo:** use Conv2LM to generate for Winograd 62

**todo:** try to load ensemble models using OpenNMT to evaluate on Winograd

```bash
# Evaluate L2EC on validation set (NOT DONE YET)


# Evaluate L2E on News Commentary (iter 80000) (greedy) (running)
CUDA_VISIBLE_DEVICES=7 python3.6 translate.py -model save/because_transformer_sep5/dissent_step_80000.pt -src /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt -tgt /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-tgt-test.txt -share_vocab -output /home/anie/DisExtract/data/news_commentary/l2e_sep5_step_80000_pred_greedy.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 1

# Evaluate L2E on News Commentary (iter 200000) (greedy)
CUDA_VISIBLE_DEVICES=6 python3.6 translate.py -model save/because_transformer_sep5/dissent_step_200000.pt -src /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt -tgt /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-tgt-test.txt -share_vocab -output /home/anie/DisExtract/data/news_commentary/l2e_sep5_step_200000_pred_greedy.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 1
     
# Evaluate L2E on News Commentary (iter 80000) (Beam=5)
CUDA_VISIBLE_DEVICES=7 python3.6 translate.py -model save/because_transformer_sep5/dissent_step_80000.pt -src /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt -tgt /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-tgt-test.txt -share_vocab -output /home/anie/DisExtract/data/news_commentary/l2e_sep5_step_80000_pred_beam5.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 5

# Evaluate L2E on News Commentary (iter 200000) (Beam=5)
CUDA_VISIBLE_DEVICES=6 python3.6 translate.py -model save/because_transformer_sep5/dissent_step_200000.pt -src /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt -tgt /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-tgt-test.txt -share_vocab -output /home/anie/DisExtract/data/news_commentary/l2e_sep5_step_200000_pred_beam5.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 5

# Evaluate L2EC on News Commentary (iter 200000) (greedy)
CUDA_VISIBLE_DEVICES=6 python3.6 translate.py -model /mnt/fs5/anie/OpenNMT-py/save/because_ctx_full_lstm_650_300ksteps/dissent_step_300000.pt -src /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-ctx-src-test.txt -tgt /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-tgt-test.txt -share_vocab -output /home/anie/DisExtract/data/news_commentary/l2ec_sep5_step_300000_pred_greedy.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 1

# Evaluate L2EC on News Commentary (iter 200000) (Beam=5) (running, train)
CUDA_VISIBLE_DEVICES=7 python3.6 translate.py -model /mnt/fs5/anie/OpenNMT-py/save/because_ctx_full_lstm_650_300ksteps/dissent_step_300000.pt -src /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-ctx-src-test.txt -tgt /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-tgt-test.txt -share_vocab -output /home/anie/DisExtract/data/news_commentary/l2ec_sep5_step_300000_pred_beam5.txt -replace_unk \
     -report_bleu -report_rouge -batch_size 512 -gpu 0 -beam_size 5

# Evaluate FairSeq (ConvLM) on News Commentary (greedy)

cat /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt | CUDA_VISIBLE_DEVICES=6 python3.6 eval_lm.py /mnt/fs5/anie/fairseq/model/gbw_fconv_lm/ \
  --path /mnt/fs5/anie/fairseq/model/gbw_fconv_lm/model.pt \
  --beam 1 --task language_modeling --nbest 1 \
  --batch-size 1 --buffer-size 1 --remove-bpe --raw-text --replace-unk --sample-break-mode eos  | tee /home/anie/DisExtract/data/news_commentary/conv_lm_lm1b_greedy.txt

# Evaluate LM1B on News Commentary 
CUDA_VISIBLE_DEVICES=8 python gen_lm1b.py --dataset news_commentary --output_file /home/anie/DisExtract/data/news_commentary/src_lm1b_greedy.txt --print_sent --max_seq 50

# Evaluate LM1B on News Commentary with Context (this is sampled after our fix)
CUDA_VISIBLE_DEVICES=9 python gen_lm1b.py --dataset news_commentary_ctx --output_file /home/anie/DisExtract/data/news_commentary/ctx_src_lm1b_greedy.txt --print_sent --max_seq 50

# Evaluate LM1B on Winograd
CUDA_VISIBLE_DEVICES=9 python gen_lm1b.py --dataset winograd --output_file /home/anie/DisExtract/data/winograd/lm1b_greedy.txt

# Evaluate LM1B on CoPA
CUDA_VISIBLE_DEVICES=9 python gen_lm1b.py --dataset copa --output_file /home/anie/DisExtract/data/copa/lm1b_greedy.txt

# Evaluate OpenSubtitles (OpenNMT) on L2E test set

```

```
2 * (precision * recall) / (precision + recall)
```

Notes:

`-fast` does not evaluate gold PPL

**News Commentary Evaluation**

We evaluate on Gold PPL and BLEU score

| Model                                 | Gold PPL | Test BLEU                                                    |
| ------------------------------------- | -------- | ------------------------------------------------------------ |
| **l2e**_sep5_step_200000_pred_greedy  | 57.0265  | 0.55, 22.3/1.4/0.2/0.1 (BP=0.629, ratio=0.683, hyp_len=63182, ref_len=92493) |
| **l2e**_sep5_step_80000_pred_greedy   | 51.3911  | 0.53, 22.0/1.3/0.2/0.1 (BP=0.662, ratio=0.708, hyp_len=65449, ref_len=92493) |
| **l2e**_sep5_step_200000_pred_beam5   | 57.0265  | 0.37, 24.1/1.3/0.3/0.1 (BP=0.423, ratio=0.538, hyp_len=49719, ref_len=92493) |
| **l2e**_sep5_step_80000_pred_beam5    | 51.3911  | 0.35, 24.2/1.3/0.2/0.1 (BP=0.431, ratio=0.543, hyp_len=50240, ref_len=92493) |
| **l2ec**_sep5_step_300000_pred_greedy | 62.075   | 0.40, 25.1/1.7/0.2/0.0 (BP=0.522, ratio=0.606, hyp_len=56021, ref_len=92493) |
| **l2ec**_sep5_step_300000_pred_beam5  | 62.075   | 0.47, 26.7/1.7/0.4/0.1 (BP=0.396, ratio=0.519, hyp_len=48007, ref_len=92493) |

So PPL and BLEU are not entirely linked up

## LM1B

LM1B code is available. Can train using it on `node15-ccncluster`

```bash
CUDA_VISIBLE_DEVICES=0 python eval_lm1b.py --mode sample \
                             --prefix "I love that I" \
                             --pbtxt /mnt/fs5/anie/lm1b/tf-lm1b/graph-2016-09-10.pbtxt \
                             --vocab_file /mnt/fs5/anie/lm1b/tf-lm1b/vocab-2016-09-10.txt  \
                             --ckpt '/mnt/fs5/anie/lm1b/tf-lm1b/ckpt-*'
```

We evaluate LM1B on Winograd / COPA

```bash
# winograd
CUDA_VISIBLE_DEVICES=9 python gen_lm1b.py --dataset winograd --output_file /home/anie/DisExtract/data/winograd/lm1b_greedy.txt

# copa
CUDA_VISIBLE_DEVICES=9 python gen_lm1b.py --dataset copa --output_file /home/anie/DisExtract/data/copa/lm1b_greedy.txt
```

**todo**: Run LM1B on News Commentary with PPL (write additional code, directly get softmax result and compute PPL on target by yourself) and BLEU (with `perl` script or check OpenNMT).

**todo**: Run OpenSubtitle on Why-bot generated results.

**todo**: Generated questions kinda suck...what to do???

## Baseline Models

Use model trained on LM1B (State-of-the-art GCCV model)

### Side ideas: Why-QA

Use Causal net to select most important phrases of explanation (result) from the cause. Then delete `because`, put S1 and S2 back to paragraph. Then you actually do get a QA style selection dataset. Question would be naturalistic (but might overlap too much with S1...so cautious!!). Answers would be a short span inside the paragraph. 

Generate explain-bot dataset

```bash
python3 gen_why_test.py --data_file /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov6-src-test.txt --output_file /home/anie/DisExtract/data/news_commentary/news_commentary_v13_nov25-src-test-Q.txt
```

Generated Qs kinda suck??? Maybe Erin can improve it maybe she can't -- she might not have enough time.

## Processing News Commentary

```
python news_commentary.py --filter_because
python news_commentary.py --parse
python news_commentary.py --produce --out_prefix news_commentary_v13_nov6
```

After filtering: 9331 sentences (with context!!)

After parsing: 7086 sentences 

(Just a note, if we overlap S1 between L2EC and L2E's test sets, we get about the same amount 7k)

We lose 5 sentences because they have \t inside them...filtering did not tokenize anything.

Also filtering out s1/s2 that are too long or too short

Final test number of examples: 6301 (which is a good amount as external evaluation)

Run L2EC and L2E both on this dataset and see who wins. Need to wait until L2E LSTM finishes training...

## Processing Holdout Newscrawl Test set

In L2E/L2EC, we don't really have generalization...partially because how could you generalize to previously unknown causes?

We get 6,661 sentence pairs for evaluation that both overlap the test set of L2E and L2EC (guaranteed not in training). Out of these, **322** sentences have exact same S1, but multiple S2 (with different context). We can test how much context influences generation on these examples.

The problem is that now S2 does not entirely match in these 322 sentences...or for that matter, other sentences too. So only use this dataset if you have to!

We do not use this as test set. These examples are too problematic!!!

## Causal Net Eval

**Todo**: Use causal net to eval on winograd...

**Todo**: after LSTM finishes training, if the performance is about the same as Transformer, use it to generate winograd!!

## Evaluation Experiment

1. Winograd Schema Challenge
   - WSC choice evaluation (try to beat previous SoTA)
   - Generate by model (L2E model) (without context)
     - Greedy
     - Beam search
   - Evaluate by Commonsense LM
   - Evaluate by Causal Net
2. COPA
   - Choice evaluation 
   - Generate by model (L2E model) (without context)
   - Evaluate by Commonsense LM
   - Evaluate by Causal Net
3. L2EC based Seq2Seq

## TODO

For human eval:

The only corpora we need are 

- News Commentary (to evaluate with context and without context)
  - If not work, let's use the original test data (call it heldout dataset) (only use it if the result on news commentary is very sh*tty...)
- Winograd / COPA

