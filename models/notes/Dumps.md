## Contextual Learning to Explain

(L2EC)

We first process and move the files.

There are some need for post-processing due to web crawling issues.

We compute some simple stats on the files.

```bash
# preprocess the data
python3 preprocess.py -train_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-train.txt -train_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-train.txt -valid_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-valid.txt -valid_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-valid.txt -save_data data/because_qa/because_nmt_ctx_s1_s2_2018oct2 -share_vocab -src_seq_length 400

# preprocess the data with length limit shorter
python3 preprocess.py -train_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-train.txt -train_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-train.txt -valid_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-valid.txt -valid_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-valid.txt -save_data data/because_qa/because_nmt_src350_ctx_s1_s2_2018oct2 -share_vocab -src_seq_length 350

python3 preprocess.py -train_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-train.txt -train_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-train.txt -valid_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-valid.txt -valid_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-valid.txt -save_data data/because_qa/because_nmt_src300_ctx_s1_s2_2018oct2 -share_vocab -src_seq_length 300

# preprocess data for PGNet
python3 preprocess.py -train_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-train.txt -train_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-train.txt -valid_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-valid.txt -valid_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-valid.txt -save_data data/because_qa/because_nmt_ctx_s1_s2_2018oct2_dym -dynamic_dict -share_vocab -src_seq_length 400

python3 preprocess.py -train_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-train.txt -train_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-train.txt -valid_src data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-src-valid.txt -valid_tgt data/because_qa/gigaword_newscrawl_ctx_s1_s2_2018oct2-tgt-valid.txt -save_data data/because_qa/because_nmt_src300_ctx_s1_s2_2018oct2_dym -share_vocab -src_seq_length 300 -dynamic_dict

# running the model
CUDA_VISIBLE_DEVICES=0,1 python3 train.py -data data/because_qa/because_nmt_ctx_s1_s2_2018oct2 -save_model save/ctx_s1_s2_because_transformer_oct6/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1 -valid_batch_size 16 -log_file save/ctx_s1_s2_because_transformer_oct6/log.txt -share_embeddings
    
# training the PGNet
CUDA_VISIBLE_DEVICES=2,3 python3 train.py -data data/because_qa/because_nmt_ctx_s1_s2_2018oct2_dym -save_model save/ctx_s1_s2_because_transformer_copy_attn_oct6/dissent \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot -copy_attn  \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1 -valid_batch_size 16 -log_file save/ctx_s1_s2_because_transformer_copy_attn_oct6/log.txt -share_embeddings -master_port 11000
```

## 