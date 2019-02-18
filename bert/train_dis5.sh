#!/usr/bin/env bash

export DIS_DIR=/mnt/fs5/anie/DisSent-Processed-data
export CUDA_VISIBLE_DEVICES=9

python run_classifier.py \
  --task_name dis5 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DIS_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /mnt/fs5/anie/pytroch_bert_models/fine_tuned_dis5/