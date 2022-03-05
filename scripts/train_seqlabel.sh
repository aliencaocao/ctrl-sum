#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

data_bin="temp"
model="bert-large-cased"
LOADDIR="cnn_bert_tagger"
GPU=0
batch=8
save_steps=2000
eval_steps=1000
max_steps=20000
update_freq=4
lr=5e-5

SAVE=${LOADDIR}
extra="--tokenizer_name ${model}"
model=${LOADDIR}

mkdir -p datasets/${data_bin}/hf_cache

python3.8 ctrlsum/token-classification/main.py \
  --data_dir datasets/${data_bin}/ \
  --model_name_or_path ${model} \
  --output_dir ${SAVE} \
  --num_train_epochs 3 \
  --max_steps ${max_steps} \
  --max_seq_length 512 \
  --per_device_train_batch_size ${batch} \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps ${update_freq} \
  --save_steps ${save_steps} \
  --eval_steps ${eval_steps} \
  --threshold 0.1 \
  --learning_rate ${lr} \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --save_total_limit 10 \
  --seed 1 \
  --do_predict \
  --eval_split test \
  --disable_tqdm \
  --fp16 \
  ${extra}
