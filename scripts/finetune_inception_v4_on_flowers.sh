#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/flowers-models/inception_v4

# Where the dataset is saved to.
DATASET_DIR=/tmp/flowers

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
  tar -xvf inception_v4_2016_09_09.tar.gz
  mv inception_v4.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt
  rm inception_v4_2016_09_09.tar.gz
fi

# Download the dataset
if [ ! -d "$DATASET_DIR" ]; then
  mkdir ${DATASET_DIR}
fi
if [ ! -f ${DATASET_DIR}/quiz_validation_00003-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_validation_00003-of-00004.tfrecord "https://d.pcs.baidu.com/file/e786118e02722782e201ebd46d9d9484?fid=2813131872-250528-487228939969963&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-c%2BrIBJEjtvBQbifNvJVaWuTVIUk%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=634292957#0"
fi
if [ ! -f ${DATASET_DIR}/quiz_validation_00002-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_validation_00002-of-00004.tfrecord "https://d.pcs.baidu.com/file/80e78d422593c7444dd28939f3da4664?fid=2813131872-250528-35599866675053&time=1514945911&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-q9Tq77BdrXCVXtYFvfgLSqKxgRs%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=154939859#1"
fi
if [ ! -f ${DATASET_DIR}/quiz_validation_00001-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_validation_00001-of-00004.tfrecord "https://d.pcs.baidu.com/file/e1142783e219814770a84f27c992f717?fid=2813131872-250528-699190862417997&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-czz%2B26JQqt0z1VSSHu1koDjfT9A%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=735752863#2"
fi
if [ ! -f ${DATASET_DIR}/quiz_validation_00000-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_validation_00000-of-00004.tfrecord "https://d.pcs.baidu.com/file/a684557fecef85ca6c13118e52e9a2a9?fid=2813131872-250528-640166493437537&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-9Aq23xYIPLBcpA8MulP1TXR7xjk%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=446837696#3"
fi
if [ ! -f ${DATASET_DIR}/quiz_train_00003-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_train_00003-of-00004.tfrecord "https://d.pcs.baidu.com/file/ed71d1d09ce16ba3fda91d32d899b092?fid=2813131872-250528-438475503588780&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-E0LeHS9JnyB8JyVikdFt6eHVzBI%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=369626520#4"
fi
if [ ! -f ${DATASET_DIR}/quiz_train_00002-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_train_00002-of-00004.tfrecord "https://d.pcs.baidu.com/file/20897b136e8602ffbe741a3eb18a6478?fid=2813131872-250528-694195130883145&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-tThTj8n3bLMcZ6q90JpNtCINl6w%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=321300069#5"
fi
if [ ! -f ${DATASET_DIR}/quiz_train_00001-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_train_00001-of-00004.tfrecord "https://d.pcs.baidu.com/file/064c60bc2813bafb19fe6f9d05c994b4?fid=2813131872-250528-57828877953216&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-4C0TNSS%2FBb6rbQvcoQsF8ro%2FkSo%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=221528701#6"
fi
if [ ! -f ${DATASET_DIR}/quiz_train_00000-of-00004.tfrecord ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/quiz_train_00000-of-00004.tfrecord "https://d.pcs.baidu.com/file/8dfd973b6bf8390840bf610152669e47?fid=2813131872-250528-834165246674701&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-CZwe8daX0ZE7xRTXv9ngLTdkOhc%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=472149194#7"
fi
if [ ! -f ${DATASET_DIR}/labels.txt ]; then
  wget -c --referer=https://pan.baidu.com/s/1c2hgVuS -O ${DATASET_DIR}/labels.txt "https://d.pcs.baidu.com/file/2a830b6765de4b3df027eaf4a83348ec?fid=2813131872-250528-799451967192523&time=1514945912&rt=sh&sign=FDTAERVY-DCb740ccc5511e5e8fedcff06b081203-JQ%2B43MK92niGntm9i%2F07Gtl2F5U%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=56347768709758300&dp-callid=0&r=489373940#8"
fi

# Fine-tune all the layers for 10 epoch.
python train_eval_image_classifier.py \
  --train_dir=${TRAIN_DIR}/ckpt \
  --dataset_name=quiz \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --batch_size=32 \
  --learning_rate=0.005 \
  --optimizer=rmsprop \
  --eval_dir=${TRAIN_DIR}/eval \
  --max_num_batches=128
