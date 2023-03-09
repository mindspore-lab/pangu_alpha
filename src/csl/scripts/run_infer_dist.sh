#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================

LOCAL_PATH=$(dirname "$PWD")
echo LOCAL_PATH=$LOCAL_PATH
RANK_SIZE=$2
RANK_START=$3
export RANK_SIZE=$RANK_SIZE
export RANK_TABLE_FILE=$1

boot_file_path=$LOCAL_PATH/main/infer_main.py
data_path=/store0/pangu_alpha/csl_public/
output_path=/store0/pangu_alpha/output_pangu/
ckpt_path=/store0/pangu_alpha/pretrained_models/2B6/
eval_model_config=$LOCAL_PATH/configs/infer_model_config_pangu_csl.yaml

for ((i = 0; i < ${RANK_SIZE}; i++)); do
  export RANK_ID=$i
  export DEVICE_ID=$((i + RANK_START))
  echo DEVICE_ID=$DEVICE_ID
  echo RANK_ID=$RANK_ID
  # evaluate
  tk evaluate --quiet \
              --boot_file_path=$boot_file_path \
              --data_path=$data_path \
              --output_path=$output_path \
              --ckpt_path=$ckpt_path  \
              --model_config_path=$eval_model_config &
done
