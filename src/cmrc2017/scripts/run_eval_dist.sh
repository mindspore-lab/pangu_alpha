#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_dist.sh [RANK_TABLE_FILE] [DEVICE_NUM] [DEVICE_START]"
echo "PER_BATCH DEVICE_START LOCAL_DEVICE_NUM"
echo "for example:"
echo "bash run_eval_dist.sh /path/hccl.json 4 0"
echo "It is better to use absolute path."
echo "=============================================================================================================="



LOCAL_PATH=$(dirname "$PWD")
echo LOCAL_PATH=$LOCAL_PATH
RANK_SIZE=$2
DEVICE_START=$3
export RANK_SIZE=$RANK_SIZE
export RANK_TABLE_FILE=$1

data_path=/store0/pangu_alpha/cmrc2017/test/
output_path=/store0/pangu_alpha/output_pangu/
ckpt_path=/store0/pangu_alpha/pretrained_models/2B6
eval_model_config=${LOCAL_PATH}/configs/eval_model_config_pangu_cmrc2017.yaml

for ((i = 0; i < ${RANK_SIZE}; i++)); do

  export RANK_ID=$i
  export DEVICE_ID=$((i + DEVICE_START))
  echo DEVICE_ID=$DEVICE_ID
  echo RANK_ID=$RANK_ID

  # eval
  tk evaluate --quiet \
              --boot_file_path=${LOCAL_PATH}/main/evaluate_main.py \
              --data_path=$data_path \
              --output_path=$output_path \
              --ckpt_path=$ckpt_path \
              --model_config_path=$eval_model_config  &

done
