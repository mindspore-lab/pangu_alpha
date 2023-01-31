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
echo "for example:"
echo "bash run_distributed_train_and_eval.sh /path/hccl.json 4 0"
echo "It is better to use absolute path."
echo "=============================================================================================================="

LOCAL_PATH=$(dirname "$PWD")
echo LOCAL_PATH=$LOCAL_PATH
RANK_SIZE=$2
DEVICE_START=$3
export RANK_SIZE=$RANK_SIZE
export RANK_TABLE_FILE=$1

data_path=/path/pangu/afqmc/dataset/
output_path=/path/pycharmproject/output/afqmc_13B/eval/
ckpt_path=/path/pangu/13B
model_config_path=${LOCAL_PATH}/configs/eval_model_config_pangu_13B_afqmc.yaml

for ((i = 0; i < ${RANK_SIZE}; i++)); do
  export RANK_ID=$i
  export DEVICE_ID=$((i + DEVICE_START))
  echo DEVICE_ID=$DEVICE_ID
  echo RANK_ID="$RANK_ID"
  # infer
  tk infer --quiet \
           --boot_file_path="${LOCAL_PATH}"/main/infer_main.py \
           --data_path=$data_path \
           --output_path=$output_path \
           --ckpt_path=$ckpt_path \
           --model_config_path="$model_config_path" &
done
