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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_dist.sh RANK_TABLE_FILE [DEVICE_NUM] [DEVICE_START]"
echo "for example:"
echo "bash run_dist_pd_infer.sh /path/hccl.json 4 0"
echo "It is better to use absolute path."
echo "=============================================================================================================="

LOCAL_PATH=$(dirname "$PWD")
echo LOCAL_PATH=$LOCAL_PATH
DEVICE_NUM=$2
DEVICE_START=$3
export DEVICE_NUM=$DEVICE_NUM
export RANK_SIZE=$DEVICE_NUM
export RANK_TABLE_FILE=$1

boot_file_path=${LOCAL_PATH}/main/infer_main.py
data_path=/store0/pangu_alpha/pf_cft/pd/
output_path=/store0/pangu_alpha/output_pangu/
model_config_path=${LOCAL_PATH}/configs/infer_model_config_pangu_pd.yaml
pretrained_model_path=/store0/pangu_alpha/pretrained_models/2B6/

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export RANK_ID=$i
  export DEVICE_ID=$((i + DEVICE_START))
  echo DEVICE_ID=$DEVICE_ID
  echo RANK_ID=$RANK_ID
  # infer
  tk infer --quiet                         \
    --boot_file_path=$boot_file_path       \
    --data_path=$data_path                 \
    --output_path=$output_path             \
    --model_config_path=$model_config_path \
    --ckpt_path=$pretrained_model_path &
done
