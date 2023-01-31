# -*- coding: UTF-8 -*-
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
import json
import os

import mindspore as ms
import numpy as np


def ProcessData(config, tokenizer, train):
    data = []
    if train:
        with open(os.path.join(config.data_path, config.data_train_name), "r",
                  encoding="utf-8") as fid:
            data_set = [json.loads(x) for x in fid]
        print("train cases num {}:".format(len(data_set)))
    else:
        with open(os.path.join(config.data_path, config.data_dev_name), "r",
                  encoding="utf-8") as fid:
            data_set = [json.loads(x) for x in fid]
        print("eval cases num {}:".format(len(data_set)))

    for instance in data_set:
        # 下面两个句子语义相同：“借呗怎么没有十二个月分期”；“借呗最长分期,不是十二个月吗”'。下面两个句子语义是否相同？
        tmp0 = tokenizer.tokenize(f"")
        tmp1 = f"“{instance['sentence1']}”；“{instance['sentence2']}”"
        input_ids = tokenizer.tokenize(tmp1)[:config.seq_length]
        mask = np.zeros(config.seq_length)
        mask[len(tmp0):len(input_ids)] = 1
        input_ids = np.pad(input_ids, ((0, config.seq_length - len(input_ids)),), 'constant', constant_values=(0, 9))
        label = np.array([0, 0])
        label[int(instance['label'])] = 1
        data.append((ms.Tensor(input_ids, dtype=ms.int32), ms.Tensor(label.reshape(2, ), dtype=ms.float32)))
    return data


class GetDataGenerator:
    def __init__(self, data):
        self.__data = data

    def __getitem__(self, index):
        return self.__data[index]

    def __len__(self):
        return len(self.__data)


def load_afqmc_train_example_for_shot():
    examples = ['下面两个句子语义相同：“借呗怎么没有十二个月分期”；“借呗最长分期,不是十二个月吗”',
                '下面两个句子语义不同：“花呗可不可以推迟还债”；“现在花呗还不了,能延迟吗”',
                '下面两个句子语义不同：“现在怎么没有蚂蚁借呗了”；“我为什么蚂蚁借呗不见了”',
                '下面两个句子语义相同：“蚂蚁借呗的每月等额是每个月还多少”；“蚂蚁借呗最多几个月还清”']
    ret = {
        'zero_shot': '',
        'one_shot': examples[0] + '\n',
        'two_shot': examples[0] + '\n' + examples[1] + '\n',
        'three_shot': examples[0] + '\n' + examples[1] + '\n' + examples[2] + '\n',
        'four_shot': examples[0] + '\n' + examples[1] + '\n' + examples[2] + '\n' + examples[3] + '\n',
    }
    return ret
