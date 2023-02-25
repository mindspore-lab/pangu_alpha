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

import itertools
import json
import os

import mindspore as ms
import numpy as np


def load_tnews_example_for_shot(data_path, en2zh_labels, num_sample=6, np_rng=None, max_len=None):
    input_str_format = "这是关于{}的文章：{}"
    with open(data_path, 'r') as fid:
        data = [json.loads(x) for x in fid.readlines()]

    if np_rng is None:
        np_rng = np.random.default_rng()
    # select sample with balanced labels
    label_key = lambda x: x[1]
    sample_groupbyed = itertools.groupby(
        sorted([(x, en2zh_labels[y['label_desc']]) for x, y in enumerate(data)], key=label_key), key=label_key)
    group_index = [np.array([z[0] for z in y]) for x, y in sample_groupbyed]
    for x in group_index:
        np_rng.shuffle(x)  # in-place
    nums = (num_sample - 1) // len(group_index) + 1
    group_index_concated = np.concatenate([x[:nums] for x in group_index])
    np_rng.shuffle(group_index_concated)
    selected_index = group_index_concated[:num_sample]

    examples = []
    for x in selected_index:
        sentence = data[x]['sentence']
        example_formated = input_str_format.format(en2zh_labels[data[x]['label_desc']], sentence)[:max_len]
        examples.append(example_formated)
    ret = {
        'zero_shot': '',
        'one_shot': examples[0] + '\n',
        'few_shot': ('\n'.join(examples)) + '\n',
    }
    return ret


def ProcessData(config, seq_length, tokenizer, train):
    data = []
    if train:
        with open(os.path.join(config.data_path, config.data_train_name), "r", encoding="utf-8") as fid:
            data_set = [json.loads(x) for x in fid]
        print("train cases num {}:".format(len(data_set)))
    else:
        with open(os.path.join(config.data_path, config.data_dev_name), "r", encoding="utf-8") as fid:
            data_set = [json.loads(x) for x in fid]
        print("eval cases num {}:".format(len(data_set)))

    label_id_list = []
    with open(os.path.join(config.data_path, config.id2label_name), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            label_id = line_json['label']
            label_id_list.append(label_id)

    label_size = len(label_id_list)
    label_num_list = list(range(label_size))

    for instance in data_set:
        tmp0 = tokenizer.tokenize(f"这是关于什么类别的文章：")
        tmp1 = f"这是关于什么类别的文章：{instance['sentence']}"
        token_list = tokenizer.tokenize(tmp1)
        token_list = token_list[:config.seq_length]
        input_ids = tokenizer.convert_tokens_to_ids(token_list)
        mask = np.zeros(seq_length)
        mask[len(tmp0):len(input_ids)] = 1
        input_ids = np.pad(input_ids, ((0, seq_length - len(input_ids)),), 'constant', constant_values=(0, 9))
        label = np.zeros(label_size, int)
        OldLabel2NewLabel = dict(zip(label_id_list, label_num_list))
        label[OldLabel2NewLabel[instance['label']]] = 1
        data.append((ms.Tensor(input_ids, dtype=ms.int32), ms.Tensor(label.reshape(label_size, ), dtype=ms.float32)))
    return data


class GetDataGenerator:
    def __init__(self, data):
        self.__data = data

    def __getitem__(self, index):
        return self.__data[index]

    def __len__(self):
        return len(self.__data)
