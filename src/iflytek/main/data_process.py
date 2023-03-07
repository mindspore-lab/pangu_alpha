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

import numpy as np


def load_iflytek_train_example_for_shot(data_path, num_sample=2, np_rng=None, max_len=None, input_str_format=None):
    if input_str_format is None:
        input_str_format = "这是关于{label}的应用程序：{sentence}"
    # input_str_format = "{s}：{label}"
    if np_rng is None:
        np_rng = np.random.default_rng()

    zc_cache = []
    if len(zc_cache) > 0:
        z0 = zc_cache[0]
    else:
        tmp0 = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith('train')]
        assert len(tmp0) == 1
        train_file = tmp0[0]
        with open(train_file, 'r') as fid:
            z0 = [json.loads(x) for x in fid.readlines()]
        zc_cache.append(z0)

    # select sample with balanced labels
    hf0 = lambda x: x[1]
    tmp0 = itertools.groupby(sorted([(x, y['label_des']) for x, y in enumerate(z0)], key=hf0), key=hf0)
    group_index = [np.array([z[0] for z in y]) for x, y in tmp0]
    for x in group_index:
        np_rng.shuffle(x)  # in-place
    tmp0 = (num_sample - 1) // len(group_index) + 1
    tmp1 = np.concatenate([x[:tmp0] for x in group_index])
    np_rng.shuffle(tmp1)
    selected_index = tmp1[:num_sample]

    examples = []
    for x in selected_index:
        sentence = z0[x]['sentence'] if max_len is None else z0[x]['sentence'][:max_len]
        tmp0 = input_str_format.format(label=z0[x]['label_des'], sentence=sentence)
        examples.append(tmp0)
    ret = {
        'zero_shot': '',
        'one_shot': examples[0] + '\n',
        'few_shot': ('\n'.join(examples)) + '\n',
    }
    return ret
