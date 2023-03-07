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
import sys

import mindspore as ms
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import set_parse
from src.utils import args_utils
from src.utils.tokenization_jieba import JIEBATokenizer
from src.iflytek.main.data_process import load_iflytek_train_example_for_shot
from src.iflytek.main.load_model import load_model

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def few_shot_infer(infer_config):
    model, config = load_model(infer_config, False)

    tokenizer = JIEBATokenizer(os.path.join(infer_config.ckpt_path, infer_config.vocab_model_name))
    infer_file_name = "infer_result.json"
    infer_out_file = os.path.join(infer_config.output_path, infer_file_name)

    print("result file:", infer_out_file, flush=True)

    with open(os.path.join(infer_config.data_path, infer_config.data_train_name), "r", encoding="utf-8") as fid:
        ground_truth = [json.loads(x) for x in fid]
    id_to_label = {int(x['label']): x['label_des'] for x in ground_truth}
    assert set(id_to_label.keys()) == set(range(len(id_to_label)))

    tmp0 = [
        ('task', ['zero_shot']),  # 'zero_shot','one_shot','few_shot'
        ('max_len', [25]),  # None,200,100
        ('tag_new_example', [True]),  # True, False
        ('zero_shot_num_sample', [3]),  # 2,3,4
        ('np_seed', [233]),  # 233,235,237,239
        ('new_mask', [False]),  # True, False
        ('input_str_format', [
            # "{label}：{sentence}",
            "这是关于{label}的应用程序：{sentence}",
        ])
    ]
    para_config_list = [{y0[0]: y1 for y0, y1 in zip(tmp0, x)} for x in itertools.product(*[x[1] for x in tmp0])]
    para_config = para_config_list[0]
    task = para_config['task']
    max_len = para_config['max_len']
    tag_new_example = para_config['tag_new_example']
    zero_shot_num_sample = para_config['zero_shot_num_sample']
    np_seed = para_config['np_seed']
    new_mask = para_config['new_mask']
    input_str_format = para_config['input_str_format']
    input_str_format_mask = input_str_format.rsplit('{', 1)[0]
    input_str_format_mask_tag_label = '{label}' in input_str_format_mask
    np_rng = np.random.default_rng(seed=np_seed)  # must be same across model-parallel

    instance_tf_label = []
    with open(os.path.join(infer_config.data_path, infer_config.id2label_name), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            zh_label = line_json['label_des']
            instance_tf_label.append(zh_label)
    print('instance_tf_label is : ', instance_tf_label)

    with open(os.path.join(infer_config.data_path, infer_config.data_test_name), "r", encoding="utf-8") as fid:
        tmp0 = [json.loads(x) for x in fid][:infer_config.test_sample_num]
        ground_truth = [tmp0[x] for x in np_rng.permutation(len(tmp0))]

    zc_print_ind = 0
    result_list = []
    if not tag_new_example:
        shot_to_example = load_iflytek_train_example_for_shot(infer_config.data_path, num_sample=zero_shot_num_sample,
                                                              np_rng=np_rng, max_len=max_len,
                                                              input_str_format=input_str_format)
        example = shot_to_example[task]
    for instance in ground_truth:
        zc_print_ind += 1
        if tag_new_example:
            shot_to_example = load_iflytek_train_example_for_shot(infer_config.data_path,
                                                                  num_sample=zero_shot_num_sample,
                                                                  np_rng=np_rng, max_len=max_len,
                                                                  input_str_format=input_str_format)
            example = shot_to_example[task]

        input_ids_list = []
        mask_list = []
        label_list = []
        input_str_list = []
        for label_i in instance_tf_label:
            if new_mask:
                tmp0 = tokenizer.tokenize(example)
            else:
                if input_str_format_mask_tag_label:
                    tmp0 = example + input_str_format_mask.format(label=label_i)
                else:
                    tmp0 = example + input_str_format_mask.format(sentence=instance['sentence'])
                tmp0 = tokenizer.tokenize(tmp0)
            tmp1 = example + input_str_format.format(label=label_i, sentence=instance['sentence'])
            tmp1 = tokenizer.tokenize(tmp1)
            input_ids = tokenizer.convert_tokens_to_ids(tmp1)[:config.seq_length]
            input_str_list.append(tmp1)

            mask = np.zeros(config.seq_length)
            mask[len(tmp0):len(input_ids)] = 1
            input_ids = np.pad(input_ids, ((0, config.seq_length + 1 - len(input_ids)),), 'constant',
                               constant_values=(0, 9))
            input_ids_list.append(input_ids)
            mask_list.append(mask)
            label_list.append(label_i)

        tmp0 = [ms.Tensor(x[np.newaxis], dtype=ms.int32) for x in input_ids_list]
        tmp1 = [ms.Tensor(x[np.newaxis], dtype=ms.float32) for x in mask_list]

        loss = np.concatenate([model.predict(x, y).asnumpy() for x, y in zip(tmp0, tmp1)])
        one_result = {'长文章': instance['sentence'], '应用程序标签': instance_tf_label[np.argmin(loss)]}
        result_list.append(one_result)

    all_results = {'result': result_list}
    with open(infer_out_file, 'w', encoding='utf-8') as out:
        json.dump(all_results, out, ensure_ascii=False)

    if os.path.isfile(infer_out_file):
        os.chmod(infer_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())
    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_infer(args)
