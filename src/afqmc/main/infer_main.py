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
import sys

import mindspore as ms
import mindspore.communication.management as management
import numpy as np
from mindspore import load_distributed_checkpoint, set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model.base_modules import CrossEntropyLoss, VirtualDatasetOneInputCell
from src.model.pangu_alpha_model import PanguAlpha, PanguAlpha_Model, PanguAlphaWithLoss
from src.model.pangu_alpha_model_config import PanguAlphaConfig, set_parse
from src.utils import args_utils
from src.utils.model_utils import get_ckpt_file_list, infer_load_file
from src.utils.tokenizer_jieba import JIEBATokenizer
from src.afqmc.main.data_process import load_afqmc_train_example_for_shot

os.environ['HCCL_CONNECT_TIMEOUT'] = '120'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def few_shot_infer(infer_config):
    if rank_id % 8 == 0:
        infer_load_file(infer_config)

    infer_file_name = "infer_result.json"
    infer_out_file = os.path.join(infer_config.output_path, infer_file_name)

    f_out = open(infer_out_file, 'a', encoding='utf-8')

    print("result file:", infer_out_file, flush=True)

    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, max_device_memory="30GB")
    if infer_config.distribute == "True":
        management.init()
        device_num = int(management.get_group_size())
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num), flush=True)
        ms.context.reset_auto_parallel_context()
        ms.context.set_auto_parallel_context(
            parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            strategy_ckpt_load_file=os.path.join(infer_config.ckpt_path, infer_config.ckpt_strategy_name),
            enable_parallel_optimizer=False)
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        device_num = 1

    model_parallel_num = device_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num
    config = PanguAlphaConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=infer_config.seq_length,
        vocab_size=infer_config.vocab_size,
        embedding_size=infer_config.embedding_size,
        num_layers=infer_config.num_layers,
        num_heads=infer_config.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        word_emb_dp=True,
        eod_reset=False,
        model_file_path=infer_config.ckpt_path,
        word_embedding_name=infer_config.word_embedding_name,
        top_query_embedding_name=infer_config.top_query_embedding_name,
        position_embedding_name=infer_config.position_embedding_name
    )

    pangu_backbone = PanguAlpha_Model(config)
    pangu = PanguAlpha(config, pangu_backbone)
    pangu = PanguAlphaWithLoss(config, pangu, CrossEntropyLoss(config), eos_token=9)
    pangu = VirtualDatasetOneInputCell(pangu)
    pangu.set_train(False)
    model = ms.Model(pangu)

    fake_input = ms.Tensor(np.ones(shape=(batch_size, config.seq_length + 1)), ms.int32)
    mask_ids = ms.Tensor(np.ones(shape=(batch_size, config.seq_length)), ms.float32)

    if infer_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input, mask_ids)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(infer_config.ckpt_path, infer_config.ckpt_dir))
    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(infer_config.ckpt_path, infer_config.ckpt_strategy_name))

    tokenizer = JIEBATokenizer(os.path.join(infer_config.ckpt_path, infer_config.vocab_vocab_name),
                               os.path.join(infer_config.ckpt_path, infer_config.vocab_model_name))

    with open(os.path.join(infer_config.data_path, infer_config.data_test_name), "r", encoding="utf-8") as fid:
        ground_truth = [json.loads(x) for x in fid]
    print(f"data_test file path is :", os.path.join(infer_config.data_path, infer_config.data_test_name), flush=True)
    print(f"All test case num:", len(ground_truth), flush=True)

    # shot choose ['zero_shot', 'two_shot', 'three_shot', 'four_shot']
    res_list = list()
    task_list = ['one_shot']
    for task in task_list:
        res_dict = dict()
        res_dict[task] = []
        print(task + " task is processing!", flush=True)

        shot_to_example = load_afqmc_train_example_for_shot()
        example = shot_to_example[task]

        instance_tf_label = ['不同', '相同']
        for instance in tqdm(ground_truth):
            input_ids_list = []
            mask_list = []
            label_list = []
            for label_i in instance_tf_label:
                tmp0 = tokenizer.tokenize(f"{example}下面两个句子语义{label_i}：")
                tmp1 = f"{example}下面两个句子语义{label_i}：“{instance['sentence1']}”；“{instance['sentence2']}”"
                input_ids = tokenizer.tokenize(tmp1)[:config.seq_length]
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
            label = instance_tf_label[np.argmin(loss)]
            tmp_dict = dict()
            tmp_dict['sentence1'] = instance['sentence1']
            tmp_dict['sentence2'] = instance['sentence2']
            tmp_dict['label'] = label
            res_dict[task].append(tmp_dict)
        res_list.append(res_dict)

    json.dump(res_list, f_out, ensure_ascii=False)
    f_out.close()
    if os.path.isfile(infer_out_file):
        os.chmod(infer_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())
    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_infer(args)
