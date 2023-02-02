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
import logging
import os
import sys

import mindspore as ms
import mindspore.communication.management as management
import numpy as np
from mindspore import load_distributed_checkpoint, set_algo_parameters
from mindspore.nn import TransformerRecomputeConfig, TransformerOpParallelConfig
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import PanguAlphaConfig, PanguAlpha, PanguAlphaWithLoss2, set_parse, CrossEntropyLoss,\
    VirtualDatasetOneInputCell
from src.utils.model_utils import eval_load_file, get_ckpt_file_list
from src.utils.tokenization_jieba import JIEBATokenizer
from src.utils import args_utils
from src.afqmc.main.data_process import load_afqmc_train_example_for_shot

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def few_shot_learning(eval_config):
    if rank_id % 8 == 0:
        eval_load_file(eval_config)

    res_dir = os.path.join(eval_config.output_path, 'eval_result_rank_' + str(device_id))
    eval_file_name = "eval_result.json"

    eval_out_file = os.path.join(eval_config.output_path, eval_file_name)
    f_out = open(eval_out_file, 'a')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    print("result file:", eval_out_file)

    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, max_device_memory="30GB")
    if eval_config.distribute == "True":
        management.init()
        device_num = int(management.get_group_size())
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num))
        ms.context.reset_auto_parallel_context()
        ms.context.set_auto_parallel_context(
            parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            strategy_ckpt_load_file=os.path.join(eval_config.ckpt_path, eval_config.ckpt_strategy_name),
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
    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(
                                                      eval_config.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  expert_parallel=eval_config.expert_parallel_num,
                                                  pipeline_stage=eval_config.stage_num,
                                                  micro_batch_num=eval_config.micro_size,
                                                  optimizer_shard=bool(eval_config.optimizer_shard),
                                                  vocab_emb_dp=bool(eval_config.word_emb_dp),
                                                  recompute=recompute_config,
                                                  gradient_aggregation_group=eval_config.gradient_aggregation_group)
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=eval_config.seq_length,
        vocab_size=eval_config.vocab_size,
        hidden_size=eval_config.embedding_size,
        ffn_hidden_size=eval_config.embedding_size * 4,
        num_layers=eval_config.num_layers,
        num_heads=eval_config.num_heads,
        expert_num=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        softmax_compute_type=ms.float16,
        use_past=False,
        eod_reset=False,
        load_ckpt_path=eval_config.ckpt_path,
        position_embedding_name=eval_config.position_embedding_name,
        word_embedding_name=eval_config.word_embedding_name,
        top_query_embedding_name=eval_config.top_query_embedding_name,
        sink_size=eval_config.sink_size,
        parallel_config=parallel_config,
    )
    print(config, flush=True)

    pangu = PanguAlpha(config)
    pangu = PanguAlphaWithLoss2(config, pangu, CrossEntropyLoss(config))
    pangu = VirtualDatasetOneInputCell(pangu)
    pangu.set_train(False)
    model = ms.Model(pangu)

    fake_input = ms.Tensor(np.ones(shape=(batch_size, config.seq_length + 1)), ms.int32)
    mask_ids = ms.Tensor(np.ones(shape=(batch_size, config.seq_length)), ms.float32)

    if eval_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input, mask_ids)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(eval_config.ckpt_path, eval_config.ckpt_dir))
    print(pangu.parameters_and_names())
    load_distributed_checkpoint(pangu,
                                ckpt_file_list,
                                predict_layout,
                                os.path.join(eval_config.ckpt_path, eval_config.ckpt_strategy_name))
    print(".............load ckpt success!!!.................")

    tokenizer = JIEBATokenizer(os.path.join(eval_config.ckpt_path, eval_config.vocab_model_name))

    with open(os.path.join(eval_config.data_path, eval_config.data_dev_name), "r", encoding="utf-8") as fid:
        ground_truth = [json.loads(x) for x in fid]
    print(f"All test case num:", len(ground_truth), flush=True)

    # shot choose ['one_shot', 'two_shot', 'three_shot', 'four_shot']
    res = list()
    for task in ["one_shot"]:
        print(task + " task is processing!")
        eval_result_path = os.path.join(res_dir, task + '_' + eval_file_name)
        fw = open(eval_result_path, 'a+')
        res_dict = dict()
        res_dict[task] = task + " task is processing!"

        shot_to_example = load_afqmc_train_example_for_shot()
        example = shot_to_example[task]

        z0 = []
        for instance in tqdm(ground_truth):
            instance_tf_label = ['不同', '相同']
            input_ids_list = []
            mask_list = []
            label_list = []
            for label_i in instance_tf_label:
                tmp0 = tokenizer.tokenize(f"{example}下面两个句子语义{label_i}：")
                tmp1 = f"{example}下面两个句子语义{label_i}：“{instance['sentence1']}”；“{instance['sentence2']}”"
                token_list = tokenizer.tokenize(tmp1)
                token_list = token_list[:config.seq_length]
                input_ids = tokenizer.convert_tokens_to_ids(token_list)
                mask = np.zeros(config.seq_length)
                mask[len(tmp0):len(input_ids)] = 1
                input_ids = np.pad(input_ids, ((0, config.seq_length + 1 - len(input_ids)),), 'constant',
                                   constant_values=(0, 9))
                input_ids_list.append(input_ids)
                mask_list.append(mask)
                label_list.append(label_i)
            tmp0 = [ms.Tensor(x[np.newaxis], dtype=ms.int32) for x in input_ids_list]
            tmp1 = [ms.Tensor(x[np.newaxis], dtype=ms.float32) for x in mask_list]
            label = int(instance['label'])
            z0.append((tmp0, tmp1, label))
        results = []
        cnt = 0
        correct_num = 0
        logging.info('start compute acc')
        for ind0, (input_ids_list, mask_list, label) in tqdm(enumerate(z0)):
            cnt += 1
            loss = np.concatenate([model.predict(x, y).asnumpy() for x, y in zip(input_ids_list, mask_list)])
            results.append(np.argmin(loss))
            if np.argmin(loss) == label:
                correct_num += 1
            if ind0 % 100 == 0:
                print(f'[{ind0}/{len(ground_truth)}] eval-{task}: cnt={cnt} acc={correct_num / cnt}', loss, flush=True)
                res_dict[
                    f'[{ind0}/{len(ground_truth)}] eval-{task}'] = f'cnt={cnt} acc={correct_num / cnt}' + ', ' + str(
                    loss)
        res_dict[f'eval-{task}'] = f'cnt={cnt} acc={correct_num / cnt}'
        res.append(res_dict)
        logging.info(f'eval-{task}: cnt={cnt} acc={correct_num / cnt}')
        json.dump(res_dict, fw)
        fw.close()
    json.dump(res, f_out)
    f_out.close()
    if os.path.isfile(eval_out_file):
        os.chmod(eval_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_learning(args)
