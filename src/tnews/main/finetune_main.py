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
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore.nn import TransformerRecomputeConfig, TransformerOpParallelConfig
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore import load_distributed_checkpoint
from pathlib2 import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import PanguAlphaConfig, PanguAlpha_Model, LearningRate, set_parse, PanguAlpha_BC, TrainingMonitor
from src.utils import args_utils
from src.utils.model_utils import get_ckpt_file_list, finetune_load_file
from src.utils.tokenization_jieba import JIEBATokenizer
from src.tnews.main.data_process import ProcessData, GetDataGenerator

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def finetuning(finetune_config):
    if rank_id % 8 == 0:
        finetune_load_file(finetune_config)
    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, max_device_memory="30GB")
    management.init()
    device_num = int(management.get_group_size())
    print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num))
    ms.context.reset_auto_parallel_context()
    ms.context.set_auto_parallel_context(
        parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
        gradients_mean=False,
        device_num=device_num,
        full_batch=True,
        strategy_ckpt_load_file=os.path.join(finetune_config.pretrained_model_path, finetune_config.ckpt_strategy_name))
    if rank_id % 8 == 0:
        ms.context.set_auto_parallel_context(
            strategy_ckpt_save_file=os.path.join(finetune_config.output_path, finetune_config.ckpt_strategy_name)
        )
    auto_parallel_context().set_loss_repeated_mean(True)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()

    model_parallel_num = finetune_config.model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = finetune_config.per_batch_size
    batch_size = per_batch_size * device_num

    # labels_ch.json 为标签翻译后结果，在原标签与ID映射文件中新增一列 label_desc_ch 为对应标签的中文类别
    with open(os.path.join(finetune_config.data_path, finetune_config.id2label_name), "r", encoding="utf-8") as fid:
        id2label_data = [json.loads(x) for x in fid]
    label_size = len(id2label_data)

    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(
                                                      finetune_config.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  expert_parallel=finetune_config.expert_parallel_num,
                                                  pipeline_stage=finetune_config.stage_num,
                                                  micro_batch_num=finetune_config.micro_size,
                                                  optimizer_shard=bool(finetune_config.optimizer_shard),
                                                  vocab_emb_dp=bool(finetune_config.word_emb_dp),
                                                  recompute=recompute_config,
                                                  gradient_aggregation_group=finetune_config.gradient_aggregation_group)
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=finetune_config.seq_length,
        vocab_size=finetune_config.vocab_size,
        hidden_size=finetune_config.embedding_size,
        ffn_hidden_size=finetune_config.embedding_size * 4,
        num_layers=finetune_config.num_layers,
        num_heads=finetune_config.num_heads,
        expert_num=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        softmax_compute_type=ms.float16,
        use_past=False,
        eod_reset=False,
        load_ckpt_path=finetune_config.pretrained_model_path,
        position_embedding_name=finetune_config.position_embedding_name,
        word_embedding_name=finetune_config.word_embedding_name,
        top_query_embedding_name=finetune_config.top_query_embedding_name,
        sink_size=finetune_config.sink_size,
        label_size=label_size,
        parallel_config=parallel_config
    )
    print(config, flush=True)

    pangu_backbone = PanguAlpha_Model(config)
    pangu = PanguAlpha_BC(config, pangu_backbone)
    lr = LearningRate(learning_rate=6e-5,
                      end_learning_rate=6e-7,
                      warmup_steps=500,
                      decay_steps=1000)
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = pangu.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    # loss = nn.CrossEntropyLoss()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    tokenizer = JIEBATokenizer(os.path.join(finetune_config.pretrained_model_path, finetune_config.vocab_model_name))

    model = ms.Model(pangu, loss_fn=loss, optimizer=optimizer, metrics={'accuracy'})

    t_data_ = ProcessData(finetune_config, config.seq_length, tokenizer, train=True)
    t_data = GetDataGenerator(t_data_)
    train_dataset = ds.GeneratorDataset(t_data, ["input_ids", "label"], shuffle=True)
    train_dataset = train_dataset.shuffle(buffer_size=500)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    e_data_ = ProcessData(finetune_config, config.seq_length, tokenizer, train=False)
    e_data = GetDataGenerator(e_data_)
    eval_dataset = ds.GeneratorDataset(e_data, ["input_ids", "label"], shuffle=False)
    eval_dataset = eval_dataset.batch(batch_size)
    print(f'train data size is {len(t_data)}, evaluate data size is {len(e_data)}', flush=True)

    predict_layout = model.infer_train_layout(train_dataset=train_dataset, dataset_sink_mode=True,
                                              sink_size=config.sink_size)

    ckpt_file_list = get_ckpt_file_list(os.path.join(finetune_config.pretrained_model_path, finetune_config.ckpt_dir))

    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(finetune_config.pretrained_model_path, finetune_config.ckpt_strategy_name))

    callback = [TrainingMonitor(model, eval_dataset, batch_size, device_num, config.sink_size,
                                finetune_config.per_step_print)]

    ckpt_dir = os.path.join(finetune_config.output_path, f"rank_{str(rank_id)}")
    print("output_path : ", ckpt_dir)
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    print("append callback!")
    config_ck = CheckpointConfig(save_checkpoint_steps=finetune_config.save_checkpoint_steps,
                                 keep_checkpoint_max=1, saved_network=pangu, integrated_save=False)

    prefix = f"tnews_rank_{rank_id}"

    ckpoint_cb = ModelCheckpoint(prefix=prefix,
                                 directory=ckpt_dir,
                                 config=config_ck)
    callback.append(ckpoint_cb)

    model.train(epoch=finetune_config.epoch_size, train_dataset=train_dataset, dataset_sink_mode=True,
                sink_size=config.sink_size, callbacks=callback)


if __name__ == "__main__":
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    finetuning(args)
