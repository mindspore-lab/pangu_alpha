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
import os
import sys

import mindspore as ms
import mindspore.communication.management as management
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore import load_distributed_checkpoint
from pathlib2 import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from pangu_alpha.model import LearningRate, TrainingMonitor, PANGUALPHA_BC, PANGUALPHA_Model, PANGUALPHAConfig, \
    get_ckpt_file_list, finetune_load_file, args_utils, set_parse, JIEBATokenizer
from pangu_alpha.afqmc.src.data_process import GetDataGenerator, ProcessData

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def finetuning(args_config):
    if rank_id % 8 == 0:
        finetune_load_file(args_config)
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
        strategy_ckpt_load_file=os.path.join(args_config.pretrained_model_path, args_config.ckpt_strategy_name),
        enable_parallel_optimizer=False)
    if rank_id % 8 == 0:
        ms.context.set_auto_parallel_context(
            strategy_ckpt_save_file=os.path.join(args_config.output_path, args_config.ckpt_strategy_name)
        )
    auto_parallel_context().set_loss_repeated_mean(True)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()

    model_parallel_num = args_config.model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = args_config.per_batch_size
    batch_size = per_batch_size * device_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_config.seq_length,
        vocab_size=args_config.vocab_size,
        embedding_size=args_config.embedding_size,
        num_layers=args_config.num_layers,
        num_heads=args_config.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        word_emb_dp=True,
        eod_reset=False,
        model_file_path=args_config.pretrained_model_path,
        position_embedding_name=args_config.position_embedding_name,
        word_embedding_name=args_config.word_embedding_name,
        top_query_embedding_name=args_config.top_query_embedding_name,
        sink_size=args_config.sink_size,
        label_size=2
    )
    print(config, flush=True)

    pangu_backbone = PANGUALPHA_Model(config)
    pangu = PANGUALPHA_BC(config, pangu_backbone)
    lr = LearningRate(learning_rate=6e-5,
                      end_learning_rate=6e-7,
                      warmup_steps=1000,
                      decay_steps=3000)
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
    loss = nn.SoftmaxCrossEntropyWithLogits()

    model = ms.Model(pangu, loss_fn=loss, optimizer=optimizer, metrics={'accuracy'})

    tokenizer = JIEBATokenizer(os.path.join(args_config.pretrained_model_path, args_config.vocab_vocab_name),
                               os.path.join(args_config.pretrained_model_path, args_config.vocab_model_name))

    t_data_ = ProcessData(args_config, tokenizer, train=True)
    t_data = GetDataGenerator(t_data_)
    train_dataset = ds.GeneratorDataset(t_data, ["input_ids", "label"], shuffle=True)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    e_data_ = ProcessData(args_config, tokenizer, train=False)
    e_data = GetDataGenerator(e_data_)
    eval_dataset = ds.GeneratorDataset(e_data, ["input_ids", "label"], shuffle=False)
    eval_dataset = eval_dataset.batch(batch_size)
    print(f'train data size is {len(t_data)}, evaluate data size is {len(e_data)}', flush=True)

    predict_layout = model.infer_train_layout(train_dataset=train_dataset, dataset_sink_mode=True,
                                              sink_size=config.sink_size)
    ckpt_file_list = get_ckpt_file_list(os.path.join(args_config.pretrained_model_path, args_config.ckpt_dir))

    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(args_config.pretrained_model_path, args_config.ckpt_strategy_name))

    ckpt_dir = os.path.join(args_config.output_path, f"rank_{str(rank_id)}")
    print("output_path : ", ckpt_dir, flush=True)

    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    print("append callback!", flush=True)
    config_ck = CheckpointConfig(save_checkpoint_steps=args_config.save_checkpoint_steps,
                                 keep_checkpoint_max=1, saved_network=pangu, integrated_save=False)
    prefix = f"afqmc_rank_{rank_id}"

    callback = [TrainingMonitor(model, eval_dataset, batch_size, device_num, config.sink_size,
                                args_config.per_step_print)]
    ckpoint_cb = ModelCheckpoint(prefix=prefix,
                                 directory=ckpt_dir,
                                 config=config_ck)

    callback.append(ckpoint_cb)

    model.train(epoch=args_config.epoch_size, train_dataset=train_dataset, dataset_sink_mode=True,
                sink_size=config.sink_size, callbacks=callback)


if __name__ == "__main__":
    print('process id:', os.getpid(), flush=True)

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args, flush=True)

    finetuning(args)
