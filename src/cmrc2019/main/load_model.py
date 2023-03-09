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
import os
import sys
import mindspore as ms
import mindspore.communication.management as management
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import numpy as np
from mindspore.nn import TransformerRecomputeConfig, TransformerOpParallelConfig
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.serialization import load_distributed_checkpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import PanguAlphaConfig, PanguAlpha, PanguAlphaWithLoss2, CrossEntropyLoss, VirtualDatasetOneInputCell
from src.utils.model_utils import eval_load_file, get_ckpt_file_list, infer_load_file

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def load_model(model_config, is_eval):
    if rank_id % 8 == 0:
        if is_eval:
            eval_load_file(model_config)
        else:
            infer_load_file(model_config)

    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, max_device_memory="30GB")
    if model_config.distribute == "True":
        management.init()
        device_num = int(management.get_group_size())
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num))
        ms.context.reset_auto_parallel_context()
        ms.context.set_auto_parallel_context(
            parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            strategy_ckpt_load_file=os.path.join(model_config.ckpt_path, model_config.ckpt_strategy_name))
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        device_num = 1

    model_parallel_num = device_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = model_config.per_batch_size * data_parallel_num

    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(
                                                      model_config.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  expert_parallel=model_config.expert_parallel_num,
                                                  pipeline_stage=model_config.stage_num,
                                                  micro_batch_num=model_config.micro_size,
                                                  optimizer_shard=bool(model_config.optimizer_shard),
                                                  vocab_emb_dp=bool(model_config.word_emb_dp),
                                                  recompute=recompute_config,
                                                  gradient_aggregation_group=model_config.gradient_aggregation_group)
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=model_config.seq_length,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.embedding_size,
        ffn_hidden_size=model_config.embedding_size * 4,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        expert_num=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        softmax_compute_type=ms.float16,
        use_past=False,
        eod_reset=False,
        load_ckpt_path=model_config.ckpt_path,
        position_embedding_name=model_config.position_embedding_name,
        word_embedding_name=model_config.word_embedding_name,
        top_query_embedding_name=model_config.top_query_embedding_name,
        sink_size=model_config.sink_size,
        parallel_config=parallel_config,
    )
    print(config, flush=True)

    pangu = PanguAlpha(config)
    pangu = PanguAlphaWithLoss2(config, pangu, CrossEntropyLoss(config), eos_token=9)
    pangu = VirtualDatasetOneInputCell(pangu)
    pangu.set_train(False)
    model = ms.Model(pangu)

    fake_input = Tensor(np.ones(shape=(config.batch_size, config.seq_length + 1)), mstype.int32)
    input_mask_ids = Tensor(np.zeros(shape=(config.batch_size, config.seq_length)), mstype.int32)

    if model_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input, input_mask_ids)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(model_config.ckpt_path, model_config.ckpt_dir))
    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(model_config.ckpt_path, model_config.ckpt_strategy_name))
    print("load model and ckpt success!!", flush=True)
    return model, config
