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
"""
network config setting
"""
import mindspore.common.dtype as mstype


class PanguAlphaConfig:
    """
    PANGUALPHA config class which defines the model size
    position_embedding_name, word_embedding_name, top_query_embedding_name 对应文件一般存放在model_file_path路径下
    """

    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
                 model_file_path,
                 position_embedding_name,
                 word_embedding_name,
                 top_query_embedding_name,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 self_layernorm=True,
                 forward_reduce_scatter=True,
                 word_emb_dp=True,
                 stage_num=16,
                 eod_reset=True,
                 micro_size=32,
                 sink_size=2,
                 label_size=2
                 ):
        self.model_file_path = model_file_path
        self.position_embedding_name = position_embedding_name
        self.word_embedding_name = word_embedding_name
        self.top_query_embedding_name = top_query_embedding_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        self.use_past = use_past
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        self.self_layernorm = self_layernorm
        self.forward_reduce_scatter = forward_reduce_scatter
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp
        self.eod_reset = eod_reset
        self.sink_size = sink_size
        self.label_size = label_size

    def __str__(self):
        info = "[PANGUALPHAConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 20
        return info


def set_parse_13B(args_opt):
    r"""
        Set config for 13B mode
    """
    args_opt.embedding_size = 5120
    args_opt.num_layers = 40
    args_opt.num_heads = 40
    args_opt.word_emb_dp = 1
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 5e-5
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 8
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse_2_6B(args_opt):
    r"""
        Set config for 2.6B mode
    """
    args_opt.embedding_size = 2560
    args_opt.num_layers = 32
    args_opt.num_heads = 32
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 1e-4
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 16
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse(args_opt):
    r"""
        Set config according to the mode
    """
    parse_fn_dict = {"13B": set_parse_13B, "2.6B": set_parse_2_6B}
    if args_opt.mode not in parse_fn_dict.keys():
        raise ValueError("Invalid mode: {}. Optional mode: 200B, 13B, 2.6B and 1.3B".format(args_opt.mode))
    parse_fn_dict[args_opt.mode](args_opt)
