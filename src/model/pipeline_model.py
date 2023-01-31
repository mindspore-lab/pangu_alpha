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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal, TruncatedNormal
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.model.attention_modules import AttentionMask
from src.model.base_modules import LayerNorm
from src.model.pangu_alpha_model import QueryLayer, Block, PanguAlpha_Head


class EmbeddingLookupPipeline(nn.Cell):
    """
    The embedding lookup table for vocabulary
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(EmbeddingLookupPipeline, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.word_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (config.dp, 1)))
        else:
            self.gather = P.Gather().shard(((config.mp, 1), (1, 1)))
            self.gather.add_prim_attr("repeated_calc_num_direction", "left")
            if config.forward_reduce_scatter:
                self.gather.add_prim_attr("forward_type", "ReduceScatter")
        self.gather.add_prim_attr("begin", 0)
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids, table):
        output = self.gather(table, input_ids, 0)
        return output


class PANGUALPHA_EmbeddingPipeLine(nn.Cell):
    def __init__(self, config):
        super(PANGUALPHA_EmbeddingPipeLine, self).__init__()
        self.word_embedding = EmbeddingLookupPipeline(config)
        self.position_embedding = nn.Embedding(config.se_length, config.embedding_size, embedding_table=Normal(0.02))
        self.position_embedding.gather.shard(((1, 1), (config.dp,)))
        self.position_embedding.expand.shard(((config.dp, 1),))
        self.add = P.Add().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.dropout = nn.transofmer.layers._Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

    def construct(self, input_ids, table, input_position):
        input_embedding = self.word_embedding(input_ids, table)
        position_embedding = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)
        hidden_states = self.dropout(hidden_states)
        hidden_states = P.Cast()(hidden_states, mstype.float16)
        return hidden_states


class PANGUALPHA_Mask(nn.Cell):
    def __init__(self, config):
        super(PANGUALPHA_Mask, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.dtype = config.compute_dtype
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))

    def construct(self, input_mask, attention_mask):
        attention_mask = self.expand_dims(attention_mask, 1)
        return attention_mask


class PANGUALPHA_ModelPipeline(nn.Cell):
    """
    The backbone of PANGUALPHA network
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map
    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(PANGUALPHA_ModelPipeline, self).__init__()
        self.pangu_alpha_embedding = PANGUALPHA_EmbeddingPipeLine(config).set_comm_fusion(1)
        self.pangu_alpha_embedding.stage = 0
        self.pangu_alpha_mask = PANGUALPHA_Mask(config)
        self.blocks = nn.CellList()
        dropout_recompute = False
        self.top_query_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                                embedding_table=TruncatedNormal(0.02))
        self.top_query_embedding.gather.shard(((1, 1), (config.dp,)))
        self.top_query_embedding.expand.shard(((config.dp, 1),))
        for i in range(config.num_layers):
            if i == config.num_layers - 1:
                self.top_query_embedding.set_comm_fusion(2)
                self.top_query_embedding.stage = i * config.stage_num // config.num_layers
                per_block = QueryLayer(config).set_comm_fusion(2)
            else:
                per_block = Block(config, i + 1).set_com_fusion(2)
            per_block.stage = i * config.stage_num // config.num_layers
            per_block.recompute()
            self.blocks.append(per_block)

        if config.self_layernorm:
            self.layernorm = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        else:
            self.layernorm = nn.LayerNorm(
                (config.embedding_size,)).to_float(mstype.float32)
            self.layernorm.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
        self.layernorm.set_comm_fusion(2)
        self.layernorm.stage = config.stage_num - 1
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.dtype = config.compute_dtype
        self.num_layers = config.num_layers


class PANGUALPHAPipeline(nn.Cell):
    """
    The PANGUALPHA network consisting of two parts the backbone and the head
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """

    def __init__(self, config):
        super(PANGUALPHAPipeline, self).__init__()
        self.backbone = PANGUALPHA_ModelPipeline(config)
        self.head = PanguAlpha_Head(config)
        self.head.stage = config.stage_num - 1
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.embedding_table = Parameter(initializer(
            Normal(0.02), [self.vocab_size, self.embedding_size]),
            name="embedding_table")

    def construct(self, input_ids, input_mask, input_position, attention_mask, past=None):
        output_states, _ = self.backbone(input_ids, input_mask, self.embedding_table, input_position, attention_mask,
                                         past)
        logits = self.head(output_states, self.embedding_table)
        return logits


class MicroBatch(nn.Cell):
    def __init__(self, config):
        super(MicroBatch).__init__()
        self.micro_slice = P.StridedSlice().shard(((1, 1),))
        self.micro_attention_slice = P.StridedSlice().shard(((1, 1),))
        self.shape = P.Shape()
        self.stage_num = config.micro_size
        self.seq_len = config.seq_length
        self.slice_mask = P.StridedSlice().shard(((1, 1, 1),))

    def construct(self, x, i, input_position, attention_mask):
        input_shape = self.shape(x)
        micro_batch_begin = (i * input_shape[0] // self.stage_num, 0)
        micro_batch_end = ((i + 1) * input_shape[0] // self.stage_num, input_shape[1])
        micro_batch_stride = (1, 1)
        micro_input = self.micro_slice(x, micro_batch_begin, micro_batch_end, micro_batch_stride)
        micro_input_position_begin = (i * input_shape[0] // self.stage_num, 0)
        micro_input_position_end = ((i + 1) * input_shape[0] // self.stage_num, self.seq_len)
        micro_input_position = self.micro_attention_slice(input_position, micro_input_position_begin,
                                                          micro_input_position_end, micro_batch_stride)
        micro_attention_mask_begin = (i * input_shape[0] // self.stage_num, 0, 0)
        micro_attention_mask_end = ((i + 1) * input_shape[0] // self.stage_num, self.seq_len, self.seq_len)
        micro_attention_mask_stride = (1, 1, 1)
        micro_attention_mask = self.slice_mask(attention_mask, micro_attention_mask_begin, micro_attention_mask_end,
                                               micro_attention_mask_stride)
        return micro_input, micro_input_position, micro_attention_mask


class PANGUALPHAWithLossPipeline(nn.Cell):
    """
    PANGUALPHA training loss
    Args:
        network: backbone network of PANGUALPHA
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=6):
        super(PANGUALPHAWithLossPipeline, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.micro_batch_step = config.micro_size
        self.micro_input = nn.CellList()
        self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))
        for i in range(self.micro_batch_step):
            micro = MicroBatch(config)
            micro.micro_slice.add_prim_attr("micro", i)
            micro.micro_slice.add_prim_attr("start", i)
            self.micro_input.append(micro)

    def construct(self, input_ids, input_position, attention_mask):
        ret = None
        for i in range(self.micro_batch_step):
            micro_input, micro_input_position, micro_attention_mask = self.micro_input[i](input_ids, i, input_position,
                                                                                          attention_mask)
            tokens = self.slice(micro_input, (0, 0), (self.batch_size // self.micro_batch_step, -1), (1, 1))

            input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
            logits = self.network(tokens, input_mask, micro_input_position, micro_attention_mask)
            labels = self.slice(micro_input, (0, 1), (self.batch_size // self.micro_batch_step,
                                                      self.len + 1), (1, 1))
            output = self.loss(logits, labels, input_mask)
            if ret is not None:
                ret = ret + output
            else:
                ret = output
        return ret
