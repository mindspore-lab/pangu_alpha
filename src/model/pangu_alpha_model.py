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

import math
import os

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from src.model.attention_modules import Attention, AttentionMask, QueryLayerAttention
from src.model.base_modules import EmbeddingLookup, LayerNorm, output, _Dropout


class Block(nn.Cell):
    """
    The basic block of PANGUALPHA network
    Args:
        config(PANGUALPHAConfig): the config of network
        layer_idx: current layer index
    Inputs:
        x: the output of previous layer(input_ids for the first layer)
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        layer_past: the previous feature map
    Returns:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    """

    def __init__(self, config, layer_idx):
        super(Block, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        if config.self_layernorm:
            self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
            self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        else:
            self.layernorm1 = nn.LayerNorm((config.embedding_size,)).to_float(mstype.float32)
            self.layernorm1.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
            self.layernorm2 = nn.LayerNorm((config.embedding_size,)).to_float(mstype.float32)
            self.layernorm2.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))

        self.attention = Attention(config, scale, layer_idx)
        self.output = output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.Add().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.add_last = P.Add().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.add_last.recompute(False)
        self.dtype = config.compute_dtype

    def construct(self, x, input_mask, layer_past=None):
        input_x = F.cast(self.layernorm1(x), self.dtype)
        # print('A', input_x.shape[0])
        attention, layer_present = self.attention(input_x, input_mask, layer_past)
        # print('B', attention.shape[0])
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        else:
            x = self.add(x, attention)
        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.add_last(output_x, mlp_logit)
        else:
            output = self.add_last(x, mlp_logit)
        # print('C', output.shape[0])
        return output, layer_present


class QueryLayer(nn.Cell):
    def __init__(self, config):
        super(QueryLayer, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm1.gamma.parallel_optimizer = False
        self.layernorm1.beta.parallel_optimizer = False
        self.layernorm2.gamma.parallel_optimizer = False
        self.layernorm2.beta.parallel_optimizer = False
        self.attention = QueryLayerAttention(config, scale)
        self.output = output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.Add().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.last_add = P.Add().shard(((config.dp, 1, 1), (config.dp, 1, 1))).add_prim_attr("recompute", False)
        self.dtype = config.compute_dtype

    def construct(self, x, query_hidden_state, input_mask, layer_past=None):
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x, query_hidden_state, input_mask, layer_past)
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        else:
            x = self.add(x, attention)
        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.last_add(output_x, mlp_logit)
        else:
            output = self.last_add(x, mlp_logit)
        return output, layer_present


class PanguAlpha_Model(nn.Cell):
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
        super(PanguAlpha_Model, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.word_embedding = EmbeddingLookup(config).set_comm_fusion(1)
        self.eod_reset = config.eod_reset
        if config.position_embedding_name:
            embedding_path = os.path.join(config.model_file_path, config.position_embedding_name)
            if os.path.exists(embedding_path):
                p_table = np.load(embedding_path)
                position_table_param = Tensor(p_table, mstype.float32)
            else:
                raise ValueError(f"{embedding_path} file not exits, please check!")
        else:
            position_table_param = TruncatedNormal(0.02)
        self.position_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                               embedding_table=position_table_param).set_comm_fusion(1)
        self.word_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.gather.shard(((1, 1), (config.dp,)))
        self.position_embedding.expand.shard(((config.dp, 1),))
        self.blocks = nn.CellList()
        fusion_group_num = 4
        fusion_group_size = config.num_layers // fusion_group_num
        fusion_group_size = max(fusion_group_size, 1)
        num_layers = config.num_layers - 1
        self.num_layers = num_layers
        for i in range(num_layers):
            per_block = Block(config, i + 1).set_comm_fusion(int(i / fusion_group_num) + 2)
            per_block.recompute()
            per_block.attention.dropout.dropout_gen_mask.recompute(False)
            per_block.attention.prob_dropout.dropout_gen_mask.recompute(False)
            per_block.output.dropout.dropout_gen_mask.recompute(False)
            per_block.attention.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            per_block.attention.prob_dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            per_block.output.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            self.blocks.append(per_block)

        if config.self_layernorm:
            self.layernorm = LayerNorm(config.embedding_size, config.dp).to_float(mstype.float32).set_comm_fusion(
                int((num_layers - 1) / fusion_group_size) + 2)
        else:
            self.layernorm = nn.LayerNorm((config.embedding_size,)).to_float(mstype.float32).set_comm_fusion(
                int((num_layers - 1) / fusion_group_size) + 2)
            self.layernorm.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
        self.layernorm.gamma.parallel_optimizer = False
        self.layernorm.beta.parallel_optimizer = False
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.add = P.Add().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))
        self.dtype = config.compute_dtype
        self.dropout = _Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))
        if config.top_query_embedding_name:
            embedding_path = os.path.join(config.model_file_path, config.top_query_embedding_name)
            if os.path.exists(embedding_path):
                top_query_table = np.load(embedding_path)
                top_query_table_param = Tensor(top_query_table, mstype.float32)
            else:
                raise ValueError(f"{embedding_path} file not exits, please check!")
        else:
            top_query_table_param = TruncatedNormal(0.02)
        self.top_query_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                                embedding_table=top_query_table_param).set_comm_fusion(
            int((config.num_layers - 1) / fusion_group_num) + 2)
        self.top_query_embedding.embedding_table.parallel_optimizer = False  # 注意！！！
        self.top_query_embedding.gather.shard(((1, 1), (config.dp,)))
        self.top_query_embedding.expand.shard(((config.dp, 1),))
        self.top_query_layer = QueryLayer(config)
        self.top_query_layer.recompute()
        self.top_query_layer.output.dropout.dropout_gen_mask.recompute(False)
        self.top_query_layer.attention.dropout.dropout_gen_mask.recompute(False)
        self.top_query_layer.attention.prob_dropout.dropout_gen_mask.recompute(False)
        self.top_query_layer.output.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
        self.top_query_layer.attention.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
        self.top_query_layer.attention.prob_dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)

        self.top_query_layer.set_comm_fusion(int((config.num_layers - 1) / fusion_group_num) + 2)
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))

    def construct(self, input_ids, input_mask=None, input_position=None, attention_mask=None, layer_past=None,
                  eos_token=9):
        if input_mask is None:
            input_mask = F.cast(self.not_equal(input_ids, eos_token), mstype.float32)
        if not self.use_past:
            layer_past = self.past
        input_embedding, embedding_table = self.word_embedding(input_ids)
        if not self.eod_reset:
            batch_size, seq_length = F.shape(input_ids)
            input_position = F.tuple_to_array(F.make_range(seq_length))
            input_position = P.Tile()(input_position, (batch_size, 1))
            attention_mask = self.get_attention_mask(input_mask)
        position_embedding = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)
        hidden_states = self.dropout(hidden_states)
        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.expand_dims(attention_mask, 1)
        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states, attention_mask, layer_past)
            present_layer = present_layer + (present,)
        output_state = self.layernorm(hidden_states)
        output_state = F.cast(output_state, self.dtype)
        top_query_hidden_states = self.top_query_embedding(input_position)
        output_state, present = self.top_query_layer(output_state, top_query_hidden_states, attention_mask, layer_past)
        present_layer = present_layer + (present,)
        return output_state, present_layer, embedding_table


# 对齐github
class PanguAlpha_Head(nn.Cell):
    """
    Head for PANGUALPHA to get the logits of each token in the vocab
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(PanguAlpha_Head, self).__init__()
        if config.word_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (config.mp, 1)))
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.embedding_size))
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits


class PanguAlpha(nn.Cell):
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

    def __init__(self, config, backbone):
        super(PanguAlpha, self).__init__()
        self.backbone = backbone
        self.head = PanguAlpha_Head(config)

    def construct(self, input_ids, input_mask=None, input_position=None, attention_mask=None, past=None):
        output_states, _, embedding_table = self.backbone(
            input_ids, input_mask, input_position, attention_mask, past)
        logits = self.head(output_states, embedding_table)
        return logits


class PanguAlpha_BCHead(nn.Cell):
    """
    Head for PANGUALPHA to get the logits of binary classification
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(PanguAlpha_BCHead, self).__init__()
        self.embedding_size = config.embedding_size
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.mapping = nn.Dense(config.embedding_size * config.seq_length, config.label_size).to_float(
            config.compute_dtype)

    def construct(self, state):
        batch = state.shape[0]
        state = P.Reshape()(state, (batch, -1))
        logits = self.mapping(state)
        return logits


class PanguAlpha_BC(nn.Cell):
    """
    The PANGUALPHA network consisting of two parts the backbone and the head for binary classification
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, 1)
    """

    def __init__(self, config, backbone):
        super(PanguAlpha_BC, self).__init__()
        self.backbone = backbone
        self.head = PanguAlpha_BCHead(config)
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))

    def construct(self, input_ids, input_mask=None, input_position=None, attention_mask=None, past=None, eos_token=9):
        if input_mask is None:
            input_mask = F.cast(self.not_equal(input_ids, F.cast(eos_token, mstype.int32)), mstype.float32)
        output_states, _, _ = self.backbone(input_ids, input_mask, input_position, attention_mask, past)
        logits = F.cast(self.head(output_states), mstype.float32)
        return logits


class PanguAlpha_BCLoss(nn.Cell):
    def __init__(self, config, network, loss, eos_token=9):
        super(PanguAlpha_BCLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.eod_reset = config.eod_reset
        if self.eod_reset:
            self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))

    def construct(self, input_ids, labels, input_position=None, attention_mask=None):
        input_mask = F.cast(self.not_equal(input_ids, self.eos_token), mstype.float32)
        logits = self.network(input_ids, input_mask, input_position, attention_mask)
        output = self.loss(logits, labels)
        return output


class PanguAlpha_Eval(nn.Cell):
    def __init__(self, config, network, eos_token=9):
        super(PanguAlpha_Eval, self).__init__(auto_prefix=False)
        self.network = network
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.eod_reset = config.eod_reset
        if self.eod_reset:
            self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))

    def construct(self, input_ids, input_position=None, attention_mask=None):
        input_mask = F.cast(self.not_equal(input_ids, self.eos_token), mstype.float32)
        logits = self.network(input_ids, input_mask, input_position, attention_mask)
        return logits


class PanguAlphaWithLoss(nn.Cell):
    """
    GPT training loss
    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=9):
        super(PanguAlphaWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.eod_reset = config.eod_reset
        if self.eod_reset:
            self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))

    def construct(self, input_ids, mask_ids_input, input_position=None, attention_mask=None):
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        if self.eod_reset:
            input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
            attention_mask = self.slice_mask(attention_mask, (0, 0, 0), (self.batch_size, self.len, self.len),
                                             (1, 1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
        logits = self.network(tokens, input_mask, input_position, attention_mask)
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1), (1, 1))
        output = self.loss(logits, labels, mask_ids_input)
        return output


class EvalNet(nn.Cell):
    """
    PANGUALPHA evaluation net
    Args:
        backbone: backbone network of PANGUALPHA
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.ArgMaxWithValue()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.log_softmax = P.Softmax(axis=-1)

    def construct(self, input_ids):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, 6), mstype.float32)
        logits = self.backbone(input_ids, input_mask)
        value, index = self.topk(logits, 5)
        probs = self.log_softmax(value)
        return probs, index


class EvalNet_p(nn.Cell):
    """
    GPT evaluation net

    Args:
        backbone: backbone network of GPT2/3
        generate: enable generate mode

    Inputs:
        input_ids: the tokenized inpus

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False):
        super(EvalNet_p, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.Argmax()
        self.generate = generate

    def construct(self, input_ids):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, 6), mstype.float32)
        logits = self.backbone(input_ids, input_mask)
        if self.generate:
            outputs = nn.LogSoftmax()(logits)
            outputs = F.tensor_pow(np.e, outputs)
        else:
            outputs = self.argmax(logits)
        return outputs
