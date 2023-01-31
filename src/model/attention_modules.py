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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from src.model.base_modules import Mapping


class AttentionMask(nn.Cell):
    """
    Get the attention matrix for self-attention module
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
    """

    def __init__(self, config):
        super(AttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        ones = np.ones(shape=(config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        attention_mask = self.multiply(attention_mask, lower_traiangle)
        return attention_mask


class Attention(nn.Cell):
    """
    Self-Attention module for each layer
    Args:
        config(PANGUALPHAConfig): the config of network
        scale: scale factor for initialization
        layer_idx: current layer index
    Inputs:
        x: output of previous layer
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        layer_past: the previous feature map

    Returns:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    """

    def __init__(self, config, scale=1.0, layer_idx=None):
        super(Attention, self).__init__()
        self.dense1 = nn.Dense(config.embedding_size, config.embedding_size).to_float(config.compute_dtype)
        self.dense1.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense1.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense2 = nn.Dense(config.embedding_size, config.embedding_size).to_float(config.compute_dtype)
        self.dense2.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense2.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense3 = nn.Dense(config.embedding_size, config.embedding_size).to_float(config.compute_dtype)
        self.dense3.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense3.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.use_past = config.use_past
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.transpose = P.Transpose().shard(((config.dp, 1, config.mp, 1),))
        self.projection = Mapping(config, config.embedding_size, config.embedding_size, scale)
        self.dropout = nn.transformer.layers._Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

        self.n_head = config.num_heads
        self.size_per_head = config.embedding_size // self.n_head
        self.scale = scale
        if layer_idx is not None:
            self.coeff = math.sqrt(layer_idx * math.sqrt(self.size_per_head))
            self.coeff = Tensor(self.coeff)
        if self.scale:
            self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        self.batch_matmul = P.BatchMatMul().shard(((config.dp, config.mp, 1, 1), (config.dp, config.mp, 1, 1)))
        self.real_div = P.RealDiv().shard(((config.dp, config.mp, 1, 1), ()))
        self.prob_dropout = nn.transformer.layers._Dropout(1 - config.dropout_rate)
        self.prob_dropout.dropout_gen_mask.shard(((config.dp, config.mp, 1, 1),))
        self.prob_dropout.dropout_do_mask.shard(((config.dp, config.mp, 1, 1),))
        self.softmax = nn.Softmax()
        self.softmax.softmax.shard(((config.dp, config.mp, 1),))
        self.multiply_data = Tensor([-10000.0], dtype=mstype.float32)
        self.sub = P.Sub().shard(((1,), (config.dp, 1, 1, 1))).add_prim_attr("_side_effect", True)
        self.mul = P.Mul().shard(((config.dp, 1, 1, 1), (1,))).add_prim_attr("_side_effect", True)
        self.add = P.Add().shard(((config.dp, 1, 1, 1), (config.dp, config.mp, 1, 1)))

        self.merger_head_transpose = P.Transpose().shard(((config.dp, config.mp, 1, 1),))
        self.reshape = P.Reshape()

    def construct(self, x, attention_mask, layer_past=None):
        # print(0, x.shape[0])
        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        # print(1, x.shape[1])
        query = self.dense1(x)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(F.reshape(query, (-1, original_shape[1], self.n_head, self.size_per_head)),
                               (0, 2, 1, 3))  # n x n_head x s x perhead
        key = self.transpose(F.reshape(key, (-1, original_shape[1], self.n_head, self.size_per_head)),
                             (0, 2, 3, 1))  # n x n_head x perhead x s
        value = self.transpose(F.reshape(value, (-1, original_shape[1], self.n_head, self.size_per_head)),
                               (0, 2, 1, 3))  # n x n_head x s x perhead
        # print(2, query.shape[0], key.shape[0], value.shape[0])
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        layer_present = P.Pack()([self.transpose(key, (0, 1, 3, 2)), value])
        # print(3, query.shape[0], key.shape[0], value.shape[0], attention_mask.shape[0])
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
        # print(4, attention_merge.shape[0])
        output = self.projection(attention_merge)
        output = self.dropout(output)
        # print(5, output.shape[0])
        return output, layer_present

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        Returns:
            weighted_values: Tensor, the weighted sum scores
        """
        if not self.scale:
            query = query / F.cast(self.coeff, F.dtype(query))
            key = key / F.cast(self.coeff, F.dtype(key))

        score = self.batch_matmul(query, key)
        if self.scale:
            score = self.real_div(
                score,
                P.Cast()(self.scale_factor, P.DType()(score)))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, mstype.float32)
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        # attention_scores = P.Cast()(attention_scores, ori_dtype)
        shape = F.shape(attention_scores)
        attention_probs = self.softmax(
            F.reshape(attention_scores,
                      (shape[0], -1, shape[-1])))  # yzz modify
        attention_probs = P.Cast()(attention_probs, ori_dtype)
        attention_probs = F.reshape(attention_probs, shape)

        attention_probs = self.prob_dropout(attention_probs)
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values

    def merge_heads(self, x):
        """
        Convert a 4d input to a 3d output
        Inputs:
            x: input tensor
        Returns:
            x_merge: the 3d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = x_shape[:-2] + (x_shape[-2] * x_shape[-1],)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def split_heads(self, x, transpose):
        """
        Split 3d tensor to 4d and switch certain axes
        Inputs:
            x: input tensor
            transpose: tuple, the transpose sequence
        Returns:
            x_transpose: the 4d output
        """
        x_size = P.Shape()(x)
        new_x_shape = x_size[:-1] + (self.n_head, self.size_per_head)
        x = self.reshape(x, new_x_shape)
        x_transpose = self.transpose(x, transpose)
        return x_transpose


class QueryLayerAttention(Attention):
    def construct(self, x, query_hidden_state, attention_mask, layer_past=None):
        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        query_hidden_state = F.reshape(query_hidden_state, (-1, original_shape[-1]))
        query = self.dense1(query_hidden_state)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(F.reshape(query, (-1, original_shape[1], self.n_head, self.size_per_head)), (0, 2, 1, 3))
        key = self.transpose(F.reshape(key, (-1, original_shape[1], self.n_head, self.size_per_head)), (0, 2, 3, 1))
        value = self.transpose(F.reshape(value, (-1, original_shape[1], self.n_head, self.size_per_head)), (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        layer_present = P.Pack()([self.transpose(key, (0, 1, 3, 2)), value])
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present
