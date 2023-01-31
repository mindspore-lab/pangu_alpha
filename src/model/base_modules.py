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
import time

import mindspore.common.dtype as mstype
import numpy as np
from mindspore import nn, context
from mindspore._checkparam import Validator
from mindspore.common.initializer import initializer, Normal
from mindspore.common.parameter import Parameter
from mindspore.common.seed import _get_graph_seed
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import _VirtualDataset
from mindspore.train.callback._callback import Callback


class TestMonitor(Callback):
    def __init__(self, cur_epoch_num, epoch_num, start_time):
        self.cur_epoch_num = cur_epoch_num
        self.epoch_num = epoch_num
        self.sum_loss = 0
        self.count = 0
        self.step = 0
        self.start_time = start_time

    def begin(self, run_context):
        self.sum_loss = 0
        self.count = 0
        self.step = 0

    def epoch_begin(self, run_context):
        self.sum_loss = 0
        self.count = 0
        self.step = 0

    def step_begin(self, run_context):
        self.step += 1

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        output = cb_params.net_outputs
        loss = output[0]
        self.count += loss.shape[0]
        self.sum_loss += np.sum(loss.asnumpy())

    def epoch_end(self, run_context):
        print("Eval Step: {}/{}, Eval Loss: {}, Eval Time: {}s".format(self.cur_epoch_num, self.epoch_num,
                                                                       self.sum_loss / self.count,
                                                                       round(time.time() - self.start_time, 2)),
              end=", ", flush=True)
        self.sum_loss = 0
        self.count = 0


class TrainingMonitor(Callback):
    def __init__(self, eval_model, eval_dataset, batch_size, device_num, sink_size, per_epoch_print):
        super(TrainingMonitor, self).__init__()
        self.eval_model = eval_model
        self.eval_dataset = eval_dataset
        self.per_epoch_print = per_epoch_print
        self.batch_size = batch_size
        self.sum_loss = 0
        self.epoch = 0
        self.count = 0
        self.start_time = time.time()
        self.device_num = device_num
        self.sink_size = sink_size

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        self.count += loss.shape[0]
        self.epoch += 1
        self.sum_loss += np.sum(loss.asnumpy())
        print("Training Step: {}/{}, Training Samples: {}, Training Time: {}s, Training Ave Loss: {}".format(
            self.epoch * self.sink_size, cb_params.epoch_num * self.sink_size,
            self.count * self.device_num * self.sink_size, round(time.time() - self.start_time, 2),
            self.sum_loss / self.count), flush=True)
        if self.epoch % self.per_epoch_print == 0:
            acc = self.eval_model.eval(self.eval_dataset, callbacks=[
                TestMonitor(self.epoch * self.sink_size, cb_params.epoch_num * self.sink_size, time.time())])
            print("Eval Accuracy: {}".format(acc['accuracy']), flush=True)
            print("---------------------------------------------------------------------------------------", flush=True)

    def end(self, run_context):
        print("Over!", flush=True)


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PANGUALPHA network.
    """

    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class VirtualDatasetOneInputCell(nn.Cell):
    def __init__(self, backbone):
        super(VirtualDatasetOneInputCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, *data):
        data_ = self._virtual_dataset(*data)
        return self._backbone(*data_)


class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss
    Args:
        config(PANGUALPHAConfig): the config of the network
    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """

    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum().shard(((config.dp, config.mp),))
        self.onehot = P.OneHot().shard(((config.dp, config.mp), (), ()))
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.vocab_size = config.vocab_size
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(
            ((config.dp, config.mp),))
        self.eps_const = Tensor(1e-24, mstype.float32)
        self.sub = P.Sub().shard(((config.dp, config.mp), (config.dp, 1)))
        self.exp = P.Exp().shard(((config.dp, config.mp),))
        self.div = P.RealDiv().shard(((config.dp, config.mp), (config.dp, 1)))
        self.log = P.Log().shard(((config.dp, config.mp),))
        self.add = P.Add().shard(((config.dp, config.mp), ()))
        self.mul = P.Mul().shard(
            ((config.dp, config.mp), (config.dp, config.mp)))
        self.neg = P.Neg().shard(((config.dp, config.mp),))
        self.sum2 = P.ReduceSum().shard(((1,),))

        self.mul2 = P.Mul().shard(((1,), (1,)))
        self.add2 = P.Add()
        self.div2 = P.RealDiv()
        if config.stage_num > 1:
            self.div2.add_prim_attr("end", 0)

    def construct(self, logits, label, input_mask):
        logits = F.cast(logits, mstype.float32)
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))
        label = P.Reshape()(label, (-1,))
        one_hot_label = self.onehot(label, self.vocab_size, self.on_value,
                                    self.off_value)
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))
        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)
        return loss


class _Dropout(nn.Cell):
    r"""
        A Dropout Implements with P.DropoutGenMask and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(_Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "dropout probability should be a number in range (0, 1], but got {}".format(
                    keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        self.is_ascend = context.get_context('device_target') in ["Ascend"]
        if self.is_ascend:
            seed0, seed1 = _get_graph_seed(0, "dropout")
            self.seed0 = seed0
            self.seed1 = seed1
            self.dtype = dtype
            self.get_shape = P.Shape()
            self.dropout_gen_mask = P.DropoutGenMask(Seed0=self.seed0, Seed1=self.seed1)
            self.dropout_do_mask = P.DropoutDoMask()
            self.cast = P.Cast()
        else:
            self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        if not self.is_ascend:
            out, _ = self.dropout(x)
            return out

        if self.keep_prob == 1:
            return x

        shape = self.get_shape(x)
        dtype = P.DType()(x)
        keep_prob = self.cast(self.keep_prob, dtype)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        return 'keep_prob={}'.format(self.keep_prob)

    def shard(self, strategy):
        if self.is_ascend:
            self.dropout_gen_mask.shard(strategy)
            self.dropout_do_mask.shard(strategy)
        else:
            self.dropout.shard(strategy)


class LayerNorm(nn.Cell):
    """
    A layer normlization function
    Args:
        normalized_shape: the size of normalized shape
        dp: the number of splits for tensor parallism
        eps: error bound
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after laywe normalization
    """

    def __init__(self, normalized_shape, dp=4, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma", parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta", parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.eps = eps

        self.square = P.Square().shard(((dp, 1, 1),))
        self.sqrt = P.Sqrt().shard(((dp, 1, 1),))
        self.sub1 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.sub2 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add1 = P.Add().shard(((dp, 1, 1), ()))
        self.add2 = P.Add().shard(((dp, 1, 1), (1,)))
        self.mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.real_div = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))

    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add1(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class Mapping(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """

    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale), [input_size, output_size]),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [output_size, ]), name="mapping_bias", parallel_optimizer=False)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.add = P.Add().shard(((config.dp, 1), (1,)))
        self.matmul = P.MatMul().shard(((config.dp, config.mp), (config.mp, 1)))
        self.matmul.add_prim_attr("recompute_comm_op", False)

    def construct(self, x):
        output_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, output_shape)
        return output


class Mapping_output(nn.Cell):
    """
    A mapping function with a 3d input (different style in the parallism of matmul)
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """

    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping_output, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale), [input_size, output_size]),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [output_size, ]), name="mapping_bias")
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.add = P.Add().shard(((config.dp, config.mp), (config.mp,)))
        self.matmul = P.MatMul().shard(((config.dp, 1), (1, config.mp)))

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        return output


class output(nn.Cell):
    """
    The output mapping module for each layer
    Args:
        config(PANGUALPHAConfig): the config of network
        scale: scale factor for initialization
    Inputs:
        x: output of the self-attention module
    Returns:
        output: Tensor, the output of this layer after mapping
    """

    def __init__(self, config, scale=1.0):
        super(output, self).__init__()
        input_size = config.embedding_size
        output_size = config.embedding_size * config.expand_ratio
        self.mapping = Mapping_output(config, input_size, output_size)
        self.projection = Mapping(config, output_size, input_size, scale)
        self.activation = nn.GELU()
        self.activation.gelu.shard(((config.dp, 1, config.mp),))
        self.dropout = _Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

    def construct(self, x):
        hidden = self.activation(self.mapping(x))
        output = self.projection(hidden)
        output = self.dropout(output)
        return output


class EmbeddingLookup(nn.Cell):
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
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.word_embedding_name:
            embedding_path = os.path.join(config.model_file_path, config.word_embedding_name)
            if os.path.exists(embedding_path):
                e_table = np.load(embedding_path)
                e_table = Tensor(e_table, mstype.float32)
                self.embedding_table = Parameter(e_table, name="embedding_table")
            else:
                raise ValueError(f"{embedding_path} file not exists, please check!")
        else:
            self.embedding_table = Parameter(initializer(Normal(0.02), [self.vocab_size, self.embedding_size]),
                                             name="embedding_table")

        if config.word_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (config.dp, 1)))
        else:
            self.gather = P.Gather().shard(((config.mp, 1), (1, 1)))
            self.gather.add_prim_attr("repeated_calc_num_direction", "left")
            if config.forward_reduce_scatter:
                self.gather.add_prim_attr("forward_type", "ReduceScatter")
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table
