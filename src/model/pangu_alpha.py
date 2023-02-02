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
"""PanguAlpha model"""
import copy
import os

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore.nn.transformer import MoEConfig
from mindspore.nn.transformer.layers import _LayerNorm
from mindspore.nn.transformer.transformer import VocabEmbedding, TransformerEncoder, TransformerEncoderLayer, \
    AttentionMask
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class EmbeddingLayer(nn.Cell):
    r"""Embedding layer of the PanGUAlpha Model"""

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        # Only for the pipeline mode, the embedding needs to be row sliced.
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("normal", [config.vocab_size, config.hidden_size],
                                                                    dtype=mstype.float32),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 embedding_size=config.hidden_size,
                                                 param_init=initializer("normal",
                                                                        [config.seq_length, config.hidden_size],
                                                                        dtype=mstype.float32),
                                                 parallel_config=copied_parallel_config.embedding_dp_mp_config)
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.batch_size = config.batch_size

    def construct(self, input_ids, input_position, init_reset, batch_valid_length):
        input_position = ms.ops.cast(input_position, ms.int32)
        word_embedding, word_table = self.word_embedding(input_ids)
        if self.use_past and not self.is_first_iteration:
            _, seq_length = F.shape(input_ids)
            input_position = batch_valid_length.view(self.batch_size, seq_length)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table

    def get_word_embedding_weight(self):
        return self.word_embedding.embedding_table


class QueryLayer(TransformerEncoderLayer):
    r"""Query Layer at the final layer."""

    def __init__(self, batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 param_init_type=mstype.float32,
                 hidden_act='fast_gelu',
                 use_past=False,
                 parallel_config=None,
                 softmax_compute_type=mstype.float32):
        super(QueryLayer, self).__init__(batch_size=batch_size,
                                         hidden_size=hidden_size,
                                         ffn_hidden_size=ffn_hidden_size,
                                         num_heads=num_heads,
                                         seq_length=seq_length,
                                         attention_dropout_rate=attention_dropout_rate,
                                         hidden_dropout_rate=hidden_dropout_rate,
                                         post_layernorm_residual=post_layernorm_residual,
                                         param_init_type=param_init_type,
                                         hidden_act=hidden_act,
                                         use_past=use_past,
                                         parallel_config=parallel_config.dp_mp_config,
                                         softmax_compute_type=softmax_compute_type)

    def construct(self, x, query_vector, input_mask, init_reset=True, batch_valid_length=None):
        r"""
        The forward process of the block.
        """
        # [bs * seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(query_vector, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present


class PanguAlphaHead(Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        config(): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self,
                 hidden_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super(PanguAlphaHead, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (
                parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embed):
        state = P.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # Used for the pipeline's stages setting
    # As the final layer is not included here, so we need to manually add here.
    # original:  if set two stages, layers on two stages will be [15, 16+1]
    # with 1 added, the layers on two stages will be [16, 15 +1]
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id
    print(f"pipeline stage id is {pp_id}", flush=True)

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class PanguAlpha_Model(Cell):
    r"""The base backbone of the PanGuAlpha model"""

    def __init__(self, config):
        super(PanguAlpha_Model, self).__init__()
        self.is_pipeline = config.parallel_config.pipeline_stage > 1
        self.embedding = EmbeddingLayer(config)
        self.config = config
        self.layernorm = _LayerNorm((config.hidden_size,)).to_float(mstype.float32)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1
        # Configure the shard configure of the Embedding layer
        self.embedding.pipeline_stage = 0
        self.num_layers = config.num_layers
        if config.use_moe:
            moe_config = MoEConfig(expert_num=config.expert_num,
                                   num_experts_chosen=config.per_token_num_experts_chosen)
        else:
            moe_config = MoEConfig(expert_num=1)
        # The shard setting of Transformer is set within the class StackedTransformer
        self.blocks = TransformerEncoder(num_layers=config.num_layers - 1,  # 31
                                         batch_size=config.batch_size,      # 1
                                         hidden_size=config.hidden_size,    # 2560
                                         ffn_hidden_size=config.ffn_hidden_size,  # 768
                                         num_heads=config.num_heads,  # 32
                                         seq_length=config.seq_length,   # 1024
                                         attention_dropout_rate=config.dropout_rate,
                                         hidden_dropout_rate=config.dropout_rate,
                                         lambda_func=set_parallel_configure_for_layer,
                                         hidden_act=config.hidden_act,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         parallel_config=config.parallel_config,
                                         moe_config=moe_config,
                                         softmax_compute_type=config.softmax_compute_type).blocks
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.top_query_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                  embedding_size=config.hidden_size,
                                                  param_init=initializer("normal",
                                                                         [config.seq_length, config.hidden_size],
                                                                         dtype=mstype.float32),
                                                  parallel_config=copied_parallel_config.embedding_dp_mp_config)
        self.top_query_embedding.pipeline_stage = config.parallel_config.pipeline_stage - 1
        if config.parallel_config.pipeline_stage > 1:
            self.top_query_embedding.set_comm_fusion(2)
        else:
            self.top_query_embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.top_query_layer = QueryLayer(batch_size=config.batch_size,
                                          hidden_size=config.hidden_size,
                                          ffn_hidden_size=config.ffn_hidden_size,
                                          num_heads=config.num_heads,
                                          seq_length=config.seq_length,
                                          attention_dropout_rate=config.dropout_rate,
                                          hidden_dropout_rate=config.dropout_rate,
                                          hidden_act=config.hidden_act,
                                          param_init_type=config.param_init_type,
                                          use_past=config.use_past,
                                          softmax_compute_type=config.softmax_compute_type,
                                          parallel_config=config.parallel_config)
        if isinstance(config.parallel_config.recompute, bool):
            if config.parallel_config.recompute:
                self.top_query_layer.recompute()
        else:
            if config.parallel_config.recompute.recompute:
                self.top_query_layer.recompute(recompute_slice_activation=
                                               config.parallel_config.recompute.recompute_slice_activation)

        self.top_query_layer.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.top_query_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.dtype = mstype.float16

        if config.load_ckpt_path:
            self.load_embedding_from_ckpt(config)
        self.run_type = config.run_type

    def construct(self, input_ids,
                  input_position,
                  encoder_masks,
                  init_reset=True,
                  batch_valid_length=None):
        r"""forward pass of the model"""
        embed, word_table = self.embedding(input_ids, input_position, init_reset, batch_valid_length)
        hidden_state = P.Cast()(embed, self.dtype)
        # the input of the incremental prediction is 3d
        if self.run_type != 'predict':
            hidden_state = self.reshape_to_2d(hidden_state)
        if self.blocks is not None:
            for i in range(self.num_layers - 1):
                hidden_state, _ = self.blocks[i](hidden_state, encoder_masks, init_reset, batch_valid_length)
        if self.is_pipeline:
            top_query_hidden_states, _ = self.top_query_embedding(input_position)
            top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
            encoder_output, _ = self.top_query_layer(hidden_state, top_query_hidden_states,
                                                     encoder_masks, init_reset, batch_valid_length)
            encoder_output = self.layernorm(encoder_output)
        else:
            hidden_state = self.reshape_to_2d(hidden_state)
            encoder_output = self.layernorm(hidden_state)
            encoder_output = P.Cast()(encoder_output, self.dtype)
            top_query_hidden_states, _ = self.top_query_embedding(input_position)
            top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
            encoder_output, _ = self.top_query_layer(encoder_output, top_query_hidden_states,
                                                     encoder_masks, init_reset, batch_valid_length)

        return encoder_output, word_table

    def reshape_to_2d(self, x):
        r"""reshape nd tensor to 2d, if n <= 2, keep original shape."""
        shape = F.shape(x)
        if len(shape) <= 2:
            return x
        x = F.reshape(x, (-1, shape[-1]))
        return x

    def load_embedding_from_ckpt(self, config):
        r"""load the weights from the checkpoint"""

        def load_param(path):
            if os.path.exists(path):
                p_table = np.load(path)
                table_param = Tensor(p_table, mstype.float32)
            else:
                raise ValueError(f"{path} file not exits, "
                                 f"please check whether embedding file exit.")
            return table_param

        # three embedding needed to be loaded
        # Loading the embedding table from the ckpt path:
        position_embedding_path = os.path.join(config.load_ckpt_path, config.position_embedding_name)
        word_embedding_path = os.path.join(config.load_ckpt_path, config.word_embedding_name)
        top_query_embedding_path = os.path.join(config.load_ckpt_path, config.top_query_embedding_name)
        self.embedding.word_embedding.embedding_table = Parameter(initializer(load_param(word_embedding_path),
                                                                              [self.config.vocab_size,
                                                                               self.config.hidden_size]),
                                                                  name='word_embedding_table', parallel_optimizer=False)
        self.embedding.position_embedding.embedding_table = Parameter(initializer(load_param(position_embedding_path),
                                                                                  [self.config.seq_length,
                                                                                   self.config.hidden_size]),
                                                                      name='position_embedding_table',
                                                                      parallel_optimizer=False)
        self.top_query_embedding.embedding_table = Parameter(initializer(load_param(top_query_embedding_path),
                                                                         [self.config.seq_length,
                                                                          self.config.hidden_size]),
                                                             name='query_embedding_table', parallel_optimizer=False)


class PanguAlpha(nn.Cell):
    """
    The PanguAlpha network consisting of two parts the backbone and the head
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """

    def __init__(self, config):
        super(PanguAlpha, self).__init__()
        # Network head to get logits over vocabulary
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        self.head = PanguAlphaHead(hidden_size=config.hidden_size,
                                   parallel_config=copied_parallel_config)
        self.head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.backbone = PanguAlpha_Model(config)
        self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

    def construct(self, input_ids, input_position, attention_mask,
                  init_reset=True, batch_valid_length=None):
        output_states, word_table = self.backbone(input_ids, input_position, attention_mask,
                                                  init_reset, batch_valid_length)
        logits = self.head(output_states, word_table)
        return logits


class PanguAlphaWithLoss(Cell):
    """
    PanguAlpha training loss for generation.
    Args:
        config(PanGUConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss):
        super(PanguAlphaWithLoss, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = network
        self.pad_token = config.pad_token
        self.loss = loss

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice_mask = P.StridedSlice().shard(((dp, 1, 1),))
        self.micro_batch_step = 1
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num

    def construct(self, input_ids, input_position=None, attention_mask=None):
        r"""Forward process of the pangu alpha model"""
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        decoder_attention_masks = self.slice_mask(attention_mask, (0, 0, 0), (self.batch_size, self.len, self.len),
                                                  (1, 1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.pad_token),
                            mstype.float32)

        logits = self.network(tokens,
                              input_position,
                              decoder_attention_masks)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1),
                            (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output


class PanguAlphaWithLoss2(nn.Cell):
    """
    GPT training loss
    Args:
        network: backbone network of pangu_alpha
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=9):
        super(PanguAlphaWithLoss2, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        dp = config.parallel_config.data_parallel
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.get_attention_mask = AttentionMask(config.seq_length)
        self.expand = P.ExpandDims()
        self.pad_token = config.pad_token

    def construct(self, input_ids, mask_ids_input, input_position=None, attention_mask=None):
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.pad_token), mstype.float32)
        input_position = F.tuple_to_array(F.make_range(self.seq_length))
        input_position = P.Tile()(self.expand(input_position, 0), (self.batch_size, 1))
        attention_mask = self.get_attention_mask(input_mask)
        logits = self.network(tokens, input_position, attention_mask)
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.seq_length + 1), (1, 1))
        output = self.loss(logits, labels, mask_ids_input)
        return output


class PanguAlphaLossWithPrompt(Cell):
    """
    PanguAlpha training loss for generation.
    Args:
        config(PanGUConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss):
        super(PanguAlphaLossWithPrompt, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = network
        self.pad_token = config.pad_token
        self.loss = loss
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.micro_batch_step = 1
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(config.seq_length)
        self.equal = P.Equal()
        self.expand = P.ExpandDims()

    def construct(self, input_ids, prompt_ids):
        r"""Forward process of the pangu alpha model"""
        tokens = input_ids
        input_mask = F.cast(self.not_equal(tokens, self.pad_token), mstype.float32)
        input_position = F.tuple_to_array(F.make_range(self.seq_length))
        input_position = P.Tile()(self.expand(input_position, 0), (self.batch_size, 1))

        input_mask_a = F.cast(self.equal(prompt_ids, self.pad_token), mstype.float32)
        attention_mask = self.get_attention_mask(input_mask)

        logits = self.network(tokens, input_position, attention_mask)

        log_probs = self.log_softmax(logits)
        input_mask_b = input_mask * input_mask_a
        return log_probs, input_mask_b


class EvalNet(nn.Cell):
    """
    PanguAlpha evaluation net
    Args:
        backbone: backbone network of PanguAlpha
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        current_index: the index of current token
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=6, seq_length=1024):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.GatherV2().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))
        # used for incremental prediction
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, seq_length)), mstype.float32)

    def construct(self, input_ids, current_index, init_reset=True, batch_valid_length=None):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        if self.is_first_iteration is False:
            attention_mask = P.Tile()(self.all_ones_attention_mask, (bs, 1, 1))
        else:
            attention_mask = self.get_attention_mask(input_mask)
        logits = self.backbone(input_ids, input_position, attention_mask,
                               init_reset, batch_valid_length)
        log_probs = self.log_softmax(logits)

        index = current_index.view(1, )
        logits = self.gather(log_probs, index, 0)
        logits = logits.view(bs, 1, -1)
        return logits


class EvalNet_p(nn.Cell):
    """
    PanguAlpha evaluation net

    Args:
        backbone: backbone network of PanguAlpha
        generate: enable generate mode

    Inputs:
        input_ids: the tokenized inpus

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, config, backbone, generate=False):
        super(EvalNet_p, self).__init__(auto_prefix=False)
        self.backbone = backbone
        dp = config.parallel_config.data_parallel
        self.pad_token = config.pad_token
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.get_attention_mask = AttentionMask(config.seq_length)
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        self.expand = P.ExpandDims()
        self.argmax = P.Argmax()
        self.generate = generate

    def construct(self, input_ids):
        """evaluation net"""
        tokens = input_ids
        # input_mask = F.cast(F.not_equal(input_ids, 6), mstype.float32)
        input_mask = F.cast(self.not_equal(tokens, self.pad_token), mstype.float32)
        input_position = F.tuple_to_array(F.make_range(self.seq_length))
        input_position = P.Tile()(self.expand(input_position, 0), (self.batch_size, 1))
        attention_mask = self.get_attention_mask(input_mask)
        logits = self.backbone(input_ids, input_position, attention_mask)
        if self.generate:
            outputs = nn.LogSoftmax()(logits)
            outputs = F.tensor_pow(np.e, outputs)
        else:
            outputs = self.argmax(logits)
        return outputs


class PANGUALPHA_BCHead(nn.Cell):
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
        super(PANGUALPHA_BCHead, self).__init__()
        self.dtype = config.softmax_compute_type
        self.batch_size = config.batch_size
        self.cast = P.Cast()
        self.mapping = nn.Dense(config.hidden_size * config.seq_length, config.label_size).to_float(
            config.softmax_compute_type)

    def construct(self, state):
        state = P.Reshape()(state, (self.batch_size, -1))
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
        self.head = PANGUALPHA_BCHead(config)
        dp = config.parallel_config.data_parallel
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.get_attention_mask = AttentionMask(config.seq_length)
        self.expand = P.ExpandDims()

    def construct(self, input_ids, input_mask=None, input_position=None, attention_mask=None, past=None, eos_token=9):
        if input_mask is None:
            input_mask = F.cast(self.not_equal(input_ids, F.cast(eos_token, mstype.int32)), mstype.float32)
        input_position = F.tuple_to_array(F.make_range(self.len))
        input_position = P.Tile()(self.expand(input_position, 0), (self.batch_size, 1))
        attention_mask = self.get_attention_mask(input_mask)
        output_states, _, _ = self.backbone(input_ids, input_position, attention_mask)
        logits = F.cast(self.head(output_states), mstype.float32)
        return logits


class PanguAlpha_BCLoss(nn.Cell):
    def __init__(self, config, network, loss, eos_token=9):
        super(PanguAlpha_BCLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        dp = config.parallel_config.data_parallel
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.eod_reset = config.eod_reset
        if self.eod_reset:
            self.slice_mask = P.StridedSlice().shard(((dp, 1, 1),))

    def construct(self, input_ids, labels, input_position=None, attention_mask=None):
        input_mask = F.cast(self.not_equal(input_ids, self.eos_token), mstype.float32)
        logits = self.network(input_ids, input_mask, input_position, attention_mask)
        output = self.loss(logits, labels)
        return output
