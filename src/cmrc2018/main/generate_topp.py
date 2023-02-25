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

from mindspore import dtype as mstype
import numpy as np
from mindspore.common.tensor import Tensor

def generate(model, origin_inputs, seq_length, end_token=50256, TOPK=5):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(6, 6))
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length - 1, :]
        p_args = probs.argsort()[::-1][:TOPK]

        p = probs[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        if p_args[target_index] == end_token or valid_length == seq_length - 1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1
    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    return outputs


def generate_cftpd(model, origin_inputs, seq_length, label_token_length, end_token=9):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 6))

    generate_tokens_num = 0
    while generate_tokens_num < label_token_length:

        inputs = Tensor(input_ids, mstype.int32)
        probs, p_args = model.predict(inputs)
        probs = probs.asnumpy()[valid_length - 1, :]
        p_args = p_args.asnumpy()[valid_length - 1, :]

        p = probs
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        if p_args[target_index] == end_token or valid_length == seq_length - 1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        generate_tokens_num += 1
        valid_length += 1
        outputs = input_ids

    length = np.sum(outputs != 6)
    outputs = outputs[0][:length]
    return outputs


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)


def top_k_logits(logits, top_k=0, top_p=0.9, filter_value=-float(0)):
    if top_k > 0:
        indices_to_remove = logits < np.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_indices = np.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]

        cumulative_probs = np.cumsum(sorted_logits, axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_samples(model, origin_inputs, seq_length, end_token=50256, top_k=0, top_p=0.9, temperature=0.9):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    PAD_ZERO_ID = 6

    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(PAD_ZERO_ID, PAD_ZERO_ID))
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length - 1, :]
        probs /= temperature

        probs = top_k_logits(probs)
        p = probs / sum(probs)
        target_index = np.random.choice(len(p), p=p)

        if target_index == end_token or valid_length == seq_length - 1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = target_index
        valid_length += 1

    length = np.sum(outputs != PAD_ZERO_ID)
    outputs = outputs[0][:length]
    return outputs


def generate_samples_cftpd(model, origin_inputs, seq_length, end_token=9, top_k=0, top_p=0.9, temperature=1.0):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    PAD_ZERO_ID = 6

    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, PAD_ZERO_ID))
    generate_tokens_num = 0

    while generate_tokens_num < 32:
        inputs = Tensor(input_ids, mstype.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length - 1, :]
        probs = probs / temperature
        probs = top_k_logits(probs, top_k=0, top_p=top_p)
        p = probs / sum(probs)
        target_index = np.random.choice(len(p), p=p)

        if target_index == end_token or valid_length == seq_length - 1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = target_index
        valid_length += 1
        generate_tokens_num += 1
        outputs = input_ids

    length = np.sum(outputs != PAD_ZERO_ID)
    outputs = outputs[0][:length]
    return outputs


def generate_samples_old(model, origin_inputs, seq_length, end_token=50256, top_k=0, top_p=0.9, temperature=1.0):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(6, 6))
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        probs, p_args = model.predict(inputs)
        probs = probs.asnumpy()[valid_length - 1, :]
        p_args = p_args.asnumpy()[valid_length - 1, :]

        probs = logits[0, valid_length - 1, :]
        probs /= temperature

        probs = top_k_logits(probs)
        p = probs / sum(probs)
        target_index = np.random.choice(len(p), p=p)

        if p_args[target_index] == end_token or valid_length == seq_length - 1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1

    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    return outputs
