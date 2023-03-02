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
import json

import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Tensor


def get_examples(train_data_path):
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    examples = []
    for query_id, instance in train_data.items():

        query_text = instance["question"]
        evidences = instance["evidences"]
        answers_true = None
        for evidence in evidences.values():
            answers = evidence["answer"]
            if "no_answer" in answers:
                continue
            answers_true = answers[0]
            break
        if not answers_true:
            continue

        example = f"问：{query_text}\n答：{answers_true}\n\n"
        examples.append(example)

    return examples


def get_random_examples(examples, k, max_str=896):
    """sample"""
    sample_examples = []

    np.random.seed(1)
    ids = np.random.choice(len(examples), k).tolist()
    for i in ids:
        sample_examples.append(examples[i])
        if len("".join(sample_examples)) > max_str:
            break
    example = "".join(sample_examples)
    return example


def generate(model, origin_inputs, seq_length, end_token=50256, TOPK=5, max_num=50):
    """
    TopK for text generation

    Inputs:
        model: the model for inference
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    pad_id = 6
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, pad_id))
    cnt = 0
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length - 1, :]
        p_args = probs.argsort()[::-1][:TOPK]

        p = probs[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        if p_args[target_index] == end_token or valid_length == seq_length - 1 or cnt >= max_num:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1

    length = np.sum(outputs != pad_id)
    outputs = outputs[0][:length]
    return outputs
