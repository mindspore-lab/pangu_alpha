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

import math
import os
import random

import mindspore as ms
import mindspore.common.dtype as mstype
import numpy as np


def top_k_logits(logits, top_k=0, top_p=0.9, filter_value=-float(0)):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < np.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_indices = np.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]

        # cumulative_probs = np.cumsum(softmax(sorted_logits), axis=-1)
        cumulative_probs = np.cumsum(sorted_logits, axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_samples_cftpd(model, origin_inputs, seq_length, label_token_length, end_token=9, top_k=0, top_p=0.9,
                           temperature=1.0):
    PAD_ZERO_ID = 6

    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, PAD_ZERO_ID))
    generate_tokens_num = 0

    while generate_tokens_num < label_token_length:

        inputs = ms.Tensor(input_ids, mstype.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length - 1, :]
        probs /= temperature

        probs = top_k_logits(probs, top_k=top_k, top_p=top_p)

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


def process_example(example, k, keeplinesRatio=1):
    example_sent = ""
    example_lens = len(example)
    # choose k examples randomly
    random_indexs = [random.randint(0, example_lens - 1) for i in range(k)]

    for num_example in random_indexs:
        tmp_lines_list = example[num_example]["context"].split('。')
        all_lines = len(tmp_lines_list)
        keep_lins_num = math.floor(all_lines * keeplinesRatio)

        for i in range(keep_lins_num):
            example_sent = example_sent + tmp_lines_list[-keep_lins_num + i]

        example_sent = example_sent + example[num_example]["question"] + example[num_example]["answer"] + "\n"

    return example_sent


def process_one_sent_eval(tokenizer, sent, example):
    # it will became list after tokenizer encode
    input_sent = example + sent["context"] + sent["question"]
    sent_ids = tokenizer.convert_tokens_to_ids(tokenizer.encode(input_sent))
    truth_label = tokenizer.convert_tokens_to_ids(tokenizer.encode(sent["answer"]))

    L = {"context": sent["context"], "question": sent["question"], "prompt": sent_ids, "truth": truth_label}

    return L


def get_cftpd_data(config, tokenizer, task, k_num=3):
    if task == "zero_shot":
        k = 0
    elif task == "one_shot":
        k = 1
    elif task == "few_shot":
        k = k_num

    all_data = {"contexts": [], "questions": [], "contents": [], "labels": []}
    # load data
    with open(os.path.join(config.data_path, config.data_test_name), "r", encoding="utf-8") as f:
        data = []
        lines = f.readlines()
        sent_dict = {"context": "", "question": "", "answer": ""}

        for line in lines:
            line = line.replace(" ", "")
            line = line.strip()
            if line.count("|||") == 1:
                if not line.find("XXXXX") == -1:
                    continue  # remove the question line in context
                else:
                    sent_dict["context"] = sent_dict["context"] + line[line.find("|||") + 3:]

            else:
                stop_pos_question = line.find("XXXXX")
                first_pos = line.find("|||") + 3
                second_pos = line[first_pos:].find("|||") + first_pos + 3
                sent_dict["question"] = sent_dict["question"] + line[first_pos:stop_pos_question]
                sent_dict["answer"] = sent_dict["answer"] + line[second_pos:]
                data.append(sent_dict)

                sent_dict = {"context": "", "question": "", "answer": ""}

        # load data
        with open(os.path.join(config.data_path, config.data_dev_name), "r", encoding="utf-8") as f:
            examples_data = []
            lines = f.readlines()
            sent_dict_examples = {"context": "", "question": "", "answer": ""}

            for line in lines:
                line = line.replace(" ", "")
                line = line.strip()
                if line.count("|||") == 1:
                    if not line.find("XXXXX") == -1:
                        continue  # remove the question line in context
                    else:
                        sent_dict_examples["context"] = sent_dict_examples["context"] + line[line.find("|||") + 3:]

                else:
                    stop_pos_question = line.find("XXXXX")
                    first_pos = line.find("|||") + 3
                    second_pos = line[first_pos:].find("|||") + first_pos + 3
                    sent_dict_examples["question"] = sent_dict_examples["question"] + line[first_pos:stop_pos_question]
                    sent_dict_examples["answer"] = sent_dict_examples["answer"] + line[second_pos:]
                    examples_data.append(sent_dict_examples)

                    sent_dict_examples = {"context": "", "question": "", "answer": ""}

        examples_i = process_example(examples_data, k)

        for line in data:
            processed_sent = process_one_sent_eval(tokenizer, line, examples_i)
            all_data["contexts"].extend([processed_sent["context"]])
            all_data["questions"].extend([processed_sent["question"]])
            all_data["contents"].extend([processed_sent["prompt"]])
            all_data["labels"].append(processed_sent["truth"])

        return all_data["contexts"], all_data["questions"], all_data["contents"], all_data["labels"]
