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
import os
import time

import mindspore.common.dtype as mstype
import numpy as np
from mindspore.common.tensor import Tensor


def generate_examples(train_data_path):
    with open(train_data_path, "r", encoding="utf-8") as f:
        data_file = f.readlines()

    examples = []
    label_map = {"0": "不是", "1": "是"}
    for i, instance in enumerate(data_file):
        instance = json.loads(instance)
        text = instance["abst"]
        keyword = ",".join(instance["keyword"])
        label = label_map[instance["label"]]
        example = "摘要：" + text + ",关键词：" + keyword + label + "真实关键词\n\n"
        examples.append(example)

    return examples


def generate_random_examples(examples, k, max_str=1000):
    """sample"""
    sample_examples = []
    np.random.seed(1)
    ids = np.random.choice(len(examples), k).tolist()
    for id in ids:
        sample_examples.append(examples[id])
        if len("".join(sample_examples)) > max_str:
            break
    example = "".join(sample_examples)
    return example


def do_eval_csl(tokenizer, model, config, rank_id, data_path_obs, save_path_obs, split="dev", max_len=1000):
    print("eval csl start!")
    train_path = os.path.join(data_path_obs, "train.json")
    examples = generate_examples(train_path)

    # 测试 one-shot精度
    for shot in ["one_shot"]:

        start_time = time.time()
        k = 1
        # 根据实际数据集进行拼接
        data_path = os.path.join(data_path_obs, split + ".json")

        with open(data_path, "r", encoding="utf-8") as f:
            data_file = f.readlines()

        print(f"All test case num: {len(data_file)}.", flush=True)
        label_map = {"0": "不是", "1": "是"}
        id_label = {0: "不是", 1: "是"}
        results, cnt, correct_num, acc = {}, 0, 0, 1
        # max_len = 1000
        for id, instance in enumerate(data_file):
            instance = json.loads(instance)
            cnt += 1
            sentence = instance["abst"]
            keyword = ",".join(instance["keyword"])
            if instance["label"] not in label_map:
                continue
            label = label_map[instance["label"]]
            # example = [f"根据摘要：{text}判断关键词：{keyword}{label}全部为真实关键词\n"]
            query_text = f"摘要：{sentence},关键词：{keyword}"
            answers_true = label
            example = generate_random_examples(examples, k)
            input_str_one = f"{example}{query_text}不是真实关键词"
            input_str_two = f"{example}{query_text}是真实关键词"
            input_ids_one = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str_one))
            input_ids_two = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str_two))

            input_ids_one = np.array(input_ids_one).reshape(1, -1)
            input_ids_two = np.array(input_ids_two).reshape(1, -1)
            if input_ids_one.shape[-1] >= max_len:
                input_ids_one = input_ids_one[:, -max_len:]
            if input_ids_two.shape[-1] >= max_len:
                input_ids_two = input_ids_two[:, -max_len:]

            pad_length_one = config.seq_length + 1 - input_ids_one.shape[-1]

            input_ids_one = np.pad(input_ids_one, ((0, 0), (0, pad_length_one)), 'constant',
                                   constant_values=(0, tokenizer.eot_id))
            pad_length_two = config.seq_length + 1 - input_ids_two.shape[-1]
            input_ids_two = np.pad(input_ids_two, ((0, 0), (0, pad_length_two)), 'constant',
                                   constant_values=(0, tokenizer.eot_id))
            mask_ids = Tensor(np.ones(shape=(1, config.seq_length)), mstype.float32)
            loss1 = model.predict(Tensor(input_ids_one, mstype.int32), mask_ids).asnumpy()
            loss2 = model.predict(Tensor(input_ids_two, mstype.int32), mask_ids).asnumpy()
            loss = np.concatenate([loss1, loss2])
            answers_pred = id_label[np.argmin(loss)]
            if answers_pred == answers_true:
                correct_num += 1
            acc = correct_num / cnt
            if cnt % 100 == 0:
                print(f'eval-{shot}, cnt={cnt} acc={acc}', flush=True)
            if cnt <= 10:
                print(f"{query_text}\n预测判断：{answers_pred}， 判断：{answers_true}\n", flush=True)
        if rank_id == 0:
            results["acc"] = acc
            with open(save_path_obs, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            end_time = time.time()
            print(
                f"Test csl end, shot：{shot}，test num: {cnt}, corect num: {correct_num}, "
                f"cost time: {end_time - start_time}, acc: {acc}!",
                flush=True)


def do_infer_csl(tokenizer, model, config, rank_id, data_path_obs, save_path_obs, split="test", max_len=1000):
    print("infer csl start!")
    train_path = os.path.join(data_path_obs, "train.json")
    examples = generate_examples(train_path)

    # 测试 one-shot精度
    results = {}
    for shot in ["one_shot"]:
        results[shot] = []

        start_time = time.time()
        k = 1
        # 根据实际数据集进行拼接
        data_path = os.path.join(data_path_obs, split + ".json")
        print("data_path : ", data_path)
        # pred_path = os.path.join(save_path_obs, prefix + "_ans_" + shot + ".json")

        with open(data_path, "r", encoding="utf-8") as f:
            data_file = f.readlines()

        print(f"All test case num: {len(data_file)}.", flush=True)
        label_map = {"0": "不是", "1": "是"}
        id_label = {0: "不是", 1: "是"}
        # max_len = 1000
        for id, instance in enumerate(data_file):
            instance = json.loads(instance)
            sentence = instance["abst"]
            keyword = ",".join(instance["keyword"])
            query_text = f"摘要：{sentence},关键词：{keyword}"

            example = generate_random_examples(examples, k)
            input_str_one = f"{example}{query_text}不是真实关键词"
            input_str_two = f"{example}{query_text}是真实关键词"
            input_ids_one = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str_one))
            input_ids_two = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str_two))

            input_ids_one = np.array(input_ids_one).reshape(1, -1)
            input_ids_two = np.array(input_ids_two).reshape(1, -1)
            if input_ids_one.shape[-1] >= max_len:
                input_ids_one = input_ids_one[:, -max_len:]
            if input_ids_two.shape[-1] >= max_len:
                input_ids_two = input_ids_two[:, -max_len:]

            pad_length_one = config.seq_length + 1 - input_ids_one.shape[-1]

            input_ids_one = np.pad(input_ids_one, ((0, 0), (0, pad_length_one)), 'constant',
                                   constant_values=(0, tokenizer.eot_id))
            pad_length_two = config.seq_length + 1 - input_ids_two.shape[-1]
            input_ids_two = np.pad(input_ids_two, ((0, 0), (0, pad_length_two)), 'constant',
                                   constant_values=(0, tokenizer.eot_id))
            mask_ids = Tensor(np.ones(shape=(1, config.seq_length)), mstype.float32)
            loss1 = model.predict(Tensor(input_ids_one, mstype.int32), mask_ids).asnumpy()
            loss2 = model.predict(Tensor(input_ids_two, mstype.int32), mask_ids).asnumpy()
            loss = np.concatenate([loss1, loss2])
            answers_pred = id_label[np.argmin(loss)]
            out = f"摘要：{sentence}, 关键词：{keyword}, {answers_pred}真实关键词"
            if id % 100 == 0:
                print(f"id : {id}, out : {out}", flush=True)
            results[shot].append(out)
        end_time = time.time()
        print(f"infer csl end, cost time: {end_time - start_time}", flush=True)

    if rank_id == 0:
        with open(save_path_obs, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
