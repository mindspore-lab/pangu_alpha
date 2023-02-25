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

import re
import numpy as np
import json

np.random.seed(666)
def gen_prompt(prompts_, N_shot, len_ori):
    if N_shot == 0:
        return ""
    else:
        ids = np.random.choice(len(prompts_), N_shot).tolist()
        res = ""
        for id in ids:
            if len(res) + len(prompts_[id]) < 1024 - len_ori - 10:
                res = res + prompts_[id]
        return res


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


class Cmrc2018_Dataset:
    def __init__(self, cmrc2018_train_json, cmrc2018_dev_json):
        with open(cmrc2018_train_json, "r", encoding="utf-8") as f:
            self.train_data = json.load(f)["data"]
        with open(cmrc2018_dev_json, "r", encoding="utf-8") as f:
            self.dev_data = json.load(f)["data"]

    def gen_prompt_from_train(self, seg=False):
        data_list = self.train_data
        index = 0
        prompts_ = []
        if seg:
            for data in data_list:
                context = data["paragraphs"][0]["context"]
                context_splits = cut_sent(context)
                qas = data["paragraphs"][0]["qas"]
                for qa in qas:
                    index += 1
                    q = qa["question"]
                    a = qa["answers"][0]["text"]
                    for sent in context_splits:
                        if a in sent:
                            prompt = f"阅读文章：{sent}\n问：{q}\n答：{a}\n"
                            prompts_.append(prompt)
            prompts_ = [x for x in prompts_ if len(x) < 80]
        else:
            for data in data_list:
                context = data["paragraphs"][0]["context"]
                qas = data["paragraphs"][0]["qas"]
                for qa in qas:
                    index += 1
                    q = qa["question"]
                    a = qa["answers"][0]["text"]
                    prompt = f"阅读文章：{context}\n问：{q}\n答：{a}\n"
                    prompts_.append(prompt)
            prompts_ = [x for x in prompts_ if len(x) < 400]
        return prompts_

    def get_data_allshot(self, N_content=50, N_q=1):
        prompts_seg = self.gen_prompt_from_train(seg=True)
        prompts_full = self.gen_prompt_from_train(seg=False)
        input_prompts = []
        answers = []
        data_list = self.dev_data
        for data in data_list[:N_content]:
            context_ = data["paragraphs"][0]["context"]
            qas = data["paragraphs"][0]["qas"]
            for qa in qas[:N_q]:
                q = qa["question"]
                a = qa["answers"][0]["text"]
                input_str0 = f"阅读文章：{context_}\n问：{q}\n答："
                demo0 = ""
                demo1 = gen_prompt(prompts_full, N_shot=1, len_ori=len(input_str0))
                demo2 = gen_prompt(prompts_seg, N_shot=5, len_ori=len(input_str0))
                input_prompts.append([demo0 + input_str0,
                                      demo1 + input_str0,
                                      demo2 + input_str0])
                answers.append(a)
        return input_prompts, answers

    def get_data(self, N_content=50, N_q=1, N_Shot=0):
        prompts_seg = self.gen_prompt_from_train(self.train_json, seg=True)
        prompts_full = self.gen_prompt_from_train(self.train_json, seg=False)
        input_prompts = []
        Answers = []

        data_list = self.dev_data
        for data in data_list[:N_content]:
            context_ = data["paragraphs"][0]["context"]
            qas = data["paragraphs"][0]["qas"]
            for qa in qas[:N_q]:
                q = qa["question"]
                a = qa["answers"][0]["text"]
                input_str0 = f"阅读文章：{context_}\n问：{q}\n答："
                if N_Shot == 0:
                    demo = ""
                elif N_Shot == 1:
                    demo = gen_prompt(prompts_full, N_shot=1, len_ori=len(input_str0))
                else:
                    demo = gen_prompt(prompts_seg, N_shot=N_Shot, len_ori=len(input_str0))
                input_prompts.append(demo + input_str0)
                Answers.append(a)
        return input_prompts, Answers
