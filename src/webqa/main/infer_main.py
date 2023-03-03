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
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import set_parse
from src.utils import args_utils
from src.utils.tokenization_jieba import JIEBATokenizer
from src.webqa.main import data_process
from src.webqa.main.load_model import load_model

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))


def do_infer_webqa(tokenizer_jieba, model_pangu, model_config, infer_args, max_k=8, max_len=1000):
    print("Eval webqa start!", flush=True)
    examples = data_process.get_examples(os.path.join(infer_args.data_path, infer_args.data_train_name))
    results = {}
    # for shot in ["zero_shot", "one_shot", "few_shot"]:
    for shot in ["few_shot"]:
        if "few_shot" == shot:
            k = max_k
        elif "one_shot" == shot:
            k = 1
        else:
            k = 0
        print("examples is : ", examples)
        with open(os.path.join(infer_args.data_path, infer_args.data_test_name), "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"All test case num: {len(eval_data)}.")

        cnt = 0
        res_list = []
        for query_id, instance in eval_data.items():
            res = ""
            query_text = instance["question"]

            example = data_process.get_random_examples(examples, k)
            input_str = f"{example}问：{query_text}\n答："

            tokenized_text = tokenizer_jieba.tokenize(input_str)
            start_sentence = tokenizer_jieba.convert_tokens_to_ids(tokenized_text)
            input_ids = np.array(start_sentence).reshape(1, -1)
            if input_ids.shape[-1] >= max_len:
                input_ids = input_ids[:, -max_len:]

            outputs = data_process.generate(model_pangu, input_ids, model_config.seq_length,
                                            end_token=tokenizer_jieba.eot_id, TOPK=5)
            output_list = outputs.tolist()
            output_list = output_list[input_ids.shape[-1]:]

            answers_pred = ''.join(tokenizer_jieba.decode(tokenizer_jieba.convert_ids_to_tokens(output_list)))
            answers_pred = answers_pred.split("\n问")[0]
            res = "query_id: " + query_id + ", query_text: " + query_text + ", answers_pred: " + answers_pred
            res_list.append(res)

            cnt += 1
            if cnt % 2 == 0:
                print(res)
            if cnt >= infer_args.test_sample_num:
                break
        results[shot] = res_list


    with open(os.path.join(infer_args.output_path, 'infer_result.json'), 'w', encoding='utf-8') as out:
        json.dump(results, out, ensure_ascii=False)


if __name__ == "__main__":
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    model, config = load_model(args, False)

    tokenizer = JIEBATokenizer(os.path.join(args.ckpt_path, args.vocab_model_name))

    do_infer_webqa(tokenizer, model, config, args)
