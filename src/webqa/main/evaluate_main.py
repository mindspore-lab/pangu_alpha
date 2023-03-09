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
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import set_parse
from src.utils import args_utils
from src.utils.tokenization_jieba import JIEBATokenizer
from src.webqa.main import data_process, evaluate_output
from src.webqa.main.load_model import load_model


def do_evaluate_webqa(tokenizer_input, gpt_eval, model_config, eval_args, max_k=8, max_len=1000):
    print("Eval webqa start!")

    examples = data_process.get_examples(os.path.join(eval_args.data_path, eval_args.data_train_name))
    result_list = []
    # shot : ["zero_shot", "one_shot", "few_shot"]
    for shot in ["few_shot"]:
        start_time = time.time()
        if "few_shot" == shot:
            k = max_k
        elif "one_shot" == shot:
            k = 1
        else:
            k = 0

        with open(os.path.join(eval_args.data_path, eval_args.data_dev_name), "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"All test case num: {len(eval_data)}.")

        pred_, answers_, logsStr, cnt = [], [], [], 0
        for query_id, instance in eval_data.items():
            query_text = instance["question"]
            evidences = list(instance["evidences"].values())
            answers_true = evidences[0]["answer"][0]
            example = data_process.get_random_examples(examples, k)
            input_str = f"{example}问：{query_text}\n答："

            tokenized_text = tokenizer_input.tokenize(input_str)
            start_sentence = tokenizer_input.convert_tokens_to_ids(tokenized_text)
            input_ids = np.array(start_sentence).reshape(1, -1)
            if input_ids.shape[-1] >= max_len:
                input_ids = input_ids[:, -max_len:]

            outputs = data_process.generate(gpt_eval, input_ids, model_config.seq_length,
                                            end_token=tokenizer_input.eot_id, TOPK=5)
            output_list = outputs.tolist()
            output_list = output_list[input_ids.shape[-1]:]

            answers_pred = ''.join(tokenizer_input.decode(tokenizer_input.convert_ids_to_tokens(output_list)))
            answers_process = answers_pred.split("\n问")[0]

            pred_.append(answers_process)
            answers_.append(answers_true)

            if cnt % 100 == 0:
                res_str = evaluate_output.evaluate_pairs(pred_, answers_)
                result_list.append(f"N_shot={shot}, {res_str}")
                print(f"N_shot={shot}, cnt={cnt}, {res_str}", flush=True)
            cnt += 1
        res_str = evaluate_output.evaluate_pairs(pred_, answers_)
        logsStr.append(f"N_shot={shot}, {res_str}")
        cost_time = time.time() - start_time
        print(f'predict samples is {cnt}, cost time is {cost_time}', flush=True)

    all_results = {'results_split': result_list, 'results_all': logsStr}
    with open(os.path.join(eval_args.output_path, 'eval_result.json'), 'w', encoding='utf-8') as out:
        json.dump(all_results, out, ensure_ascii=False)


if __name__ == "__main__":
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    model, config = load_model(args, True)

    tokenizer = JIEBATokenizer(os.path.join(args.ckpt_path, args.vocab_model_name))

    do_evaluate_webqa(tokenizer, model, config, args)
