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

import mindspore.communication.management as management
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import set_parse
from src.utils import args_utils
from src.utils.tokenization_jieba import JIEBATokenizer
from src.pd_cft.main.data_process import get_cftpd_data, generate_samples_cftpd
from src.pd_cft.main.load_model import load_model


def few_shot_learning(eval_config, top_k=0, top_p=0.1):
    model, config = load_model(eval_config, True)

    eval_file_name = "eval_result.json"
    eval_out_file = os.path.join(eval_config.output_path, eval_file_name)
    f_out = open(eval_out_file, 'a')
    print("result file:", eval_out_file)

    tokenizer = JIEBATokenizer(os.path.join(eval_config.ckpt_path, eval_config.vocab_model_name))

    res = list()  # shot choose ['zero_shot', 'one_shot', 'few_shot']
    for task in ["zero_shot"]:
        rank = management.get_rank()

        contexts, questions, samples, labels = get_cftpd_data(eval_config, tokenizer, task)  # load dataset
        print(f"All test case num:", len(samples))

        print(task + " task is processing!")
        res_dict = dict()
        res_dict[task] = task + " task is processing!"

        cnt, num_match = 0, 0
        for i, sample in enumerate(samples):
            cnt += 1
            match = "False"

            input_ids = np.array(sample).reshape(1, -1)
            if input_ids.shape[-1] >= config.seq_length - 4:
                input_ids = input_ids[:, -900:]

            outputs = generate_samples_cftpd(model, input_ids, config.seq_length, len(labels[i]), end_token=9,
                                             top_k=top_k, top_p=top_p)
            output_list = outputs.tolist()
            output_list = output_list[input_ids.shape[-1]:]

            answers_pred = "".join(tokenizer.decode(tokenizer.convert_ids_to_tokens(output_list)))
            gt_label = "".join(tokenizer.decode(tokenizer.convert_ids_to_tokens(labels[i])))

            if rank == 0 and i % 100 == 0:
                print(f"{i}: ", answers_pred)

            if answers_pred == gt_label:
                match = "True"
                num_match += 1
            answers_pred = answers_pred + " -Matched:" + match + " -Label: " + gt_label

            if i % 200 == 0:
                print(f"[{i}/{len(labels)}] eval-{task}: cnt={cnt} acc={num_match / cnt}", answers_pred)
                res_dict[
                    f"[{i}/{len(labels)}] eval-{task}: cnt={cnt} acc={num_match / cnt}"] = f"cnt={cnt} acc={num_match / cnt}" + ', ' + answers_pred

        res_dict[f'eval-{task}'] = f'cnt={cnt} acc={num_match / cnt}'
        res.append(res_dict)
        print(f'eval-{task}: cnt={cnt} acc={num_match / cnt}')
    json.dump(res, f_out)
    f_out.close()
    if os.path.isfile(eval_out_file):
        os.chmod(eval_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_learning(args)
