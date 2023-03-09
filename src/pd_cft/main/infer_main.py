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
from src.utils.tokenization_jieba import JIEBATokenizer
from src.utils import args_utils
from src.model import set_parse
from src.pd_cft.main.data_process import get_cftpd_data, generate_samples_cftpd
from src.pd_cft.main.load_model import load_model


def few_shot_infer(infer_args, top_k=0, top_p=0.1):
    model, config = load_model(infer_args, False)

    infer_file_name = "infer_result.json"
    infer_out_file = os.path.join(infer_args.output_path, infer_file_name)

    f_out = open(infer_out_file, 'a', encoding='utf-8')

    print("result file:", infer_out_file, flush=True)

    tokenizer = JIEBATokenizer(os.path.join(infer_args.ckpt_path, infer_args.vocab_model_name))

    res_list = list()  # shot choose ['zero_shot', 'one_shot', 'few_shot']
    for task in ["zero_shot"]:
        res_dict = dict()
        res_dict[task] = []
        print(task + " task is processing!", flush=True)
        rank = management.get_rank()

        contexts, questions, samples, labels = get_cftpd_data(infer_args, tokenizer, task)  # load dataset
        print(f"All test case num:", len(samples))

        for i, sample in enumerate(samples):

            input_ids = np.array(sample).reshape(1, -1)
            if input_ids.shape[-1] >= config.seq_length - 4:
                input_ids = input_ids[:, -900:]

            outputs = generate_samples_cftpd(model, input_ids, config.seq_length, len(labels[i]), end_token=9,
                                             top_k=top_k, top_p=top_p)
            output_list = outputs.tolist()
            output_list = output_list[input_ids.shape[-1]:]

            answers_pred = "".join(tokenizer.decode(tokenizer.convert_ids_to_tokens(output_list)))
            if rank == 0:
                print(f"{i}: ", answers_pred)

            tmp_dict = dict()
            tmp_dict["contexts"] = contexts[i]
            tmp_dict["questions"] = questions[i]
            tmp_dict["answers_pred"] = answers_pred
            res_dict[task].append(tmp_dict)
        res_list.append(res_dict)

    json.dump(res_list, f_out, ensure_ascii=False)
    f_out.close()
    if os.path.isfile(infer_out_file):
        os.chmod(infer_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_infer(args)
