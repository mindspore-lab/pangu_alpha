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

import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import set_parse
from src.utils import args_utils
from src.utils.tokenization_jieba import JIEBATokenizer
from src.csl.main.load_model import load_model
from src.csl.main.csl_prompts import do_infer_csl

rank_id = int(os.getenv('RANK_ID', default=0))


def shot_eval_main(eval_args):
    eval_file_name = "infer_result.json"
    eval_out_file = os.path.join(eval_args.output_path, eval_file_name)
    print("result file:", eval_out_file)
    model, config = load_model(eval_args, True)

    tokenizer = JIEBATokenizer(os.path.join(eval_args.ckpt_path, eval_args.vocab_model_name))

    do_infer_csl(tokenizer, model, config, rank_id, eval_args.data_path, eval_args.output_path)


if __name__ == "__main__":
    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)
    shot_eval_main(args)
