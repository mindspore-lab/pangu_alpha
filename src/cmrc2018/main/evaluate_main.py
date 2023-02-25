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

import json
import os
import sys

import mindspore as ms
import mindspore.communication.management as management
import numpy as np
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn import TransformerRecomputeConfig, TransformerOpParallelConfig
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.model import Model
from mindspore.train.serialization import load_distributed_checkpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.model import PanguAlphaConfig, PanguAlpha, EvalNet_p, set_parse
from src.utils.model_utils import get_ckpt_file_list, eval_load_file
from src.utils.tokenization_jieba import JIEBATokenizer
from src.utils import args_utils
from src.cmrc2018.main.data_process import Cmrc2018_Dataset
from src.cmrc2018.main.generate_topp import generate_samples_cftpd
from src.cmrc2018.main.evaluate_output import evaluate_pairs

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def few_shot_learning(eval_config):
    if rank_id % 8 == 0:
        eval_load_file(eval_config)

    res_dir = os.path.join(eval_config.output_path, 'eval_result_rank_' + str(device_id))
    eval_file_name = "eval_result.json"

    eval_out_file = os.path.join(eval_config.output_path, eval_file_name)

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    print("result file:", eval_out_file, flush=True)

    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, max_device_memory="30GB")
    if eval_config.distribute == "True":
        management.init()
        device_num = int(management.get_group_size())
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num), flush=True)
        ms.context.reset_auto_parallel_context()
        ms.context.set_auto_parallel_context(
            parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            strategy_ckpt_load_file=os.path.join(eval_config.ckpt_path, eval_config.ckpt_strategy_name))
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        device_num = 1

    model_parallel_num = device_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = eval_config.per_batch_size * data_parallel_num

    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(
                                                      eval_config.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  expert_parallel=eval_config.expert_parallel_num,
                                                  pipeline_stage=eval_config.stage_num,
                                                  micro_batch_num=eval_config.micro_size,
                                                  optimizer_shard=bool(eval_config.optimizer_shard),
                                                  vocab_emb_dp=bool(eval_config.word_emb_dp),
                                                  recompute=recompute_config,
                                                  gradient_aggregation_group=eval_config.gradient_aggregation_group)
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=eval_config.seq_length,
        vocab_size=eval_config.vocab_size,
        hidden_size=eval_config.embedding_size,
        ffn_hidden_size=eval_config.embedding_size * 4,
        num_layers=eval_config.num_layers,
        num_heads=eval_config.num_heads,
        expert_num=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        softmax_compute_type=ms.float16,
        use_past=False,
        eod_reset=False,
        load_ckpt_path=eval_config.ckpt_path,
        position_embedding_name=eval_config.position_embedding_name,
        word_embedding_name=eval_config.word_embedding_name,
        top_query_embedding_name=eval_config.top_query_embedding_name,
        sink_size=eval_config.sink_size,
        parallel_config=parallel_config,
    )
    print(config, flush=True)

    pangu = PanguAlpha(config)
    pangu = EvalNet_p(config, pangu, generate=True)
    pangu.set_train(False)
    model = Model(pangu)

    fake_input = Tensor(np.ones(shape=(batch_size, config.seq_length)), mstype.int32)

    if eval_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(eval_config.ckpt_path, eval_config.ckpt_dir))
    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(eval_config.ckpt_path, eval_config.ckpt_strategy_name))

    topP = 0.9
    for N_shot, shot in enumerate(["zero_shot"]):
        print(shot + "task is running")
        tokenizer = JIEBATokenizer(os.path.join(eval_config.ckpt_path, eval_config.vocab_model_name))

        cmrc2018_train = os.path.join(eval_config.data_path, eval_config.data_train_name)
        cmrc2018_dev = os.path.join(eval_config.data_path, eval_config.data_dev_name)

        DataSet_cmrc2018 = Cmrc2018_Dataset(cmrc2018_train_json=cmrc2018_train, cmrc2018_dev_json=cmrc2018_dev)
        Prompts, Answers = DataSet_cmrc2018.get_data_allshot(N_content=500, N_q=100)

        pred_ = []
        answers_ = []
        result_list = []
        logsStr = []
        for idx, (Prompt, answer) in enumerate(zip(Prompts, Answers)):
            prompt = Prompt[N_shot]

            tokenized_text = tokenizer.tokenize(prompt)
            start_sentence = tokenizer.convert_tokens_to_ids(tokenized_text)

            input_ids = np.array(start_sentence).reshape(1, -1)

            if input_ids.shape[-1] >= config.seq_length - 4:
                input_ids = input_ids[:, -900:]

            outputs = generate_samples_cftpd(model, input_ids, config.seq_length, end_token=9, top_k=0, top_p=topP,
                                             temperature=0.8)
            output_list = outputs.tolist()
            output_list = output_list[input_ids.shape[-1]:]

            answers_pred = tokenizer.convert_ids_to_tokens(output_list)
            answers_pred = "".join(tokenizer.decode(answers_pred))
            pred_.append(answers_pred.split("\n")[0])
            answers_.append(answer)
            res_str = evaluate_pairs(pred_, answers_)
            if idx % 100 == 0:
                result_list.append(f"N_shot={N_shot}, index={idx}, {res_str}")
                print(f"N_shot={N_shot}, index={idx}, {res_str}", flush=True)
        all_results = {'results_split': result_list}
        res_str = evaluate_pairs(pred_, Answers)
        logsStr.append(f"N_shot={N_shot}, {res_str}")
        all_results['results_all'] = logsStr
        with open(eval_out_file, 'w', encoding='utf-8') as out:
            json.dump(all_results, out, ensure_ascii=False)

        if os.path.isfile(eval_out_file):
            os.chmod(eval_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())
    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_learning(args)
