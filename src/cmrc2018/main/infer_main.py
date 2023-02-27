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
from src.utils.model_utils import get_ckpt_file_list, infer_load_file
from src.utils.tokenization_jieba import JIEBATokenizer
from src.utils import args_utils
from src.cmrc2018.main.data_process import Cmrc2018_Dataset
from src.cmrc2018.main.generate_topp import generate_samples_cftpd

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))
ms.set_seed(431436)


def few_shot_infer(infer_config):
    if rank_id % 8 == 0:
        infer_load_file(infer_config)

    infer_file_name = "infer_result.json"
    infer_out_file = os.path.join(infer_config.output_path, infer_file_name)

    print("result file:", infer_out_file, flush=True)

    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, max_device_memory="30GB")
    if infer_config.distribute == "True":
        management.init()
        device_num = int(management.get_group_size())
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num), flush=True)
        ms.context.reset_auto_parallel_context()
        ms.context.set_auto_parallel_context(
            parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            strategy_ckpt_load_file=os.path.join(infer_config.ckpt_path, infer_config.ckpt_strategy_name))
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        device_num = 1

    model_parallel_num = device_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = infer_config.per_batch_size * data_parallel_num

    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(
                                                      infer_config.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                  expert_parallel=infer_config.expert_parallel_num,
                                                  pipeline_stage=infer_config.stage_num,
                                                  micro_batch_num=infer_config.micro_size,
                                                  optimizer_shard=bool(infer_config.optimizer_shard),
                                                  vocab_emb_dp=bool(infer_config.word_emb_dp),
                                                  recompute=recompute_config,
                                                  gradient_aggregation_group=infer_config.gradient_aggregation_group)
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=infer_config.seq_length,
        vocab_size=infer_config.vocab_size,
        hidden_size=infer_config.embedding_size,
        ffn_hidden_size=infer_config.embedding_size * 4,
        num_layers=infer_config.num_layers,
        num_heads=infer_config.num_heads,
        expert_num=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        softmax_compute_type=ms.float16,
        use_past=False,
        eod_reset=False,
        load_ckpt_path=infer_config.ckpt_path,
        position_embedding_name=infer_config.position_embedding_name,
        word_embedding_name=infer_config.word_embedding_name,
        top_query_embedding_name=infer_config.top_query_embedding_name,
        sink_size=infer_config.sink_size,
        parallel_config=parallel_config,
    )
    print(config, flush=True)

    pangu = PanguAlpha(config)
    pangu = EvalNet_p(config, pangu, generate=True)
    pangu.set_train(False)
    model = Model(pangu)

    fake_input = Tensor(np.ones(shape=(batch_size, config.seq_length)), mstype.int32)

    if infer_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(infer_config.ckpt_path, infer_config.ckpt_dir))
    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(infer_config.ckpt_path, infer_config.ckpt_strategy_name))

    topP = 0.9
    for N_shot, shot in enumerate(["zero_shot"]):
        tokenizer = JIEBATokenizer(os.path.join(infer_config.ckpt_path, infer_config.vocab_model_name))

        cmrc2018_train = os.path.join(infer_config.data_path, infer_config.data_train_name)
        cmrc2018_test = os.path.join(infer_config.data_path, infer_config.data_test_name)

        DataSet_cmrc2018 = Cmrc2018_Dataset(cmrc2018_train_json=cmrc2018_train, cmrc2018_dev_json=cmrc2018_test)
        prompts_input, _ = DataSet_cmrc2018.get_data_allshot(N_content=infer_config.test_sample_num, N_q=100)

        pred_ = []
        result_list = []
        for idx, Prompt in enumerate(prompts_input):
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

            answers_pred = "".join(tokenizer.decode(tokenizer.convert_ids_to_tokens(output_list)))
            pred_.append(answers_pred.split("\n")[0])
            result_list.append(
                {'N_shot': N_shot, 'Index': idx, 'Prompt': prompt, 'PanGu-2B6': answers_pred.split("\n")[0]})
        all_results = {'all_results': result_list}
        with open(infer_out_file, 'w', encoding='utf-8') as out:
            json.dump(all_results, out, ensure_ascii=False)
        if os.path.isfile(infer_out_file):
            os.chmod(infer_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())
    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_infer(args)
