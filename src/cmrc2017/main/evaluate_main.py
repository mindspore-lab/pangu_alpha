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
import re
import sys
import time

import mindspore as ms
import mindspore.communication.management as management
import numpy as np
from mindspore import load_distributed_checkpoint
from mindspore.nn import TransformerRecomputeConfig, TransformerOpParallelConfig
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))

from src.model import PanguAlpha, PanguAlphaConfig, EvalNet_p, set_parse
from src.utils import args_utils
from src.utils.model_utils import get_ckpt_file_list, eval_load_file
from src.utils.tokenization_jieba import JIEBATokenizer
from src.cmrc2017.main.generate import generate_samples_new, remove_punctuation, generate_input_format

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))


def load_model(eval_config):
    if rank_id % 8 == 0:
        eval_load_file(eval_config)

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
            strategy_ckpt_load_file=os.path.join(eval_config.ckpt_path, eval_config.ckpt_strategy_name),
            enable_parallel_optimizer=False)
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        device_num = 1

    model_parallel_num = device_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num

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
    model = ms.Model(pangu)

    fake_input = ms.Tensor(np.ones(shape=(batch_size, config.seq_length)), ms.int32)
    if eval_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(eval_config.ckpt_path, eval_config.ckpt_dir))
    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(eval_config.ckpt_path, eval_config.ckpt_strategy_name))
    return model, config


def run_train():
    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    model, model_config = load_model(args)
    print("Model load finished")

    eval_file_name = "eval_result.json"
    eval_out_file = os.path.join(args.output_path, eval_file_name)
    if rank_id == 0:
        f_out = open(eval_out_file, 'a')

    test_file_path = os.path.join(args.data_path, args.data_test_name)
    test_answer_file_path = os.path.join(args.data_path, args.data_test_answer_name)

    tokenizer = JIEBATokenizer(os.path.join(args.ckpt_path, args.vocab_model_name))
    print(f"tokenizer loaded!")

    answer_dict = {}
    with open(test_answer_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(' ||| ')
            answer_dict[line_list[0]] = line_list[1].replace("\n", "")

    text_dict = {}
    question_dict = {}
    question_IDs = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        pattern = re.compile(r"<qid_\d+>")
        text_line = ''
        for line in lines:
            if pattern.match(line) == None:
                text_line += line.split(' ||| ')[-1]
            else:
                question_id = line.split(' ||| ')[0]
                question_text = line.split(' ||| ')[1].replace("\n", "")
                text_dict[question_id] = text_line
                question_dict[question_id] = question_text
                text_line = ''
                question_IDs.append(question_id)

    one_shot_words = ' 小的时候非常贪玩 ，不用功读书 。有一天，他到野外游玩,见到河边有位白发苍苍的老婆婆 ，手里拿着一根大铁棒，在石头上用力' \
                     '磨着。李白很奇怪，就上前问道：“老婆婆您这是在干什么呀？“老婆婆一边磨铁棒，一边回答说：“我想把它磨成一根绣花针。“李白被' \
                     '老婆婆的行为所感动，向她深深行了个礼，回家发奋读书去了。\n李白小的时候非常贪玩，不用功读书。'

    few_hot_words = "有一天，他到野外游玩,见到河边有位白发苍苍的老婆婆 ，手里拿着一根大铁棒，在石头上用力磨着。李白很奇怪，就上前 问道：“老" \
                    "婆婆您这是在干什么呀？“老婆婆一边磨铁棒，一边回答说：“我想把它磨成一根绣花针。“李白被老婆婆的行为所感动，向她深深行了" \
                    "个礼，回家发奋读书去了。\n李白小的时候非常贪玩，不用功读书。" + '小斑点狗仰起头，问妈妈：“为什么你的影子那么长，我的' \
                                                        '影子那么短呢？”\n小斑点狗和妈妈去散步，妈妈的影子长长的，小斑点狗的影子短短的。'

    shot_dict = {'zero-shot': '',
                 'one-shot': one_shot_words,
                 'few-shot': few_hot_words}
    start_time = time.time()

    # 'zero-shot', 'one-shot', 'few-shot'
    eval_result = list()
    for shot in ['one-shot']:
        shot_words = shot_dict[shot]

        print(f"test num:{len(question_IDs)}", flush=True)
        N_sample = min(args.test_sample_num, len(question_IDs))
        num_match = 0
        generate_num = 5  # Limit Token num
        tested_num = 0
        for question_id in question_IDs[:N_sample]:
            para_context = text_dict[question_id]
            answer_text = answer_dict[question_id]

            para_context = para_context.replace(' ', '')
            input_str = generate_input_format(para_context, model_config, generate_num, shot_words, tokenizer)
            tokenized_text = tokenizer.tokenize(input_str)
            start_sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
            input_ids = np.array(start_sentence).reshape(1, -1)
            # print("input_ids : ", input_ids)

            output_list = generate_samples_new(model=model, origin_inputs=np.array(input_ids),
                                               seq_length=model_config.seq_length, gen_len=generate_num, end_token=9)

            output_list = [int(i) for i in output_list]

            generate_ids = output_list[len(input_ids[0]):]
            result_gen_txt = tokenizer.convert_ids_to_tokens(generate_ids)
            result_gen_txt = tokenizer.decode(result_gen_txt)
            # print("result_gen_txt : ", result_gen_txt)

            # process out result
            processed_gen_txt = remove_punctuation(result_gen_txt)
            processed_gen_txt = processed_gen_txt[:len(answer_text)]
            if processed_gen_txt == answer_text:
                num_match += 1
            tested_num += 1

            if rank_id == 0 and tested_num % 100 == 0:
                print("total samples: ", N_sample, "test: ", tested_num, "ACC: ", num_match / tested_num, flush=True)

                eval_result.append(
                    f"task: {shot}, tested samples: {tested_num}, match num: {num_match}, ACC: {num_match / tested_num}"
                )

        end_time = time.time()
        acc = num_match / tested_num
        print(f"Test cmrc, cost time: {end_time - start_time}! \nAcc: {acc}\n Total samples: {N_sample}", flush=True)
        eval_result.append(
            f"task : {shot}, total samples: {N_sample}, match num:{num_match}, ACC：{num_match / tested_num}")

        if rank_id == 0:
            json.dump(eval_result, f_out, ensure_ascii=False)
            f_out.close()
            if os.path.isfile(eval_out_file):
                os.chmod(eval_out_file, 0o750)


if __name__ == "__main__":
    run_train()
