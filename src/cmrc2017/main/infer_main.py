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
from src.model import PanguAlphaConfig, PanguAlpha, EvalNet_p, set_parse
from src.utils import args_utils
from src.utils.model_utils import get_ckpt_file_list, infer_load_file
from src.utils.tokenization_jieba import JIEBATokenizer
from src.cmrc2017.main.generate import generate_samples_new, generate_input_format, remove_punctuation

os.environ['HCCL_CONNECT_TIMEOUT'] = '1800'
rank_id = int(os.getenv('RANK_ID', default=0))
device_id = int(os.getenv('DEVICE_ID', default=0))


def load_model(infer_config):
    if rank_id % 8 == 0:
        infer_load_file(infer_config)

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
            strategy_ckpt_load_file=os.path.join(infer_config.ckpt_path, infer_config.ckpt_strategy_name),
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
    model = ms.Model(pangu)

    fake_input = ms.Tensor(np.ones(shape=(batch_size, config.seq_length)), ms.int32)

    if infer_config.distribute == "True":
        predict_layout = model.infer_predict_layout(fake_input)
    else:
        predict_layout = None

    ckpt_file_list = get_ckpt_file_list(os.path.join(infer_config.ckpt_path, infer_config.ckpt_dir))
    load_distributed_checkpoint(pangu, ckpt_file_list, predict_layout,
                                os.path.join(infer_config.ckpt_path, infer_config.ckpt_strategy_name))
    return model, config


def run_infer(infer_args):
    res_dir = os.path.join(infer_args.output_path, 'infer_result_rank_' + str(device_id))
    infer_file_name = "infer_result.json"

    infer_out_file = os.path.join(infer_args.output_path, infer_file_name)
    f_out = open(infer_out_file, 'a')

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    print("result file:", infer_out_file, flush=True)

    model, model_config = load_model(infer_args)
    print("Model load finished", flush=True)

    test_file_path = os.path.join(infer_args.data_path, infer_args.data_test_name)

    tokenizer = JIEBATokenizer(os.path.join(infer_args.ckpt_path, infer_args.vocab_model_name))
    print(f"tokenizer loaded!", flush=True)

    text_dict = {}
    question_dict = {}
    question_IDs = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        pattern = re.compile(r"<qid_\d+>")
        text_line = ''
        for line in lines:
            if pattern.match(line) is None:
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
    res_list = list()
    # shot : 'zero-shot', 'few-shot'
    for shot in ['one-shot']:
        shot_words = shot_dict[shot]

        print(f"test num:{len(question_IDs)}", flush=True)
        N_sample = infer_args.test_sample_num
        print('test samples num : ', N_sample, flush=True)

        generate_num = 5  # 限制生成的Token的数量
        for question_id in question_IDs[:N_sample]:
            para_context = text_dict[question_id]
            question_text = question_dict[question_id]

            # 去掉原始数据中空格
            para_context = para_context.replace(' ', '')
            question_text = question_text.replace(' ', '')

            # 输入形式
            input_str = generate_input_format(para_context, model_config, generate_num, shot_words, tokenizer)

            # 转为Token id
            tokenized_text = tokenizer.tokenize(input_str)
            start_sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
            input_ids = np.array(start_sentence).reshape(1, -1).tolist()

            output_list = generate_samples_new(model=model, origin_inputs=np.array(input_ids),
                                               seq_length=model_config.seq_length, gen_len=generate_num, end_token=9)

            output_list = [int(i) for i in output_list]
            output_ids = output_list[len(input_ids):]

            output_txt = tokenizer.convert_ids_to_tokens(output_ids)
            output_txt = tokenizer.decode(output_txt)

            generate_ids = output_list[len(input_ids[0]):]
            result_gen_txt = tokenizer.convert_ids_to_tokens(generate_ids)
            result_gen_txt = tokenizer.decode(result_gen_txt)

            # 处理输出结果
            processed_gen_txt = remove_punctuation(result_gen_txt)

            if rank_id == 0:
                print(f"原始文章={para_context}\n{question_text}\n", flush=True)
                print(f"生成答案={processed_gen_txt}\n", flush=True)
                print(f"\n\n", flush=True)
                res_list.append(f"原始文章:{para_context}\n{question_text}\n生成输出:{output_txt}"
                                f"\n生成答案:{processed_gen_txt}\n")

        end_time = time.time()
        print(f"Test cmrc, cost time: {end_time - start_time}!", flush=True)

        if rank_id == 0:
            json.dump(res_list, f_out, ensure_ascii=False)
            f_out.close()
            if os.path.isfile(infer_out_file):
                os.chmod(infer_out_file, 0o750)


def run_main():
    print('process id:', os.getpid(), flush=True)

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args, flush=True)

    run_infer(args)


if __name__ == "__main__":
    run_main()
