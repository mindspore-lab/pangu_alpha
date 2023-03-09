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

import mindspore.common.dtype as mstype
import numpy as np
from mindspore.common.tensor import Tensor
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")))
from src.utils import args_utils
from src.model import set_parse
from src.utils.tokenization_jieba import JIEBATokenizer
from src.cmrc2019.main.load_model import load_model


def compare_list(ground_truth, prediction):
    right_count = 0
    min_len = len(ground_truth) if len(ground_truth) < len(prediction) else len(prediction)
    gap_count = len(ground_truth) - min_len

    for k in range(min_len):
        if str(ground_truth[k]) == str(prediction[k]):
            right_count += 1

    final_right_count = right_count - gap_count
    if final_right_count < 0:
        final_right_count = 0
    return final_right_count


def run_eval_cmrc2019(shot, eval_args, config, model, tokenizer):
    qac = 0
    pac = 0
    qac_score = 0
    pac_score = 0
    total_question_count = 0
    skip_question_count = 0
    total_passage_count = 0

    eods_token = 9

    with open(os.path.join(eval_args.data_path, eval_args.data_dev_name), "r", encoding="utf-8") as f:
        file_json_obj = json.load(f)
    all_data_list = file_json_obj['data']

    one_shot_words = '伏明霞，这个中国的小明星，像天边一道绚丽彩霞，11岁就照亮了天际，成为亿万人瞩目的世界冠军；' \
                     '14岁成为巴塞罗那奥运会女子10米跳台跳水金牌得主。'

    few_hot_words = "伏明霞，这个中国的小明星，像天边一道绚丽彩霞，11岁就照亮了天际，成为亿万人瞩目的世界冠军；" \
                    "14岁成为巴塞罗那奥运会女子10米跳台跳水金牌得主。\n" \
                    + '歌德，德国诗人、剧作家和思想家。生于法兰克福一个富有市民家庭。\n' \
                    + '从前，蒙古族有两个汗，一个叫阿拉齐汗，一个叫阿吾兰齐汗。阿拉齐汗经常带着人马侵略阿吾兰齐汗的国家。\n'
    shot_dict = {'zero-shot': '',
                 'one-shot': one_shot_words,
                 'few-shot': few_hot_words}
    shot_words = shot_dict[shot]

    result_list = []
    logsStr = []
    for idx, sample_data in enumerate(all_data_list):
        answer = sample_data['answers']
        text_context = sample_data['context']
        choices = sample_data['choices']

        blank_num = len(answer)
        predict_ans_list = []
        loss_matrix = [[1 for i in range(len(choices))] for j in range(blank_num)]
        for i in range(blank_num):
            text_context_list = text_context.split("[BLANK" + str(i + 1) + "]")
            left_id = max(0, len(text_context_list[0]) - 300)
            right_id = max(0, len(text_context_list[1]) - 200)
            short_text_context = text_context[left_id:(len(text_context) - right_id)]
            split_out = short_text_context.split("[BLANK" + str(i + 1) + "]")

            # 把剩余的其他空格的符号删掉
            right_text = re.sub('\[BLANK.*?\]', '', split_out[1])
            left_text = re.sub('\[BLANK.*?\]', '', split_out[0])

            # 把每个选项替换到空格中计算PPL
            choice_ppl_list = []
            for index in range(len(choices)):
                choice_text = choices[index]

                input_str = shot_words + "\n" + left_text + choice_text + right_text
                tokenized_text = tokenizer.tokenize(input_str)
                start_sentence_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                input_ids = np.array(start_sentence_ids).reshape(1, -1)
                if input_ids.shape[-1] > config.seq_length:
                    input_ids = input_ids[:, :config.seq_length + 1]

                pad_length = config.seq_length + 1 - input_ids.shape[-1]

                # 增加mask，避免其他部分被计算loss
                blank_left_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(shot_words + "\n" + left_text))
                blank_right_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                    shot_words + "\n" + left_text + choice_text + right_text))
                input_mask = [0] * config.seq_length
                for idx in range(len(blank_left_tokens) - 1, len(blank_right_tokens) - 1):
                    input_mask[idx] = 1
                input_mask_ids = Tensor(input_mask, mstype.float32)

                input_ids = np.pad(input_ids, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, eods_token))

                loss = model.predict(Tensor(input_ids, mstype.int32), input_mask_ids).asnumpy()
                choice_ppl_list.append(loss)

                loss_matrix[i][index] = loss

            # 选择最小的PPL选项作为当前空格的答案
            ans_indx = [index for index, value in sorted(list(enumerate(choice_ppl_list)), key=lambda x: x[1])]

            predict_ans_list.append(ans_indx[0])

        # 通过求loss矩阵的不同行不同列的最小和来选择答案
        data = np.array(loss_matrix)
        data = data.reshape(blank_num, len(choices))
        r_index, c_index = linear_sum_assignment(data, maximize=False)
        predict_res = [1 for i in range(blank_num)]
        for i in range(len(r_index)):
            predict_res[r_index[i]] = int(c_index[i])
        right_question_count = compare_list(answer, predict_res)

        qac += right_question_count
        pac += (right_question_count == len(answer))

        total_question_count += len(answer)
        skip_question_count += len(answer) - len(predict_res)
        total_passage_count += 1
        qac_score = 100.0 * qac / total_question_count
        pac_score = 100.0 * pac / total_passage_count

        # 按照评测输出格式保存结果
        if idx % 30 == 0:
            result_list.append(
                f"N_shot={shot}, qac_score={qac_score}, pac_score={pac_score},"
                f" total_question_count={total_question_count}, skip_question_count={skip_question_count}")
            print(f"N_shot={shot}, qac_score={qac_score}, pac_score={pac_score},"
                  f" total_question_count={total_question_count}, skip_question_count={skip_question_count}",
                  flush=True)

    result_list.append(
        f"N_shot={shot}, qac_score={qac_score}, pac_score={pac_score},"
        f" total_question_count={total_question_count}, skip_question_count={skip_question_count}")
    print(f"N_shot={shot}, qac_score={qac_score}, pac_score={pac_score},", flush=True)

    all_result = {'results_split': result_list}
    logsStr.append(
        f"N_shot={shot}, qac_score={qac_score}, pac_score={pac_score}, total_question_count={total_question_count},"
        f" skip_question_count={skip_question_count}")
    all_result['results_all'] = logsStr
    with open(os.path.join(eval_args.output_path, 'eval_result.json'), 'w', encoding='utf-8') as out:
        json.dump(all_result, out, ensure_ascii=False)


if __name__ == "__main__":

    print('process id:', os.getpid())

    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    model, config = load_model(args, True)

    tokenizer = JIEBATokenizer(os.path.join(args.ckpt_path, args.vocab_model_name))
    # for shot in ['zero-shot', 'one-shot', 'few-shot']:
    for shot in ['zero-shot']:
        run_eval_cmrc2019(shot, args, config, model, tokenizer)
