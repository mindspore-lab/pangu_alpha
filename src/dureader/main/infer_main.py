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
from src.dureader.main.generate_topp import generate_samples
from src.dureader.main.load_model import load_model


def few_shot_infer(infer_config):
    model, config = load_model(infer_config, False)

    infer_file_name = "infer_result.json"
    infer_out_file = os.path.join(infer_config.output_path, infer_file_name)

    print("result file:", infer_out_file, flush=True)

    tokenizer = JIEBATokenizer(os.path.join(infer_config.ckpt_path, infer_config.vocab_model_name))
    out_result = {}
    for shot in ["zero_shot"]:
        with open(os.path.join(infer_config.data_path, infer_config.data_test_name), "r", encoding="utf-8") as f:
            data_file = f.readlines()
            index = 0
        demo = "阅读：有没有人看过?今天听说冰血暴特别好看,豆瓣上评分也很高。还有大家最近都在看什么剧,最近看完了几部以前很经典的剧,都是全部" \
               "完结的,现在的新剧很多,大家推荐一下, 第一季,相当精彩,久久不能平息! 问：冰血暴好看吗? 答：第一季，相当精彩\n\n" \
               "阅读：南宁属于广西省 南宁是一座历史悠久的文化古城,同时也是一个以壮族为主的多民族和睦相处的现代化城市,壮族是世代居住在" \
               "本地的土著民族。问：南宁属于什么省? 答：南宁属于广西省。\n\n" \
               "阅读：壁虎在正常情况下是不会咬人的，若是激怒了小壁虎，它同样是会咬人的壁虎通常以昆虫为食。 问：壁虎会咬人吗? 答：壁虎在正常" \
               "情况下是不会咬人的，若是激怒了小壁虎，它同样是会咬人的壁虎通常以昆虫为食。\n\n" \
               "阅读：脸部颧骨高不管你留什么发型你颧骨高的问题还是始终会存在的呀~避免不良别人对你说三道四的噢。问：要是颧骨高适合什么发型啊? " \
               "答：不管留什么发型，颧骨高的问题还是始终会存在的。\n\n" \
               "阅读：360度旋转鞋架,不锈钢旋转鞋柜,有多种形状设计，功能也略有不同，其优点是：在室内摆放更加方便，不锈钢的材质结实耐用，" \
               "内部小格子空间规划、分类的存放鞋子物品，查找和整理更舒适便捷。 问：360旋转鞋柜好不好? 答：其优点是：在室内摆放更加方便，" \
               "不锈钢的材质结实耐用，内部小格子空间规划、分类的存放鞋子物品，查找和整理更舒适便捷。\n\n" \
               "阅读：资格灵活组排或者单双 段位要求在钻1以上不过这东西真的坑  我自己王者嘛。问：峡谷之巅怎么进? 答：灵活组排或者单双，" \
               "段位钻一以上才能进去。\n\n"
        demo_one = "阅读：资格灵活组排或者单双 段位要求在钻1以上不过这东西真的坑  我自己王者嘛。问：峡谷之巅怎么进？答：灵活组排或者单双，" \
                   "段位钻一以上才能进去。\n\n"
        results = {}
        if "few_shot" == shot:
            example = demo
            example = "".join(example)
        elif "one_shot" == shot:
            example = demo_one
        else:
            example = ""
        cnt = 0
        for instance in data_file[:infer_config.test_sample_num]:
            cnt += 1
            instance = json.loads(instance)
            documents = instance.get("documents", [])
            q = instance["question"]
            question_type = instance["question_type"]
            question_id = instance["question_id"]
            results[question_id] = {"question_id": question_id, "question": q,
                                    "question_type": question_type, "answers_pred": []}
            for document in documents:
                if not document["is_selected"]: continue
                paragraphs = document["paragraphs"]
                context_ = paragraphs[0]
                index += 1
                input_str = f"{example}阅读：{context_} 问：{q}？答："
                tokenized_text = tokenizer.tokenize(input_str)

                start_sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
                input_ids = np.array(start_sentence).reshape(1, -1)
                if input_ids.shape[-1] >= config.seq_length:
                    input_ids = input_ids[:, -1000:]
                output_ids = generate_samples(model, input_ids, config.seq_length, end_token=tokenizer.eot_id,
                                              top_p=0.9, temperature=0.7)
                output_list = output_ids.tolist()
                output_list = output_list[input_ids.shape[-1]:]

                answers_pred = "".join(tokenizer.decode(tokenizer.convert_ids_to_tokens(output_list)))
                answers_pred = answers_pred.split("\n")[0]
                results[question_id]["answers_pred"].append(answers_pred)
            if cnt % 100 == 0:
                print("cnt is : ", cnt, flush=True)
            out_result[shot] = results
            print(f"task {shot} infer success!")

    with open(infer_out_file, 'w', encoding='utf-8') as out:
        json.dump(out_result, out, ensure_ascii=False)
    if os.path.isfile(infer_out_file):
        os.chmod(infer_out_file, 0o750)


if __name__ == "__main__":
    print('process id:', os.getpid())
    args = args_utils.get_args(True)
    set_parse(args)
    print('args : ', args)

    few_shot_infer(args)
