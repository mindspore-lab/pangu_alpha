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


def ckpt_copy_tar_new(obs_path, target_path="/cache/ckpt"):
    """
        requires the obs_path to be a complete name
        Copy tar file from the obs to the /cache/
    """
    sub_name_list = ['_0.tar', '_1.tar', '_2.tar', '_3.tar']
    for item in sub_name_list:
        sub_name = obs_path + item
        tmp_name = 'model.tar'

        # mox.file.copy(sub_name, os.path.join(target_path, tmp_name))
        os.system('mv {} {}'.format(sub_name, os.path.join(target_path, tmp_name)))
        os.system('cd {}; tar -xvf {}'.format(target_path, tmp_name))


def finetune_load_file(finetune_args_config):
    if os.path.exists(os.path.join(finetune_args_config.pretrained_model_path,
                                   finetune_args_config.position_embedding_name)):
        print("loading " + finetune_args_config.position_embedding_name, flush=True)
    else:
        print(finetune_args_config.position_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(finetune_args_config.pretrained_model_path,
                                   finetune_args_config.top_query_embedding_name)):
        print("loading " + finetune_args_config.top_query_embedding_name, flush=True)
    else:
        print(finetune_args_config.top_query_embedding_name + " not exits !", flush=True)

    if os.path.exists(
            os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.word_embedding_name)):
        print("loading " + finetune_args_config.word_embedding_name, flush=True)
    else:
        print(finetune_args_config.word_embedding_name + " not exits !", flush=True)

    if os.path.exists(
            os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.ckpt_strategy_name)):
        print("loading " + finetune_args_config.ckpt_strategy_name, flush=True)
    else:
        print(finetune_args_config.ckpt_strategy_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(finetune_args_config.data_path, finetune_args_config.data_train_name)):
        print("loading " + finetune_args_config.data_train_name, flush=True)
    else:
        print(finetune_args_config.data_train_name + "not exits !", flush=True)

    if os.path.exists(os.path.join(finetune_args_config.data_path, finetune_args_config.data_dev_name)):
        print("loading " + finetune_args_config.data_dev_name, flush=True)
    else:
        print(finetune_args_config.data_dev_name + " not exits !", flush=True)

    if os.path.exists(
            os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.vocab_model_name)):
        print("loading " + finetune_args_config.vocab_model_name, flush=True)
    else:
        print(finetune_args_config.vocab_model_name + " not exits !", flush=True)

    if os.path.exists(
            os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.vocab_vocab_name)):
        print("loading " + finetune_args_config.vocab_vocab_name, flush=True)
    else:
        print(finetune_args_config.vocab_vocab_name + " not exits !", flush=True)

    if not os.path.exists(os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.ckpt_dir)):
        print("loading " + finetune_args_config.ckpt_dir, flush=True)
        if not os.path.exists(
                os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.ckpt_dir)):
            os.mkdir(os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.ckpt_dir))
        ckpt_copy_tar_new(os.path.join(finetune_args_config.pretrained_model_path, finetune_args_config.ckpt_name),
                          target_path=os.path.join(finetune_args_config.pretrained_model_path,
                                                   finetune_args_config.ckpt_dir))
    else:
        print(finetune_args_config.ckpt_dir + " exits !", flush=True)

    print("data are ready!", flush=True)


def eval_load_file(eval_args_config):
    if os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.position_embedding_name)):
        print("loading " + eval_args_config.position_embedding_name, flush=True)
    else:
        print(eval_args_config.position_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.top_query_embedding_name)):
        print("loading " + eval_args_config.top_query_embedding_name, flush=True)
    else:
        print(eval_args_config.top_query_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.word_embedding_name)):
        print("loading " + eval_args_config.word_embedding_name, flush=True)
    else:
        print(eval_args_config.word_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.ckpt_strategy_name)):
        print("loading " + eval_args_config.ckpt_strategy_name, flush=True)
    else:
        print(eval_args_config.ckpt_strategy_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(eval_args_config.data_path, eval_args_config.data_dev_name)):
        print("loading " + eval_args_config.data_dev_name, flush=True)
    else:
        print(eval_args_config.data_dev_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.vocab_model_name)):
        print("loading " + eval_args_config.vocab_model_name, flush=True)
    else:
        print(eval_args_config.vocab_model_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.vocab_vocab_name)):
        print("loading " + eval_args_config.vocab_vocab_name, flush=True)
    else:
        print(eval_args_config.vocab_vocab_name + " not exits !", flush=True)

    if not os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.ckpt_dir)):
        print("loading " + eval_args_config.ckpt_dir, flush=True)
        if not os.path.exists(os.path.join(eval_args_config.ckpt_path, eval_args_config.ckpt_dir)):
            os.mkdir(os.path.join(eval_args_config.ckpt_path, eval_args_config.ckpt_dir))
        ckpt_copy_tar_new(os.path.join(eval_args_config.ckpt_path, eval_args_config.ckpt_name),
                          target_path=os.path.join(eval_args_config.ckpt_path, eval_args_config.ckpt_dir))
    else:
        print(eval_args_config.ckpt_dir + " exits !", flush=True)

    print("data are ready!", flush=True)


def infer_load_file(infer_args_config):
    if os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.position_embedding_name)):
        print("loading " + infer_args_config.position_embedding_name, flush=True)
    else:
        print(infer_args_config.position_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.top_query_embedding_name)):
        print("loading " + infer_args_config.top_query_embedding_name, flush=True)
    else:
        print(infer_args_config.top_query_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.word_embedding_name)):
        print("loading " + infer_args_config.word_embedding_name, flush=True)
    else:
        print(infer_args_config.word_embedding_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.ckpt_strategy_name)):
        print("loading " + infer_args_config.ckpt_strategy_name, flush=True)
    else:
        print(infer_args_config.ckpt_strategy_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(infer_args_config.data_path, infer_args_config.data_test_name)):
        print("loading " + infer_args_config.data_test_name, flush=True)
    else:
        print(infer_args_config.data_test_name + "not exits !", flush=True)

    if os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.vocab_model_name)):
        print("loading " + infer_args_config.vocab_model_name, flush=True)
    else:
        print(infer_args_config.vocab_model_name + " not exits !", flush=True)

    if os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.vocab_vocab_name)):
        print("loading " + infer_args_config.vocab_vocab_name, flush=True)
    else:
        print(infer_args_config.vocab_vocab_name + " not exits !", flush=True)

    if not os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.ckpt_dir)):
        print("loading " + infer_args_config.ckpt_dir, flush=True)
        if not os.path.exists(os.path.join(infer_args_config.ckpt_path, infer_args_config.ckpt_dir)):
            os.mkdir(os.path.join(infer_args_config.ckpt_path, infer_args_config.ckpt_dir))
        ckpt_copy_tar_new(os.path.join(infer_args_config.ckpt_path, infer_args_config.ckpt_name),
                          target_path=os.path.join(infer_args_config.ckpt_path, infer_args_config.ckpt_dir))
    else:
        print(infer_args_config.ckpt_dir + " exits !", flush=True)

    print("data are ready!", flush=True)


def get_ckpt_file_list(ckpt_path, ckpt_name_prefix='filerted_', ckpt_num=512):
    returned_list = []
    for i in range(0, ckpt_num):
        returned_list.append('{}{}.ckpt'.format(ckpt_name_prefix, i))
    returned_list = [os.path.join(ckpt_path, item) for item in returned_list if 'embedding' not in item]
    for item in returned_list:
        f_size = os.path.getsize(item)
        f_mb = f_size / float(1024) / 1024
        print(item, " :{:.2f} MB ".format(f_mb))
    print("get ckpt file list len is : ", len(returned_list), flush=True)

    return returned_list
