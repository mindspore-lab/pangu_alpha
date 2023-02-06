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
import mindspore as ms
import numpy as np
from mindspore.common.tensor import Tensor


def top_k_logits(logits, top_k=0, top_p=0.9, filter_value=-float(0)):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < np.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_indices = np.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]

        # cumulative_probs = np.cumsum(softmax(sorted_logits), axis=-1)
        cumulative_probs = np.cumsum(sorted_logits, axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_samples_new(model, origin_inputs, seq_length, gen_len=50, end_token=50256, temperature=1.0):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape

    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 6))

    while valid_length < (origin_inputs.shape[1] + gen_len):
        inputs = Tensor(input_ids, ms.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length - 1, :]

        probs = probs / temperature

        probs = top_k_logits(probs)
        # p = softmax(probs)
        sums = sum(probs)
        if sums == 0:
            sums = 1
        p = probs / sums
        target_index = np.random.choice(len(p), p=p)

        if target_index == end_token or valid_length == (origin_inputs.shape[1] + gen_len - 1):
            outputs = input_ids
            break
        input_ids[0][valid_length] = target_index
        valid_length += 1

    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    return outputs


def generate_input_format(para_context, config, generate_num, shot_words, tokenizer):
    para_list = para_context.split('\n')

    for i in range(len(para_list)):
        sentence = para_list[i]
        if len(sentence.split('XXXXX')) == 2:
            question_sentence_index = i
            break

    question_sentences = []
    para_sentences = []
    for i in range(len(para_list)):
        sentence = para_list[i]
        if question_sentence_index - 3 < i <= question_sentence_index:
            question_sentences.append(sentence)
        else:
            para_sentences.append(sentence)
    question_text_res = ''.join(question_sentences)
    para_text = ''.join(para_sentences)

    if shot_words:
        input_str = shot_words + "\n" + para_text + question_text_res.split('XXXXX')[0]
    else:
        input_str = para_text + question_text_res.split('XXXXX')[0]

    tokenized_text = tokenizer.tokenize(input_str)

    if len(tokenized_text) > config.seq_length - generate_num:
        cut_length = (config.seq_length - generate_num - 2) * -1
        input_str = input_str[cut_length:]

    return input_str


def remove_punctuation(in_str):
    # remove punctuation
    in_str = str(in_str).lower().strip()
    in_str = in_str.replace('<pad>', '')
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；', '’', '《',
               '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』', ' ']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)
