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
import re
import jieba


def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                # ss = nltk.word_tokenize(temp_str)
                ss = list(jieba.cut(temp_str))
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        # ss = nltk.word_tokenize(temp_str)
        ss = list(jieba.cut(temp_str))
        segs_out.extend(ss)

    return segs_out


def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def evaluate_out_txt(gptout="", start_with=""):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0

    with open(gptout, encoding="utf-8") as f:
        lines = f.readlines()
        pairs = []
        pair = []
        s1 = len(start_with)
        s2 = len("Answer:")
        for line in lines:
            line = line.strip()
            if line.startswith(start_with) and pair == [] and len(line) > s1:
                pair.append(line[s1:])
            elif line.startswith("Answer:"):
                if len(pair) == 1:
                    pair.append(line[s2:])
                    pairs.append(pair)
                    pair = []
                else:
                    pair = []
            else:
                continue

    for (prediction, answer) in pairs:
        total_count += 1
        f1 += calc_f1_score([answer], prediction)
        em += calc_em_score([answer], prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def evaluate_pairs(pred_, ans_):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for (prediction, answer) in zip(pred_, ans_):
        total_count += 1
        f1 += calc_f1_score([answer], prediction)
        em += calc_em_score([answer], prediction)
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f"f1_score={f1_score}, em_score={em_score}, total_count={total_count}, skip_count={skip_count}"
