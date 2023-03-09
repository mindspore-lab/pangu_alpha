# 盘古alpha下游任务

## 一、Pangu-Alpha介绍

Pangu-Alpha由以鹏城实验室为首的技术团队联合攻关，首次基于“鹏城云脑Ⅱ”和国产MindSpore框架的自动混合并行模式实现在2048卡算力集群上的大规模分布式训练，训练出业界首个2000亿参数以中文为核心的预训练生成语言模型。鹏程·盘古α预训练模型支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备很强的小样本学习能力。


## 二、运行环境
- mindspore >= 1.8.1
- python >= 3.7
- jieba >= 0.42.1
- sentencepiece >= 0.1.97

## 三、微调套件安装

套件采用云端本地统一架构， **提供一键微调、评估、推理、在线部署** 等能力，提供 **命令行** 和 **API接口** 两种调用形式，可支持大模型平台的建设与使用。

### 1.本地场景安装

```shell
pip install TuningKit/Ascend_mindxsdk_mxTuningKit-3.0.0-py3-none-any.whl
```

[软件下载](https://github.com/mindspore-lab/pangu_alpha/tree/master/mxTuningKit) 安装详细教程请参考[此文档](https://www.hiascend.com/document/detail/zh/mind-sdk/30rc3/mxtuningkit/tuningkitug/mxtuningug_0001.html)解决。

### 2.ModelArts(AICC场景)安装

```shell
pip install TuningKit/Ascend_mindxsdk_mxFoundationModel-1.0.1.RC2.b001-py3-none-any.whl
```

[软件下载](https://github.com/mindspore-lab/pangu_alpha/tree/master/mxTuningKit)，安装过程遇到问题请参考[此文档](https://github.com/mindspore-lab/pangu_alpha/blob/master/mxTuningKit/%E5%BE%AE%E8%B0%83%E7%BB%84%E4%BB%B6(%E4%BA%91%E4%B8%8A%E5%9C%BA%E6%99%AF).md)解决。


## 四、支持下游任务

| 下游任务                                                                              | 任务类型             | 论文精度（pangu-2B6）  | 复现精度（pangu-2B6）   | 样本类型       |
|-----------------------------------------------------------------------------------| --------------------------- |------------------|-------------------|------------|
| [AFQMC](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/afqmc)       | 文本相似度                   | acc=64.62%       | acc=68.9%         | one-shot   |
| [CMRC2017](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/cmrc2017) | 中文阅读理解（填空型阅读理解） | acc=38.00%       | acc=37.5%         | one-shot |
| [TNEWS](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/tnews)       | 新闻短文本分类                 | acc=57.95%       | acc=60.18%        | one-shot   |
| [CMRC2018](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/cmrc2018) | 中文阅读理解（句子级填空型阅读理解） | Em/F1=1.21/16.65 | Em/F1=1.06/16.53  | zero-shot |
| [WebQA](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/webqa)       | 中文问答 | Em/F1=24/33.94   | Em/F1=23.77/33.86 | few_shot |
| [IFLYTEK](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/iflytek)       | 应用长文本分类 | acc=74.26%    | acc=73.72%  | zero-shot |
| [PD&CFT](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/pd_cft)     | 中文阅读理解（填空型阅读理解）    | PD: acc=58.05%<br />CFT: acc=42.39% | PD: acc=58.56%<br />CFT: acc=42.31% | zero-shot  |
| [DUREADER](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/dureader) | 中文阅读理解（知识问答）       | Rouge-l=21.07% | Rouge-l=28.98% | zero_shot  |
| [CMRC2019](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/cmrc2019) | 中文阅读理解（片段抽取任务）     | acc=67.89%    | acc=67.93%     | zero-shot  |
| [CSL](https://github.com/mindspore-lab/pangu_alpha/tree/master/src/csl)           | 论文关键词识别               | acc=50.9% | acc=52.1% | one-shot   |

## 其他注意事项

目前下游任务微调支持8卡执行

评估和推理支持单卡、4卡和8卡执行