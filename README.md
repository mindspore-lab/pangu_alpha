# 盘古alpha下游任务

## 一、Pangu-Alpha介绍

Pangu-Alpha由以鹏城实验室为首的技术团队联合攻关，首次基于“鹏城云脑Ⅱ”和国产MindSpore框架的自动混合并行模式实现在2048卡算力集群上的大规模分布式训练，训练出业界首个2000亿参数以中文为核心的预训练生成语言模型。鹏程·盘古α预训练模型支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备很强的小样本学习能力。


## 二、运行环境
- mindspore >= 1.8.1
- python >= 3.7
- jieba >= 0.42.1
- sentencepiece >= 0.1.97

## 三、大模型使能套件安装

大模型使能套件是华为大模型全流程平台的重要组成部分，套件采用云端本地统一架构，可同时面向本地和 Modelarts(AICC)场景使用， **提供一键微调、评估、推理、在线部署** 等能力，提供 **命令行** 和 **API接口** 两种调用形式，并提供大模型平台参考设计，可助力企业快速实现大模型全流程私有化部署。

### 1.本地场景安装

```shell
pip install TuningKit/Ascend_mindxsdk_mxTuningKit-3.0.0-py3-none-any.whl
```

[下载](https://www.hiascend.com/software/mindx-sdk/community) Ascend-mindxsdk-mxtuningkit_3.0.0_linux.tar.gz解压，安装详细教程请参考[此文档](https://www.hiascend.com/document/detail/zh/mind-sdk/30rc3/mxtuningkit/tuningkitug/mxtuningug_0001.html)解决。

### 2.ModelArts(AICC场景)安装

```shell
pip install TuningKit/Ascend_mindxsdk_mxFoundationModel-1.0.1.RC2.b001-py3-none-any.whl
```

[软件下载](https://github.com/mindspore-lab/pangu_alpha/tree/master/mxTuningKit)，安装过程遇到问题请参考[此文档](https://github.com/mindspore-lab/pangu_alpha/blob/master/mxTuningKit/%E5%BE%AE%E8%B0%83%E7%BB%84%E4%BB%B6(%E4%BA%91%E4%B8%8A%E5%9C%BA%E6%99%AF).md)解决。


## 四、支持下游任务

| 下游任务      | 任务类型             | 论文精度（pangu-2B6）               | 复现精度（pangu-2B6）                   | 样本类型       |
| -------------------------------------------------------------------------------------------------- | --------------------------- | ----------------------------------- | ----------------------------------- |------------|
| [AFQMC](https://gitee.com/foundation-models/tk-models/tree/master/models/pangu_alpha/afqmc)       | 文本相似度                   | acc=64.62%                          | acc=68.9%                           | one-shot   |
| [CMRC2017](https://gitee.com/foundation-models/tk-models/tree/master/models/pangu_alpha/cmrc2017) | 中文阅读理解（填空型阅读理解） | acc=38.00% | acc=37.5% | one-shot |

## 其他注意事项

目前下游任务微调支持2卡、4卡和8卡执行
评估和推理支持单卡、2卡、4卡和8卡执行