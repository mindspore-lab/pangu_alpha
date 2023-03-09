# 任务介绍
语义相似度-AFQMC 
- CMRC2019 (`Chinese Machine Reading Comprehension`) ：第三届“讯飞杯”中文机器阅读理解（CMRC 2019）的任务是句子级填空型阅读理解（Sentence Cloze-Style Machine Reading Comprehension, SC-MRC）。 根据给定的一个叙事篇章以及若干个从篇章中抽取出的句子，通过建立模型将候选句子精准的填回原篇章中，使之成为完整的一篇文章。

# 资源准备
- 建立 `dataset`、`ckpt`、`output`文件夹
    - 本地场景时这些文件夹的建立无限制
    - **<font color=#FF000 >云上场景时先上传代码到OBS中，然后在OBS中建立这些文件夹，并下载文件到对应文件夹中</font>**
- 下载 [预训练文件](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha#user-content-%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD) 于`ckpt`文件夹
- 下载 [CMRC2019数据集](https://hfl-rc.github.io/cmrc2019/) 于`dataset`文件夹
- 下载 [并行策略ckpt文件](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/strategy_load_ckpt/pangu_alpha_2.6B_ckpt_strategy.ckpt) 于`ckpt`文件夹
- 下载 [jieba分词的词典文件](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha-GPU/src/branch/master/inference_mindspore_gpu/tokenizer) 于`ckpt`文件夹

注意: 除数据集外，其余文件为下游任务共用文件，下载一次即可, finetune时入参为pretrained_model_path。

# 本地场景使用
### 1.环境准备
- 确认成功安装TuningKit中的mxTuningKit.whl
- 生成RANK_TABLE_FILE所需文件
```bash
python /pmxTuningKit/hccl_tools.py --device_num "[0,8)" # 指定卡号[0,4)、[4,8)、[0,8)皆可
```

### 2.模型评估和推理
- 注意 `run_**_dist.sh` 中涉及到的传参路径要换成绝对路径
- 注意 `run_**_dist.sh` 中涉及到的传参路径权限修改为750
- `hccl_xp_xxxx.json`为上述命令生成文件，[DEVICE_NUM]为NPU卡数量，[DEVICE_START]为当前使用的NPU卡中第一张卡的编号

评估命令为
```bash
# 单卡运行
tk evaluate --quiet \
            --boot_file_path=/path/tk-models/models/pangu_alpha/cmrc2019/src/evaluate_main.py \
            --data_path=/path/pangu/cmrc2019-master/data/ \
            --output_path=/path/output/cmrc2019/eval/ \
            --ckpt_path=/path/pangu/2B6/ \
            --model_config_path=/path/pangu_alpha/src/cmrc2019/configs/eval_model_config_pangu_cmrc2019.yaml
	    
# 分布式运行命令
bash run_eval_dist.sh /path/hccl_xp_xxxx.json [DEVICE_NUM] [DEVICE_START]
```
- 注意：修改对应的`model_config`中的`distribute`参数，并行为True，单卡为False

推理命令为
```bash
# 单卡运行
tk infer --quiet \
         --boot_file_path=/path/pangu_alpha/src/cmrc2019/main/infer_main.py \
         --data_path=/path/pangu/cmrc2019-master/data/ \
         --output_path=/path/output/cmrc2019/infer/ \
         --ckpt_path=/path/pangu/2B6/ \
         --model_config_path=//path/pangu_alpha/src/cmrc2019/configs/infer_model_config_pangu_cmrc2019.yaml
	 
# 分布式运行命令
bash run_infer_dist.sh /path/hccl_xp_xxxx.json [DEVICE_NUM] [DEVICE_START]
```
- 注意：修改对应的`model_config`中的`distribute`参数，并行为True，单卡为False

# ModelArts(AICC场景)使用
### 1.环境准备
- 确认成功安装TuningKit中的mxFoundationModel.whl
- 如需构建镜像，请参照[文档](https://gitee.com/foundation-models/tk-models/tree/master/tools/docker/modelarts)，权限申请请联系仓库管理员。
- `app_config.yaml`和`model_config.yaml`已经随着代码被上传到OBS中，注意在app_config.yaml中指定OBS中的各个路径。(主要修改`data_path、output_path、code_url、boot_file_path、log_path、user_image_url、pretrained_model_path或ckpt_path`等参数)

```bash
cd /path/cmrc2019/configs/ && ls
## eval_app_config_pangu_cmrc2019.yaml eval_model_config_pangu_cmrc2019.yaml...
```

### 2.模型微调、评估和推理
评估命令为
```shell
# 注意job_name参数不能重复
fm evaluate --scenario modelarts \
            --app_config obs://path/cmrc2019/configs/eval_app_config_pangu_acmrc2019.yaml \
            --model_config_path obs://path/cmrc2019/configs/eval_model_config_pangu_cmrc2019.yaml \
            --job_name cmrc2019_eval_100801
```

推理命令为
```shell
# 注意job_name参数不能重复
fm infer --scenario modelarts \
         --app_config obs://path/cmrc2019/configs/infer_app_config_pangu_cmrc2019.yaml \
         --model_config_path obs://path/cmrc2019/configs/infer_model_config_pangu_cmrc2019.yaml \
         --job_name cmrc2019_infer_100801
```

### 3.查看任务运行状态
```shell
fm job-status --scenario modelarts --app_config obs://path/cmrc2019/configs/eval_app_config_pangu_cmrc2019.yaml --job_id ***  # ***为job_id，任务拉起成功后生成
```

任务结束后，可在任务对应的`app_config_*.yaml`中指定的`output_path`下查看任务输出结果；在指定的`log_path`下查看任务输出日志， 更多功能接口参数详解请参考微调组件文档 。
