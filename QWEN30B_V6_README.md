# Qwen30B + SAM Audio：四卡 V6 可移植调试包

这是给协作者调试/迁移的完整依赖补充版，替代旧49文件源码包。包含当前联合训练模块、实际引用的加速实现、SAM修改源码、四个C++扩展源码、检查点/验证实现、模型下载校验工具、CPU准备工具和单节点四卡Slurm入口。数据和模型权重不打包进GitHub。

**状态：本包完成静态依赖、语法、源码哈希检查，以及集群CPU环境17个关键运行模块的实际导入（6923689）；接收方环境安装、真实四卡训练/恢复、817条验证仍未通过本包端到端验收。不能承诺下载就跑通或四卡0.8秒。** 原研究单A100/global2/固定64输入的V6均值约0.809299秒，非四卡global8速度。

## 1. 环境与模型（CPU/传输节点）

解压后进入本目录，先读 `ENVIRONMENT.md`，建立Python3.12环境并安装匹配的CUDA PyTorch与列出的外部包。使用自己的HF账号申请SAM模型访问并登录；不要发送或提交token。

```bash
python verify_package.py
python download_models.py --model-root /YOUR_SHARED_STORAGE/models
```

Qwen下载先将请求的revision解析成不可变commit，再下载并核对本任务记录的模型文件SHA256。默认main如已变化导致哈希不同会停止；用 `--qwen-revision COMMIT` 指定匹配版本。SAM固定官方revision。绝不自动接受另一套模型来掩盖缓存不匹配。

## 2. 数据（HF上传进行中）

需111068训练行、817验证行、最终prompt/GT/split映射、Qwen processor与冻结AV缓存、SAM mixture/video/clean缓存，及验证原mixture/GT音频。缓存验证路径不需要原始视频。模型语言部分继续训练，并非缓存语言隐藏状态。

数据仓库：TerryShenty/qwen30b-sam-joint-111k。仅当存在UPLOAD_COMPLETE.json才是完整发布。使用 `python download_data.py --destination /YOUR_SHARED_STORAGE/joint_data` 下载校验解包。

数据目录应带：`PROCESSOR_COMPLETE.json`、`QWEN_AV_COMPLETE.json`、`MEDIA.json`、`BOUND_ROWS.json`、`TRAIN_SPANS.json`、`base_identity.json`、`TARGET_DEMANDING_CASES_CPU.json`、`VALIDATION_PROTOCOL_BINDING.json`与`qwen_cache/`。这些都属于HF数据包，不是本GitHub包里的伪造样本。

**迁移数据时必须保留缓存逻辑身份。** `base_identity.json`的原始键、MEDIA中参与cache key的历史路径是逻辑身份，不能全局替换。实际验证音频路径需要单独重定位，验证源码绑定要重新计算本地路径和SHA。目标机器应重新生成 `PRODUCTION_INPUT_ADMISSION.json`，原机器的inode/mtime凭据不可复用。`prepare.py`遇到旧凭据不匹配会停止，不能把失败改成通过。

## 3. 本地绑定和CPU编译

将`paths.example.json`复制为自己的配置，填data_root、sam_namespace、qwen_model、sam_model四个目录。

```bash
python prepare.py --paths my_paths.json
```

该步骤核对模型内容、完整数据元数据、配置和源依赖，编译四个ATen扩展，写本机native哈希和CPU_READY。不会提交作业、不读取研究checkpoint、不在GPU上编译。首次安装还需在目标环境确认完整模块导入与FFmpeg解码，CPU_READY不是GPU通过证明。

## 4. 单节点四卡检查（无测速搜索）

```bash
export QWEN_PACKAGE="$PWD"
export QWEN_PYTHON=/YOUR_ENV/bin/python
export QWEN_OUTPUT=/YOUR_SHARED_STORAGE/runs/qwen-check-001
export QWEN_MODE=qualify
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION train.sbatch
```

按集群政策修改GPU类型、CPU/内存/时间；模板为4GPU/16CPU/128GiB/1小时/no-requeue。只有一次Slurm资源请求，内部一个torchrun启动四个常驻rank，batch2/rank、global8。无逐样本作业/数组、无自动续租、无候选测速比较。检查最长输入、真实更新、完整保存/恢复及继续更新后才会生成`PORTABLE_GATE.json`。需要Slurm设置SLURM_JOB_END_TIME和有限cgroup v2内存；不支持的平台会明确失败。

## 5. 训练与恢复

```bash
export QWEN_MODE=train
export QWEN_OUTPUT=/YOUR_SHARED_STORAGE/runs/qwen-train-001
export QWEN_GATE=/YOUR_SHARED_STORAGE/runs/qwen-check-001/PORTABLE_GATE.json
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --time=24:00:00 train.sbatch
```

训练入口是`production_v6_attention_backward/portable_train.py`，没有epoch上限，按原Slurm结束时间预留保存。默认每900秒完整保存；600秒保存预留和120秒step保护是保守初值，不是新平台实测保证。先在短任务测量后调整并重新封存。每个rank保存模型/优化器/调度器/EMA/RNG/采样位置/config/identity。用新output加`QWEN_RESUME=/path/checkpoints/step-.../COMPLETE.json`恢复同一部署的正式state；不能用资格检查/研究state替代正式初始化。新进程冷编译后的前三次更新为真实训练且计入数据位置。

## 6. 验证边界

完整验证代码为`validation_run.py`、`joint_validation.py`及`validation_reference/{music,speech}`。生成使用缓存，评分需要原始GT与mixture音频，固定原cohort/noise、midpoint16和live权重。`validation_run.run`要求真实训练状态下最长padding/codec/scoring的四rank显存批准记录，再允许batch1/2/4/8。

**目前训练CLI不自动触发817验证，也不伪造显存批准。** 协作者还需在目标设备完成codec装载、最长样本显存检查并调用该验证接口；无验证结果时TRAIN_COMPLETE明确记录`full817_validation_completed=false`。数据HF发布和这一目标GPU集成是剩余工作，不应把源码交付当实验完成。

## 文件与修改

`DEPENDENCY_INVENTORY.json`记录源码依赖选择；保留的旧版本命名模块是V6实际调用/绑定的代码，不是要重跑旧实验。没有附带研究历史目录、任务队列、训练产物、token或其他用户数据。源码内历史校验被明确重绑为本导出副本的哈希；原始接受合同SHA保留作出处，导出不继承原GPU验收。训练实际优化内核保持选定V6。不得使用旧仓库根目录SOLOs quickstart启动本Qwen任务。

本包不会修改原集群排队作业。后续debug代码改动需重新生成SOURCE_MANIFEST并重新做受影响检查；不要关闭哈希和恢复检查来绕过问题。

## V6更新与限制
新增attention输入梯度的动态INT8近似及转置INT8权重缓存，单A100/global2两full32均值0.809298911秒，配对快1.64726%。这不是四卡速度或质量等价保证。保留V5独立包作为fallback。CPU准备现在自动绑定验证评分源码和验证音频实际路径，保持原缓存逻辑身份不变。验证尚未自动集成到训练CLI；目标GPU显存批准及完整817推理仍待完成。
