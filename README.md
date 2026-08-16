# C2C-compress

`C2C-compress` 是一个围绕 Cache-to-Cache / Rosetta 的实验仓库。它的核心目标是让两个大语言模型不通过文本中间结果通信，而是直接在 KV cache 层面传递、投影、融合和压缩信息。

仓库当前同时包含四条主要实验线：

- Rosetta / C2C：把 sharer 模型的 KV cache 投影到 receiver 模型的 KV cache 空间，再由 receiver 继续生成。
- Latent KV：把 KV 融合改成低维 latent bottleneck，支持 `latent_kv_joint` 和 `latent_kv_split`。
- CacheJPEG：对 KV cache 或 latent cache 做编码、量化、熵编码和传输模拟，用于压缩率、延迟和质量评测。
- Adaptive Quant Table：训练可学习量化表，在码率约束下自适应选择量化强度。

## 目录结构

```text
.
├── rosetta/                    # 主 Python 包
│   ├── model/                  # Rosetta wrapper、projector、latent KV、自适应量化
│   ├── cachejpeg/              # 单模型 KV cache CacheJPEG 评测封装
│   ├── cachejpeg_rosetta/      # CacheJPEG + Rosetta 联合评测封装
│   ├── train/                  # 数据集适配、collator、训练辅助函数
│   ├── baseline/               # 两阶段文本通信 baseline
│   └── utils/                  # 模型加载、答案抽取、评测辅助、registry
├── script/
│   ├── train/                  # 训练入口
│   ├── evaluation/             # 统一评测入口
│   ├── dataset/                # 数据集构建脚本
│   ├── playground/             # demo、chat、推理样例
│   ├── analysis/               # 结果分析、缩放实验、Venn、t-SNE、长度统计
│   ├── consistency/            # Rosetta 与 LLM 标签一致性检查
│   └── ablation/               # 消融实验入口
├── recipe/
│   ├── train_recipe/           # 训练 JSON 配置
│   └── eval_recipe/            # 评测 YAML 配置
├── bash/                       # 常用训练、评测 shell 脚本
├── longbench/config/           # LongBench prompt 和 max length 配置
├── test/                       # pytest 单元/集成测试
├── environment.yml             # conda 环境
└── pyproject.toml              # Python 包配置
```

## 核心概念

### Sharer 与 Receiver

仓库里的配置通常把两个模型称为：

- `teacher_model` / sharer：提供额外语义信息，负责产生可被传输或融合的 KV cache。
- `base_model` / receiver：接收 sharer 的信息，融合后继续生成最终答案。

### RosettaModel

`rosetta/model/wrapper.py` 中的 `RosettaModel` 是主要封装。它持有多个 Hugging Face causal LM 和一组 projector，通过 `kv_cache_index` 控制哪些 token 位置需要执行 cache-to-cache 投影。

主要能力：

- 冻结 base/teacher，只训练 projector。
- 支持单 sharer 和多 sharer。
- 支持 `sequential` 与 `parallel` 多源融合。
- 支持 `include_response`，可以选择是否把 response 段也纳入训练/融合。
- 支持把 projector 配置保存为 `projector_config.json`，权重保存为 `projector_*.pt`。

### Projector

`rosetta/model/projector.py` 定义 projector registry 和实现。

当前重点 projector：

- `AllInOneProjector`：通用 C2C projector，支持 gate、weight、concat、SwiGLU、residual、token/head/value 粒度控制。
- `C2CProjector`：兼容早期 C2C 用法的 projector 类型。
- `LatentKVCompressor`：在 `rosetta/model/latent_kv.py` 中注册，用于 latent KV 融合。
- `AblationProjector`：用于消融实验。

新增 projector 时，需要继承 `Projector`，用 `@register_model` 和 `@capture_init_args` 注册，然后在 recipe 的 `projector_type` 或 fusion 配置中引用。

## 功能总览

### 1. C2C / Rosetta 训练

入口：`script/train/SFT_train.py`

支持模式：

- Rosetta projector 训练：配置中提供 `base_model` 和 `teacher_model`。
- baseline 训练：配置中提供 `baseline_model`。
- LoRA baseline：通过 `lora_config` 启用 PEFT LoRA。
- partial training：按层或参数比例解冻部分模型参数。
- 分布式训练：支持 `torchrun` + DDP。
- 断点/阶段训练：支持从 `initial_projector_checkpoint` 加载已有 projector。
- adaptive quant QAT：训练 projector 的同时训练 `AdaptiveCoefficientQuantizer`。

常用命令：

```bash
python script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_latent_kv_split.json
```

多 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_latent_kv_split_adaptive_quant.json
```

现有阶段训练脚本：

```bash
bash bash/train/train_longbench_e_adaptive_quant_stages.sh
```

### 2. 统一评测

入口：`script/evaluation/unified_evaluator.py`

支持模型类型：

- `Rosetta`：普通 C2C/Rosetta projector 评测。
- `cachejpeg`：单模型 KV cache 先压缩传输再解码生成。
- `cachejpeg_rosetta`：sharer cache 或 latent cache 经 CacheJPEG 压缩后，再融合到 receiver。
- `two_stage`：文本两阶段 baseline，先生成背景/上下文，再回答。
- `two_stage_rosetta`：两阶段文本上下文 + Rosetta 组合。

支持数据集：

- `mmlu-redux`
- `mmmlu`
- `gpqa`
- `math-500`
- `longbench`
- `gsm8k`
- `openbookqa`
- `ai2-arc`
- `mmlu-pro`
- `ceval`

常用命令：

```bash
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/unified_eval.yaml
```

LongBench-E 子集评测示例：

```bash
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/C2C_longbench_latent_kv_split_cachejpeg_zlib_subset200.yaml
```

### 3. LongBench / LongBench-E 流程

LongBench prompt 与最大生成长度配置位于：

- `longbench/config/dataset2prompt.json`
- `longbench/config/dataset2maxlen.json`

评测配置中常用字段：

```yaml
eval:
  dataset: longbench
  longbench_e: true
  longbench_e_test_subset:
    enabled: true
    size: 200
    seed: 42
  longbench_local_data_dir: /path/to/LongBench
  sample_interval: 1
```

训练数据中可通过 `_id` hash 做划分。当前 `LongBenchChatDataset` 支持 `filter_mod4`，评测端的 LongBench-E held-out subset 会选择互补划分，避免训练/评测混用。

### 4. CacheJPEG

单模型路径在 `rosetta/cachejpeg/`：

- `wrapper.py`：`CacheJPEGEvalWrapper`，执行 prefill -> encode -> transport -> decode -> generate。
- `config.py`：CacheJPEG 配置解析。
- `transport.py`：传输模拟，包括 direct、带宽限制等模式。
- `gpu_codec.py`：GPU 侧变换/编码封装。
- `entropy_backends.py`：安装 HomoC2C-KV 所需 entropy backend。

注意：CacheJPEG 依赖外部 HomoC2C-KV 源码，默认路径来自配置项：

```yaml
cachejpeg_config:
  homo_c2c_kv_src: /data/smy/HomoC2C-KV/src
```

如果路径不存在，CacheJPEG 相关评测会失败。

### 5. CacheJPEG + Rosetta

联合路径在 `rosetta/cachejpeg_rosetta/`：

- `wrapper.py`：`CacheJPEGRosettaEvalWrapper`，执行 teacher/sharer prefill、压缩传输、Rosetta fuser、base/receiver generate。
- `fuser_bridge.py`：把加载好的模型、projector 和 latent/adaptive quant 连接起来。
- `layer_streaming.py`：层级流式压缩和 prefill 计时。
- `config.py`：联合配置解析。

支持 fusion 类型：

```yaml
cachejpeg_rosetta_config:
  fusion_type: original          # 原始 projector 融合
  # fusion_type: latent_kv_joint # joint latent bottleneck
  # fusion_type: latent_kv_split # 传输 split latent payload
```

`latent_kv_split` 可单独压缩 latent payload：

```yaml
cachejpeg_rosetta_config:
  fusion_type: latent_kv_split
  split_latent_cachejpeg:
    enabled: true
    codec:
      entropy:
        representation: dense_int16
        backend: zlib1
      compute:
        backend: gpu
```

### 6. Adaptive Quant Table

实现：`rosetta/model/adaptive_quant_table.py`

用途：在训练时学习每层、每个 KV head 或频带的量化参数选择，目标是在保持生成质量的同时降低 payload bit rate。

典型配置：

```json
"adaptive_quant_table": {
  "enabled": true,
  "feature_bands": 8,
  "hidden_dim": 128,
  "alpha_candidates": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
  "initial_temperature": 1.0,
  "final_temperature": 0.1,
  "anneal_steps": 1896,
  "rate_weight": 1e-6,
  "rate_warmup_steps": 1000
}
```

训练完成后会保存 adaptive quant table 权重。评测时如果启用：

```yaml
cachejpeg_rosetta_config:
  adaptive_quant_table:
    enabled: true
    checkpoint_path: local/checkpoints/.../final/adaptive_quant_table.pt
```

### 7. 数据集适配

实现：`rosetta/train/dataset_adapters.py`

已注册数据集：

- `LongBenchChatDataset`
- `MMLUChatDataset`
- `MMLUCotChatDataset`
- `LLMGeneratedChatDataset`
- `OpenBookChatDataset`
- `OpenHermesChatDataset`
- `ChatDataset`
- `AlignedChatDataset`
- `BaselineChatDataset`

创建方式来自训练配置：

```json
"data": {
  "type": "LongBenchChatDataset",
  "kwargs": {
    "use_longbench_e": true,
    "local_data_dir": "/path/to/LongBench/data",
    "filter_mod4": true,
    "max_length": 8192,
    "num_samples": 1896
  }
}
```

### 8. Demo 与交互

推理样例：

```bash
python script/playground/inference_example.py
```

单轮采样：

```bash
python script/playground/sample_response.py
```

交互聊天：

```bash
python script/playground/live_chat_example.py \
  --checkpoint_dir local/checkpoints/your_checkpoint/final
```

Gradio demo：

```bash
python script/playground/gradio_demo.py
```

### 9. 分析与消融

常用分析脚本：

- `script/analysis/scaling/`：checkpoint 批量评测、T2T scaling、自动 scaling 实验。
- `script/analysis/proportion/`：KV cache 占比和性能评估。
- `script/analysis/length_ratio/`：生成长度与准确率统计。
- `script/analysis/gate_weight/`：收集 projector gate/weight。
- `script/analysis/venn/`：多模型答题交集 Venn 分析。
- `script/analysis/tsne/`：KV cache 表示 t-SNE 可视化。
- `script/consistency/`：检查 Rosetta 与 LLM 标签一致性。
- `script/ablation/ablation_study.py`：统一消融实验入口。

## 安装环境

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate rosetta
pip install -e ".[training,evaluation,dev]"
```

最小安装：

```bash
pip install -e .
```

主要依赖版本来自 `pyproject.toml`：

- Python `>=3.10`
- `torch==2.6.0`
- `transformers==4.52.4`
- `lz4>=4.3`

训练和评测还会用到 `datasets`、`accelerate`、`wandb`、`peft`、`jsonlines`、`math_verify` 等。

## 端到端实验流程

### Step 1：准备模型与数据

可以直接使用 Hugging Face 模型名，也可以在配置中指定本地路径：

```json
"base_model": "Qwen/Qwen3-0.6B",
"base_model_local_dir": null,
"teacher_model": "Qwen/Qwen2.5-1.5B-Instruct",
"teacher_model_local_dir": "/path/to/Qwen2.5-1.5B-Instruct"
```

离线运行时设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

仓库默认会把 `HF_ENDPOINT` 设置为 `https://hf-mirror.com`。如需代理，可设置：

```bash
export ROSETTA_HTTP_PROXY=http://host:port
export ROSETTA_HTTPS_PROXY=http://host:port
```

### Step 2：选择训练 recipe

普通 Rosetta：

```bash
recipe/train_recipe/C2C_longbench_latent_kv_split.json
```

先训练 raw latent KV，再做 adaptive quant QAT：

```bash
recipe/train_recipe/C2C_longbench_latent_kv_split_raw_stage1.json
recipe/train_recipe/C2C_longbench_latent_kv_split_adaptive_quant.json
```

baseline：

```bash
recipe/train_recipe/baseline_config.json
recipe/train_recipe/baseline_lora_config.json
recipe/train_recipe/baseline_partial_config.json
```

### Step 3：启动训练

```bash
python script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_latent_kv_split_raw_stage1.json
```

输出目录由配置控制：

```json
"output": {
  "output_dir": "local/checkpoints/...",
  "save_steps": 120,
  "eval_steps": 3000
}
```

典型 checkpoint 内容：

```text
final/
├── projector_0.json
├── projector_0.pt
├── projector_config.json
└── adaptive_quant_table.pt   # 启用 adaptive quant 时存在
```

### Step 4：选择评测 recipe

普通 Rosetta：

```bash
recipe/eval_recipe/C2C_longbench_latent_kv_split.yaml
```

CacheJPEG + Rosetta：

```bash
recipe/eval_recipe/C2C_longbench_latent_kv_split_cachejpeg_zlib_subset200.yaml
recipe/eval_recipe/C2C_longbench_latent_kv_split_cachejpeg_zlib_full.yaml
```

单模型 CacheJPEG：

```bash
recipe/eval_recipe/cachejpeg_longbench_eval.yaml
recipe/eval_recipe/cachejpeg_single_eval.yaml
```

### Step 5：启动评测

```bash
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/C2C_longbench_latent_kv_split_cachejpeg_zlib_subset200.yaml
```

输出通常包括：

- 每个 subject 的预测文件。
- 汇总指标 JSON/CSV。
- CacheJPEG payload bytes、compression factor、transport latency 等统计。
- bad sample dump，便于排查失败样本。

## 重要配置字段

### `model`

```yaml
model:
  model_name: cachejpeg_rosetta
  max_length: 12000
  rosetta_config:
    base_model: Qwen/Qwen3-0.6B
    teacher_model: Qwen/Qwen2.5-1.5B-Instruct
    checkpoints_dir: local/checkpoints/...
    is_do_alignment: false
    alignment_strategy: longest
  generation_config:
    do_sample: false
    max_new_tokens: 2048
```

### `cachejpeg_rosetta_config`

```yaml
cachejpeg_rosetta_config:
  fusion_type: latent_kv_split
  latent_kv_bridge:
    enabled: true
    latent_dim: 128
    layer_mapping: proportional
  sharer_model_role: teacher
  receiver_model_role: base
  transport:
    mode: direct
```

### `eval`

```yaml
eval:
  dataset: longbench
  gpu_ids: [0]
  answer_method: generate
  use_cot: false
  use_template: true
  sample_interval: 1
```

## 测试

运行全部测试：

```bash
pytest
```

按模块运行：

```bash
pytest test/test_latent_kv.py
pytest test/test_cachejpeg_rosetta_wrapper.py
pytest test/test_adaptive_quant_table.py
```

`pyproject.toml` 默认启用 coverage 输出。如果只想快速检查某个文件，可以指定单个测试文件。

## 常见问题

### 1. CacheJPEG 提示找不到 HomoC2C-KV

检查配置中的：

```yaml
homo_c2c_kv_src: /path/to/HomoC2C-KV/src
```

该路径必须存在，并且能导入 `homo_c2c_kv.cache.interop` 和 `homo_c2c_kv.codec.cachejpeg.codec`。

### 2. 本地模型路径和 HF 模型名哪个优先

评测器会优先使用存在的本地路径字段，例如：

- `base_model_path`
- `base_model_local_dir`
- `teacher_model_path`
- `teacher_model_local_dir`

如果本地路径不存在，则回退到 Hugging Face model id。

### 3. `latent_kv_split` 和 `layer_streaming` 不能同时开

当前实现中 `latent_kv_split` 不支持 `layer_streaming.enabled=true`，配置解析会直接报错。

### 4. `split_latent_cachejpeg` 为什么要求 zlib

`split_latent_cachejpeg.enabled=true` 时，配置解析要求 entropy backend 以 `zlib` 开头，例如 `zlib1`。

### 5. README 里没有列出的临时输出

仓库中 `tmp/`、`local/`、`LongBench/` 通常是数据、checkpoint 或实验产物，不是核心源码。阅读和修改源码时建议先排除这些目录。

## 开发建议

- 改训练逻辑优先看 `script/train/SFT_train.py` 和 `rosetta/train/dataset_adapters.py`。
- 改评测逻辑优先看 `script/evaluation/unified_evaluator.py`。
- 改 cache 融合优先看 `rosetta/model/wrapper.py`、`rosetta/model/latent_kv.py` 和 `rosetta/cachejpeg_rosetta/fuser_bridge.py`。
- 改压缩传输优先看 `rosetta/cachejpeg/` 和 `rosetta/cachejpeg_rosetta/`。
- 新增实验时尽量新增 recipe，不要把路径和超参硬编码进 Python。
