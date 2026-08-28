# C2C-compress

`C2C-compress` 是一个围绕 Cache-to-Cache / Rosetta 的实验仓库。它的核心目标是让两个大语言模型不通过文本中间结果通信，而是直接在 KV cache 层面传递、投影、融合和压缩信息。

仓库当前同时包含四条主要实验线，其中当前主线是
`concat + LCFProjectedKV + pre-RoPE + Adaptive Quant Table`：

- Rosetta / C2C：把 sharer 模型的 KV cache 投影到 receiver 模型的 KV cache 空间，再由 receiver 继续生成。
- Latent KV：把 KV 融合改成低维 latent bottleneck，支持 `latent_kv_joint` 和 `latent_kv_split`。
- CacheJPEG：对 KV cache 或 latent cache 做编码、量化、熵编码和传输模拟，用于压缩率、延迟和质量评测。
- Adaptive Quant Table：在 LCF transport K/V 上训练可学习量化表，在码率约束下自适应选择量化强度。

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

- 可冻结 base/teacher，训练 projector；QAT 阶段也可联合训练 adaptive quantizer。
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
- `LCFFirstProjector`：joint LCF latent 直接拆成 pseudo K/V。
- `LCFProjectedKVProjector`：joint LCF latent 后使用独立 learned K/V transport projection。
- `AblationProjector`：用于消融实验。

新增 projector 时，需要继承 `Projector`，用 `@register_model` 和 `@capture_init_args` 注册，然后在 recipe 的 `projector_type` 或 fusion 配置中引用。

## 当前主线：concat + LCF + 自适应量化

### 方法边界

当前主线不是把 Sharer cache 加到 Receiver 当前 cache 上，也不经过原来的
fuser。它把 Sharer 信息解码成一段独立的 Receiver KV prefix，再与 Receiver
prompt 在序列维拼接：

```text
Sharer prompt
  -> Sharer prefill，捕获 pre-RoPE K 和普通 V
  -> 每层 LCF shared encoder
  -> learned K/V transport projections
  -> DCT + adaptive quantization + IDCT（训练阶段）
  -> 每层 LCF K/V decoder
  -> Receiver compact RoPE
  -> concat(prefix cache, Receiver prompt cache)
  -> Receiver prefill
  -> Receiver autoregressive decode
```

`cache_alignment: concat` 与 `cache_alignment: fuser` 是平级后端。选择 concat
不会进入 `RosettaFuserBridge`；`fusion_type: original` 在这里仅表示不使用
`latent_kv_joint/split` fuser 分支。

### 单层张量形状

设 Sharer KV 为 `[B,Hs,Ss,Ds]`，Receiver KV 为 `[B,Hr,Sr,Dr]`。当前
`lcf_projected_kv` 默认维度为 shared 128、transport K/V 各 64：

| 阶段 | K/V 或 latent 形状 | 说明 |
|---|---:|---|
| Sharer pre-RoPE K / V | `[B,Hs,Ss,Ds]` | K 在 Sharer RoPE 之前捕获；V 来自正常 cache |
| flatten + concat(K,V) | `[B,Ss,2*Hs*Ds]` | 只合并 head/channel，不改变 token 数 |
| `shared_encoder` | `[B,Ss,128]` | LCF channel bottleneck |
| `key_projection` | `[B,Ss,64]` | learned `128 -> 64` |
| `value_projection` | `[B,Ss,64]` | 与 K 独立的 learned `128 -> 64` |
| CacheJPEG pseudo K/V | 各 `[B,1,Ss,64]` | 增加一个 pseudo KV head，作为传输对象 |
| DCT/量化/IDCT | 各 `[B,1,Ss,64]` | DCT 沿序列轴 `Ss`；量化分配按 layer/KV/head/frequency band |
| `decoder_k/v` | 各 `[B,Hr,Ss,Dr]` | 恢复 Receiver cache geometry |
| compact Receiver RoPE | K `[B,Hr,Ss,Dr]` | prefix 位置为 `0..Ss-1` |
| Receiver prompt | token `[B,Sr]` | `position_ids` 从 `Ss` 开始偏移 |

这里的 LCF 是 channel/head geometry 下采样与上采样，`Ss` 不变；它不是 token
或序列长度下采样。跨层映射由 `layer_mapping: last_aligned` 决定，当前 concat
要求每个 Receiver layer 恰好对应一个 Sharer layer/projector。

### `lcf_first` 与 `lcf_projected_kv`

| 配置 | transport K/V 的产生方式 | 对应类 |
|---|---|---|
| `lcf_first` | joint latent 直接沿最后一维 `chunk(2)` 成 pseudo K/V | `LCFFirstProjector` |
| `lcf_projected_kv` | shared latent 后分别经过 learned K projection 和 V projection | `LCFProjectedKVProjector` |

后者是当前主线。实现入口：

- `rosetta/model/lcf_projected_kv.py`：shared encoder、K/V projection、K/V decoder。
- `rosetta/model/wrapper.py::_forward_concat_lcf_first`：训练期主流程和 rate loss。
- `rosetta/cachejpeg_rosetta/pre_rope.py`：Sharer pre-RoPE K 捕获与 Receiver RoPE。
- `rosetta/cachejpeg_rosetta/projected_kv_cache_aligner.py`：评测期 concat encode/decode。
- `rosetta/cachejpeg_rosetta/concat_layer_streaming.py`：跨层流水化压缩、传输和解码。

### Tokenizer 与 cache 对齐不是一回事

LCF 只处理 Sharer cache geometry，不要求两个 tokenizer 产生相同 token id。但训练
仍必须给 Receiver 和 Sharer 各自提供正确的 `input_ids`、`attention_mask` 和 prompt
边界，因此 concat 前向接收 per-model list，而不是单个共享 tensor。

- 同 tokenizer 或旧的对齐实验可使用 `is_do_alignment: true`。
- Qwen3-0.6B -> Qwen3-4B 主线使用 `is_do_alignment: false` 与
  `independent_tokenizers: true`，两个模型分别 tokenize；Sharer 只读取自己的 prompt。
- `alignment_strategy` 解决 token stream/padding 的组织问题；`layer_mapping`、LCF 和
  concat 解决 cache layer/channel/position 的组织问题，不要混为一谈。

### 两阶段训练

两个阶段都冻结 Sharer 和 Receiver；Stage 1 训练 projector，Stage 2 从 Stage 1
权重初始化后联合训练 projector 与 Adaptive Quant Table。

| 阶段 | 通信路径 | 可训练模块 | 关键字段 |
|---|---|---|---|
| Stage 1 raw projector | LCF encode -> LCF decode | LCF projector | `training_stage: raw_projector`、`adaptive_quant_table.enabled: false` |
| Stage 2 QAT | LCF encode -> adaptive DCT quant -> LCF decode | LCF projector + adaptive quantizer | `training_stage: adaptive_quant_qat`、`initial_projector_checkpoint` |

主线 recipe：

- 0.6B Receiver + 1.5B Sharer Stage 1：`recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_raw.json`
- 0.6B Receiver + 1.5B Sharer Stage 2：`recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant.json`
- 0.6B Receiver + 4B Sharer Stage 1：`recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_raw_qwen3_4b.json`
- 0.6B Receiver + 4B Sharer Stage 2：`recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant_qwen3_4b.json`

不要在 Stage 2 的 `training.freeze` 中加入 `projector`。正确设置是：

```json
"freeze": ["teacher", "base"]
```

`rate_weight` 乘的是估计 payload bits；`rate_warmup_steps` 让码率项逐步进入，
`anneal_steps` 控制离散表选择温度。训练日志中的 `estimated_payload_bits` 是可微码率
估计，不等同于评测时 entropy backend 序列化后的真实 `payload_bytes * 8`。

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

- `wrapper.py`：`CacheJPEGRosettaEvalWrapper`，执行 teacher/sharer prefill、压缩传输、fuser 或 concat、base/receiver generate。
- `fuser_bridge.py`：把加载好的模型、projector 和 latent/adaptive quant 连接起来。
- `layer_streaming.py`：fuser 路径的层级流式压缩和 prefill 计时。
- `concat_layer_streaming.py`：concat 路径的 LCF/codec/transport 跨层流水线。
- `cache_aligner.py`、`projected_kv_cache_aligner.py`：两种 concat LCF 对齐器。
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

concat 路径不使用上面的 latent fusion 类型，而是：

```yaml
cachejpeg_rosetta_config:
  cache_alignment: concat
  fusion_type: original
  concat_projector:
    type: lcf_projected_kv
    shared_latent_dim: 128
    key_latent_dim: 64
    value_latent_dim: 64
  codec:
    method: cachejpeg
    compute: {backend: gpu, transform_dtype: float32}
  layer_streaming:
    enabled: true
    queue_size: 4
    gpu_streams: 2
    max_inflight_layers: 4
```

### 6. Adaptive Quant Table

实现：`rosetta/model/adaptive_quant_table.py`

用途：在训练时学习每层、K/V、pseudo KV head 和频带的量化参数选择，目标是在保持生成质量的同时降低 payload bit rate。当前主线中输入是 LCF 后的 pseudo K/V，而不是原始 Sharer KV。

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

训练完成后会同时保存更新后的 projector 和 adaptive quant table 权重。fuser 路径评测时可以启用：

```yaml
cachejpeg_rosetta_config:
  adaptive_quant_table:
    enabled: true
    checkpoint_path: local/checkpoints/.../final/adaptive_quant_table.pt
```

注意：当前 `rosetta/cachejpeg_rosetta/config.py` 明确禁止在
`cache_alignment: concat` 下启用 fuser-side `adaptive_quant_table`。因此现阶段：

- concat QAT 训练中的 learned adaptive table 已接入训练 loss 和 checkpoint；
- concat CacheJPEG 评测使用 `codec.quant` 与 entropy backend 产生真实码流；
- 不能仅在 concat 评测 YAML 中加入 `adaptive_quant_table.enabled: true`，解析器会报错；
- 若要评测 learned table 本身，需要先补齐 concat aligner 的 table checkpoint 加载与
  DCT/IDCT 推理路径。复盘结果时务必区分“QAT 估计码率”和“CacheJPEG 实际码流”。

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

## 消融实验地图

做消融时应保持模型、数据集、prompt template、`max_new_tokens`、checkpoint 和
随机种子一致，只替换被研究的通信模块。不同 recipe 中目前存在 MMLU、CEval、
OpenBookQA 和 LongBench 路径混用，不能直接横向比较不同数据集的 accuracy。

### 模型贡献基线

| 实验 | 实际执行路径 | 参考 recipe | 回答的问题 |
|---|---|---|---|
| Receiver-only | 只加载/运行 Receiver | `recipe/eval_recipe/receiver_only_qwen3_0.6b_openhermes_mmlu_redux.yaml` | 小模型本身的能力下界 |
| Sharer-only | 只加载/运行 Sharer | `recipe/eval_recipe/sharer_only_qwen2.5_1.5b_instruct_openhermes_mmlu_redux.yaml` | 大模型本身的能力上界/参考 |
| C2C/Rosetta | raw Sharer KV -> projector/fuser -> Receiver | `recipe/eval_recipe/rosetta_qwen3_0.6b_qwen2.5_1.5b_instruct_mmlu_redux.yaml` | 不压缩 cache 通信是否有效 |
| Raw KV + 50 MB/s | raw KV 串行 socket transport -> fuser | `recipe/eval_recipe/rosetta_raw_kv_transport_50mbps_mmlu_redux.yaml` | 无压缩传输的 payload 和延迟基线 |

`receiver_only` 和 `sharer_only` 配置放在 `cachejpeg_rosetta_config.ablation` 中，
但 wrapper 会绕过另一个模型、projector、codec 和 transport；它们不是“传零 cache”。

### CacheJPEG / JPEG-Rosetta

| 实验 | 改变量 | 参考 recipe |
|---|---|---|
| CacheJPEG single | 单模型自身 KV 的压缩与恢复 | `recipe/eval_recipe/cachejpeg_single_eval.yaml`、`recipe/eval_recipe/cachejpeg_longbench_eval.yaml` |
| JPEG-Rosetta | Sharer KV -> CacheJPEG -> fuser -> Receiver | `recipe/eval_recipe/C2C_longbench_0.6+1.5_jpegcache_rosetta_gpu0.yaml` |
| Frequency prune | 不发送指定高频 DCT bands，Receiver 以 0 恢复 | `recipe/eval_recipe/C2C_longbench_0.6+1.5_jpegcache_rosetta_gpu0_prune_b4.yaml` |
| Zero Sharer cache | 完整编解码后、fuser 前把 Sharer K/V 置零 | `recipe/eval_recipe/C2C_longbench_0.6+1.5_jpegcache_rosetta_gpu0_zero_sharer_cache.yaml` |
| Shuffle Sharer cache | Receiver 样本使用另一个样本的 Sharer cache | `recipe/eval_recipe/C2C_longbench_0.6+1.5_jpegcache_rosetta_gpu0_sharer_cache_shuffle.yaml` |
| Layer streaming + 50 MB/s | Sharer 层完成后立即排队压缩/传输/解码 | `recipe/eval_recipe/longbench_jpegcache_stream_50mbps.yaml` |

Zero 与 Shuffle 用来验证提升是否来自与当前问题相关的 Sharer 信息；frequency prune
用于研究频带贡献；single CacheJPEG 用来区分“codec 对单模型 cache 的损伤”和
“跨模型 projector/fusion 的损伤”。

### 对齐、融合与 LCF

| 维度 | 选项 | 配置/实现位置 |
|---|---|---|
| Cache alignment | `fuser` / `concat` | `cachejpeg_rosetta_config.cache_alignment` |
| Fuser representation | `original` / `latent_kv_joint` / `latent_kv_split` | `cachejpeg_rosetta_config.fusion_type` |
| Concat projector | `lcf_first` / `lcf_projected_kv` | `concat_projector.type` |
| Position encoding | post-RoPE legacy / `pre_rope` + Receiver re-RoPE | `rope_mode` 与 `pre_rope.py` |
| Token input | aligned streams / independent tokenizers | `is_do_alignment`、`independent_tokenizers` |
| Quantization | raw / fixed CacheJPEG / learned adaptive QAT | train/eval recipe 的 stage 和 codec 字段 |
| Layer execution | whole-cache sequential / layer streaming | `layer_streaming.enabled` |

推荐用下列链条逐步增加模块，便于归因：

```text
Receiver-only
  -> raw C2C fuser
  -> JPEG-Rosetta fuser
  -> concat + LCF-first raw
  -> concat + LCFProjectedKV raw
  -> concat + LCFProjectedKV + adaptive QAT
  -> concat + LCFProjectedKV + CacheJPEG transport
  -> 上述路径 + layer streaming / 50 MB/s
```

旧 latent KV 消融仍可使用：

- `recipe/eval_recipe/C2C_longbench_latent_kv_joint.yaml`
- `recipe/eval_recipe/C2C_longbench_latent_kv_split.yaml`
- `recipe/eval_recipe/C2C_longbench_latent_kv_split_adaptive_quant.yaml`
- `recipe/eval_recipe/C2C_longbench_latent_kv_split_cachejpeg_zlib_subset200.yaml`
- `recipe/eval_recipe/C2C_longbench_latent_kv_split_cachejpeg_zlib_full.yaml`

`latent_kv_joint/split` 属于 fuser representation 消融，不等同于 concat 下的
`LCFFirstProjector/LCFProjectedKVProjector`。

### C2C projector 内部组件消融

`AblationProjector` 通过 `ablation_level` 控制原 C2C projector 的 gate、scalar
weight 和 target contribution：

| level | 保留内容 | 移除内容 |
|---:|---|---|
| 0 | 完整 C2C | 无 |
| 1 | gate、target | scalar weights |
| 2 | target | scalar weights、gates |
| 3 | source projection | target、scalar weights、gates |
| 4 | scalar weights、target | gates |

训练模板为 `recipe/train_recipe/C2C_ablation.json`，批量入口为
`script/ablation/ablation_study.py`，评测模板为
`recipe/eval_recipe/ablation_base.yaml`。

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

当前 0.6B -> 4B concat + LCFProjectedKV 主线：

```bash
recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_raw_qwen3_4b.json
recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant_qwen3_4b.json
```

0.6B -> 1.5B 对应版本：

```bash
recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_raw.json
recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant.json
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
  --config recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_raw_qwen3_4b.json
```

Stage 1 完成后确认 Stage 2 的 `initial_projector_checkpoint` 指向其 `final/`，再运行：

```bash
python script/train/SFT_train.py \
  --config recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant_qwen3_4b.json
```

断点恢复会同时读取 projector、adaptive table、optimizer、scheduler 和 step：

```bash
python script/train/SFT_train.py \
  --config recipe/train_recipe/cache-kv/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant_qwen3_4b.json \
  --resume_from_checkpoint local/checkpoints/.../checkpoint-1000
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

concat + LCFProjectedKV + CacheJPEG：

```bash
recipe/eval_recipe/cachejpeg_rosetta_openhermes_lcf_projected_kv_mmlu.yaml
```

该文件当前默认指向 0.6B + 1.5B raw projector checkpoint；切换模型规模或 Stage 2
checkpoint 时，需要同步修改 `base_model`、`teacher_model`、`checkpoints_dir` 和
`rosetta_checkpoint_subfolder`，不能只替换一个权重路径。

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

### Step 6：记录实验身份

每次结果至少同时记录以下字段，否则后续很难判断两个目录是否可比较：

```text
receiver / sharer model id
dataset + local_jsonl_file + sample_interval
checkpoint directory + checkpoint subfolder
cache_alignment + fusion_type + concat_projector.type
token alignment / independent tokenizer mode
codec compute + quant + entropy + frequency prune
transport mode + bandwidth + layer_streaming
generation max_new_tokens + template + CoT
git commit / dirty diff
```

## 结果指标定义

统一评测会把逐样本字段写入 `*_length.json`，并聚合到 `*_summary.json`、
`*_performance.json`。主线复盘常用定义如下：

| 指标 | 定义 |
|---|---|
| accuracy / score | 任务答案正确率；LongBench 为任务 scorer 的归一化 score，不一定是选择题 accuracy |
| `sharer_cache_bytes` | 原始 Sharer K/V tensor 的元素数乘 dtype bytes，未包含 Python/pickle 元数据 |
| `lcf_latent_kv_bytes` | LCF 后、codec 前 pseudo K/V tensor 的原始字节数 |
| `payload_bytes` | codec 输出经过真实序列化后的 wire payload 大小 |
| compression ratio | `sum(sharer_cache_bytes) / sum(payload_bytes)`；越大表示压缩越强 |
| payload-to-sharer ratio | `sum(payload_bytes) / sum(sharer_cache_bytes)` |
| space saving ratio | `1 - payload / sharer` |
| `encode_seconds` | concat 中仅指 CacheJPEG encode；不包含 LCF encode |
| `decode_seconds` | concat 中仅指 CacheJPEG decode；不包含 LCF decode |
| `sender_encode_seconds` | `lcf_encode_seconds + encode_seconds` |
| `receiver_decode_seconds` | `decode_seconds + lcf_decode_seconds` |
| `transmit_seconds` | transport 实际测得的发送阶段时间，包含所选 transport 实现的等待 |
| `bandwidth_only_transmit_seconds` | `payload_bytes / bandwidth_bytes_per_sec`，不含固定延迟和序列化 |
| end-to-end latency | evaluator 包围整个单样本 generate 调用的 wall time，包括模型 prefill、通信管线和生成 |

聚合压缩率应优先使用总字节比
`aggregate_sharer_to_payload_compression_ratio`，而不是简单平均每条样本的 ratio；
前者会正确按样本 payload 大小加权。`avg_encode_ms/avg_decode_ms` 是 codec wall
time；分析完整 sender/receiver 开销时应使用 `avg_sender_encode_ms` 和
`avg_receiver_decode_ms`。

layer streaming 还会输出 service time 与 wall time：service time 是各层任务耗时之
和，wall time 是流水线关键路径，两者不能相加当作端到端时间。50 MB/s 配置使用十进制
`50,000,000 bytes/s`。

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

Cache 对齐默认仍使用现有 fuser。若要把 Sharer cache 作为 Receiver 的独立
causal prefix，可选择平级的 concat 对齐后端：

```yaml
cachejpeg_rosetta_config:
  cache_alignment: concat
  fusion_type: original
  concat_projector:
    type: lcf_projected_kv
    shared_latent_dim: 128
    key_latent_dim: 64
    value_latent_dim: 64
  codec:
    method: cachejpeg
    compute: {backend: gpu, transform_dtype: float32}
    transport:
      mode: socketpair
      bandwidth_bytes_per_sec: 50000000
  layer_streaming:
    enabled: true
    queue_size: 4
    gpu_streams: 2
    max_inflight_layers: 4
```

concat 支持 `lcf_first` 与 `lcf_projected_kv` 两类 checkpoint。两者都在 Sharer
侧捕获 pre-RoPE K，CacheJPEG 只编解码 pseudo/latent K/V，Receiver 端恢复
Receiver KV geometry 后重新施加 compact RoPE。`lcf_projected_kv` 在 shared
latent 后有独立 learned K/V projection，是当前主线。concat 不调用
`RosettaFuserBridge`，但已经支持 `concat_layer_streaming`；它仍不接受
fuser-side `adaptive_quant_table` eval 配置。

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
pytest test/test_concat_cache_alignment.py
pytest test/test_lcf_projected_kv.py
pytest test/test_independent_dual_tokenizer_dataset.py
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

### 6. concat checkpoint 类型不匹配

`concat_projector.type: lcf_first` 只能加载 `LCFFirstProjector`，
`lcf_projected_kv` 只能加载 `LCFProjectedKVProjector`。YAML 的类型、训练 recipe
和 checkpoint 内的 `projector_*.json` 必须一致。

### 7. Stage 2 为什么没有更新 projector

Stage 2 当前设计为 projector 与 adaptive quantizer 联合训练。检查：

```json
"freeze": ["teacher", "base"]
```

如果加入 `"projector"`，就会退化为只训练量化器的实验，不再是当前主线。

### 8. concat 是否必须对齐 tokenizer

不要求 token id 一致，但要求 forward 收到两个模型各自的输入 list。不同 tokenizer
优先设置 `is_do_alignment: false` 和 `independent_tokenizers: true`；旧 aligned
路径则设置 `is_do_alignment: true`。错误地只返回一个 tensor 会触发
“requires per-model input_ids and attention_mask lists”。

## 开发建议

- 改训练逻辑优先看 `script/train/SFT_train.py` 和 `rosetta/train/dataset_adapters.py`。
- 改评测逻辑优先看 `script/evaluation/unified_evaluator.py`。
- 改 cache 融合优先看 `rosetta/model/wrapper.py`、`rosetta/model/latent_kv.py` 和 `rosetta/cachejpeg_rosetta/fuser_bridge.py`。
- 改压缩传输优先看 `rosetta/cachejpeg/` 和 `rosetta/cachejpeg_rosetta/`。
- 新增实验时尽量新增 recipe，不要把路径和超参硬编码进 Python。
