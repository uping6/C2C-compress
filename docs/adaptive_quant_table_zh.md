# 可训练系数量化表模块

`rosetta/model/adaptive_quant_table.py` 参考并本地化了
`/data/smy/AdaptiveJPEG-KV/src/homo_c2c_kv/training/adaptive_jpeg.py` 中的核心设计：

- 沿 KV cache 的序列维执行正交 1D-DCT；
- 对每个 `layer / K-V / KV-head` group 使用 dynamic RMS scale；
- local-global allocator 从离散 `alpha_candidates` 中选择量化表；
- 训练时用 hard Gumbel-Softmax 和 STE 量化；
- 使用 factorized logistic entropy model 提供可微 `Rhat`；
- 总目标为 `task CE + rate_weight * estimated_payload_bits`。

该实现已经复制进本仓库，不会在运行时导入 `/data/smy/AdaptiveJPEG-KV`。
为了适配 LongBench，DCT 使用 FFT 实现，而不是建立长度平方级的 DCT 矩阵。

## 训练

训练采用与 AdaptiveJPEG-KV 相同的两阶段语义：

1. Stage 1（raw projector）：关闭量化器，冻结 base/teacher，仅优化 projector 的任务 CE。
2. Stage 2（adaptive QAT）：从 Stage 1 的 `final/` warm-start projector；量化器与
   optimizer 重新初始化，projector 和量化器联合优化 `task CE + lambda * Rhat`。

AdaptiveJPEG-KV 的 HellaSwag 配置在 Stage 1/2 分别使用 `3e-4`/`1e-4` 学习率、
`weight_decay=0.01`、两轮训练；Stage 2 使用 `lambda=1e-6` 和 1000 optimizer-step
rate warmup。本仓库的 LongBench-E 配置保留这两个学习率、weight decay、lambda 和
1000-step warmup，但每阶段为一轮 LongBench-E（1896 个筛选样本），沿用现有
LongBench-E 的 batch/DDP 设置。

在 Rosetta 训练配置的 `model` 下加入：

```json
"adaptive_quant_table": {
  "enabled": true,
  "feature_bands": 8,
  "hidden_dim": 128,
  "alpha_candidates": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
  "initial_alpha_index": 0,
  "initial_temperature": 1.0,
  "final_temperature": 0.1,
  "anneal_steps": 1896,
  "q_base_min": 1.0,
  "q_base_max": 8.0,
  "q_base_power": 1.0,
  "scale_side_info_bits": 16,
  "rate_weight": 1e-6,
  "rate_warmup_steps": 500
}
```

Stage 1 配置是
`recipe/train_recipe/C2C_longbench_latent_kv_joint_raw_stage1.json`；Stage 2 配置是
`recipe/train_recipe/C2C_longbench_latent_kv_joint_adaptive_quant.json`。可用
`bash/train/train_longbench_e_adaptive_quant_stages.sh` 顺序运行两个阶段。
训练 checkpoint 会额外保存 `adaptive_quant_table.pt`。

## 评测

评测配置也要提供同样的结构参数，并设置 `enabled: true`。评测器默认从
Rosetta checkpoint 目录加载 `adaptive_quant_table.pt`；也可用
`adaptive_quant_table.checkpoint_path` 显式指定。完整样例是
`recipe/eval_recipe/C2C_longbench_latent_kv_joint_adaptive_quant.yaml`。

关闭模块或省略该配置时，训练和评测都保持原有行为。

## 当前边界

该模块作用在 layer route 选出的 sharer/source KV 上，并且位于 projector 之前。
它不读取 projector 输出或 receiver KV，当前提供训练内量化重建、离散表选择和
可微码率代理。重建后的 source KV 再送入 projector，因此任务 CE 仍可训练量化表
allocator 和 projector。`estimated_payload_bits` 不是实际熵编码后的网络字节数。
现有 `cachejpeg_rosetta` 评测仍可在 sharer 原始 KV 上启用独立的 CacheJPEG transport；
若两者同时开启，应分别报告 source-side 实际 payload 与 mapped-KV 的 `Rhat`，不能把
二者当成同一个压缩率。要得到当前 pre-projector source KV 的实际码率，还需要为
`rounded_symbols + table_indices + FP16 scales` 增加真实打包与熵编码协议。
