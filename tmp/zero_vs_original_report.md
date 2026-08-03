# Zero-sharer-cache vs JPEGCache-Rosetta

- Zero summary: `local/final_results/0.6+1.5B_instruct/zero_sharer_cache/cachejpeg_rosetta_longbench_generate_20260728_234851_summary.json`
- Original summary: `local/final_results/0.6+1.5B_instruct/jpegcache_rosetta/cachejpeg_rosetta_longbench_generate_20260729_151636_summary.json`

| 分析组 | 数据集 | Zero | Original | Original−Zero | 主要怀疑因素 |
| --- | --- | ---: | ---: | ---: | --- |
| 单文档QA | Qasper | 31.99% | 30.43% | -1.56% | 证据定位、答案长度 |
| 单文档QA | MultiFieldQA-en | 37.57% | 38.92% | +1.34% | 通用文档理解 |
| 多文档QA | HotpotQA | 29.16% | 28.25% | -0.92% | 多跳推理、文档选择 |
| 多文档QA | 2WikiMQA | 24.67% | 21.75% | -2.93% | 实体关系、多跳推理 |
| 摘要 | GovReport | 13.70% | 12.75% | -0.94% | 输出长度、全局覆盖 |
| 摘要 | MultiNews | 9.64% | 9.92% | +0.28% | 去重、跨文档整合 |
| Few-shot | TREC | 24.64% | 21.74% | -2.90% | 标签格式、示例归纳 |
| Few-shot | TriviaQA | 43.02% | 44.43% | +1.40% | 短答案、知识问答 |
| Few-shot摘要 | SAMSum | 20.22% | 20.22% | -0.00% | 风格模仿、对话摘要 |
| 合成计数 | PassageCount | 1.67% | 0.00% | -1.67% | 全局计数 |
| 合成检索 | PassageRetrieval-en | 44.12% | 39.71% | -4.41% | 精确位置检索 |
| 代码 | LCC | 5.95% | 7.14% | +1.19% | 文件内长依赖 |
| 代码 | RepoBench-P | 4.11% | 4.11% | +0.00% | 跨文件依赖 |

### 分析组加权均分

| 分析组 | Zero | Original | Original−Zero | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| 单文档QA | 34.30% | 33.95% | -0.35% | 82 |
| 多文档QA | 26.97% | 25.07% | -1.90% | 182 |
| 摘要 | 11.85% | 11.47% | -0.39% | 165 |
| Few-shot | 34.68% | 34.13% | -0.55% | 152 |
| Few-shot摘要 | 20.22% | 20.22% | -0.00% | 81 |
| 合成计数 | 1.67% | 0.00% | -1.67% | 60 |
| 合成检索 | 44.12% | 39.71% | -4.41% | 68 |
| 代码 | 5.10% | 5.73% | +0.64% | 157 |
