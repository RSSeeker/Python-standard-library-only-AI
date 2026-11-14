# 同步状态：Python → C / Web

本文件记录 Python 主仓库（source of truth）能力在 C 引擎（`D:\vscode\AI_C`）与 Web 页面
（`D:\vscode\RSSeeker.github.io`）两个副仓库中的同步状态，以及尚未同步的清单和后续批次建议。

> 记录时间：2026-11-13
> Python：HEAD `2d0e18c`，2597 tests（`pytest test_ai.py` 全绿）
> C：HEAD `edba066`，`make check` 两次 EXIT=0、58 条 compare 汇总、885 条 OK、失败 0
> Web：HEAD `09faa7a`，`node tools/check.js` ALL PASS (303 checks)

## 1. 已同步到 C 的能力

C 仓库通过 `c/tools/*_reference.py` 从 Python `AI.py` 导出参考 JSON，
`c/tests/compare_*.c` 逐位/容差比对，接入 `c/Makefile` 的 `check` 目标。
下表按能力域列出（对拍数取最近一次 `make check` 的汇总）。

| 能力域 | 涵盖内容 | 对拍（checks） |
| --- | --- | --- |
| 基础组件 | Layer / MLP / Conv2d / FFN / attention / 归一化 / 激活 / 损失 / 优化器 / 调度器 / 文本与分词工具 | 逐位 |
| 指标族 | 文本 METEOR 完整版 / chrF / WER / CER；音频 STOI / PESQ；视觉 LPIPS / FVD；CLIP 双塔 | 逐位 |
| 解码与导出 | CTC 前缀 beam search；MLP → ONNX 导出/解析/执行；Medusa / EAGLE 投机解码（推理） | 逐位 |
| 训练与 RL | IQL / CQL 离线 RL；EGNN；EDM / DPM-Solver / Consistency 扩散采样；Flow Matching | 逐位 |
| 生成模型 | Pix2Pix / CycleGAN / StyleGAN / Progressive 构件与训练步；GAN 辅助与 R1/R2 罚项 | 逐位 |
| 训练技术 | FSDP / ZeRO / 张量并行 / 序列并行数值模拟；fp16/bf16/fp8 舍入与对称伪量化 | 逐位 |
| 训练辅助 | NF4 量化/双重量化；WeightStandardization；GradScaler；MoE 负载均衡损失 | 逐位 |
| 归一化/位置扩展 | QK-Norm / DynamicTanh；ABF / LongRoPE / 2D / 3D RoPE | QK-Norm 531/531；RoPE 2324/2324 |
| 线性注意力 | naive / lightning / GLA / DeltaNet / Hyena 因果长卷积 | 610 项 / 601 逐位（max 4.441e-16） |
| RWKV time-mixing | RWKV-5（逐位）/ RWKV-6、7（≤1 ULP，1e-12 容差） | 514 项 / 511 逐位 |
| 模型合并 | soup / SLERP / TIES / DARE | 266 项 / 262 逐位 |
| 训练目标 | FIM（PSM/SPM）与 T5 span corruption | 752/752 逐位 |
| RL / μP 工具 | GRPO 组相对优势、Best-of-N、拒绝采样、自洽投票；μP target_std / 分层 lr / MuPAdam | 逐位 |
| 可解释性 | SAE（重建 + L1、手写反向、激活统计） | 逐位 |
| 服务与推理 | 树状注意力；PagedAttention 块分配器（分配/共享/COW/swap/统计） | 逐位 |
| 幻觉检测 | 余弦/句向量相似度、NLI 比例、token 级支撑检测 | 逐位 |
| 语音 | TextToMel、Griffin-Lim、Mel 反演 | 逐位 |
| KAN | B 样条边函数前向 + 手写反向 | 267/267 |
| 模型类（前向） | LLaMA、BERT、ViT、ELECTRA、RWKV、RetNet、xLSTM、Mamba、DeBERTa | 逐位 |

### 近期同步提交

| 模块 | C 提交 | 说明 |
| --- | --- | --- |
| FIM / span corruption | `abbfcc1` | C 侧要求显式 seed；`sentinel_id == -1` 表示 Python 的 `None` |
| QK-Norm / DynamicTanh | `c8f8b42` | 扁平入参下无法表达行长不一致等结构校验 |
| RoPE 扩展（ABF/LongRoPE/2D/3D） | `daa8178` | 含 `.PRECIOUS` 修复 |
| 线性注意力 / 长卷积 | `de8ddf1` | pow 的 libm 差按 1e-12 容差 |
| RWKV-5/6/7 time-mixing | `270c9c8` | `exp(-exp(w))` ≤1 ULP，1e-12 容差 |
| 模型合并 | `edba066` | SLERP 的 sqrt/acos/sin ≤1 ULP；DARE 要求显式 seed |
| 参考导出确定性 | `8c4399e` | 5 个参考脚本固定 `random.seed(20261113)`，消除 `make check` 抖动 |

## 2. 未同步到 C 的清单

状态标记：
- **缺失**：C 仓库 `c/*.h` 中没有对应实现（用 grep 核对过头文件名/常见标识符）；
- **部分**：只有构件/结构体，没有可对拍的前向/反向或训练路径；
- **仅 Python 设计**：依赖 Python 对象、文件系统、回调或全局随机，C 扁平输入无法等价表达。

### 2.1 模型级（有前向/反向/生成，可直接对拍）

| 能力 | 状态 | 备注 |
| --- | --- | --- |
| T5 / Seq2SeqLM（encoder-decoder、交叉注意力） | 缺失 | C README 已声明「需要解码器/交叉注意力的模型类」未包含；仅 `fim.h` 里出现 T5 span 相关名字 |
| DETR（检测 Transformer） | 缺失 | 同上 |
| Swin Transformer / ConvNeXt | 缺失 | Python 视觉模型族，C 只有 ViT |
| MTPLanguageModel（多 token 预测头） | 缺失 | 训练型头 + 训练循环 |
| FlashAttention（`attn_impl='flash'` 接入 MHA/TransformerEncoder/CausalLM） | 缺失 | C `attention.h` 只有稠密注意力；分块在线 softmax 前向/反向均为纯数值，可移植 |
| KV cache 量化生成（`generate_kv_quantized`、`CausalLM.generate(kv_cache_bits=...)`） | 缺失 | 生成循环 + 参考 RNG 可控 |
| PagedKVCache 驱动生成（`generate_paged`） | 部分 | C 有 `block_mgr.h` 块分配器，无生成路径 |
| Medusa 多头训练（`train_medusa_heads`） | 缺失 | C 只有 `spec_decode.h` 推理 |
| RLHF 训练循环（`RLHFTrainer` / `rlhf_align`，DPO/IPO/SimPO/CPO/ORPO） | 缺失 | C `rl_utils.h` 只有 GRPO 优势等工具，无偏好损失训练环 |
| PRM 训练管道（`PRMTrainer` / `prm_train`） | 缺失 | — |
| LoRA 一键微调（`finetune_lora`） | 缺失 | C `adapt.h` 只有适配器构件 |
| PEFT 包装类（LoRA / QLoRA / DoRA / AdaLoRA / IA3 / VeRA / ReFT） | 部分 | `adapt.h` 有 LoRA 相关构件，无完整包装与训练 |
| 梯度检查点接入训练（`checkpoint_causal_lm_backward`、`fit_causal_lm_checkpointed`、`checkpoint_grads/transformer`） | 缺失 | C README 明确未移植 |
| 模拟混合精度训练（`fit_mixed_precision`） | 缺失 | C 有舍入模拟，无 `fit` 路径 |
| Prefix Tuning（`prefix_attention`、`PrefixTuning`） | 缺失 | — |
| 2D DeepONet（`DeepONet2D`） | 缺失 | C README 已声明只 Python 版 |
| DiT | 缺失 | C README 已声明只 Python 版 |
| Mamba-2 | 缺失 | C 只有 Mamba-1 前向 |
| FNO / Graphormer / Jamba / Titans / VMamba / DINOv2 / iBOT | 缺失 | C README 已声明这些「新增训练型模块」只 Python 版 |

### 2.2 训练与后训练工具

| 能力 | 状态 | 备注 |
| --- | --- | --- |
| GRPO / RLOO / Dr.GRPO / DAPO / GSPO 训练循环 | 部分 | 优势/损失工具已同步；无完整训练循环（依赖策略模型对象与采样回调） |
| RFT / Best-of-N / 拒绝采样 后训练管道 | 部分 | 纯函数已同步；训练侧管道未同步 |
| MTP / Medusa 训练 | 缺失 | 见 2.1 |
| 工程便利件（`unwrap_model`、`Trainer`、`benchmark_forward`） | 仅 Python 设计 | 依赖 `Model`/`MLP` Python 对象与回调协议 |

### 2.3 Agent / RAG / 推理搜索

| 能力 | 状态 | 备注 |
| --- | --- | --- |
| `ReActAgent`（循环 + 工具注册表 + 解析） | 缺失 | 需要回调/字典数据结构 |
| `RAGPipeline`（切块 → 检索 → 拼 prompt） | 缺失 | C 有 CLIP 检索，无 BM25/RRF 文本检索管道 |
| BM25 / RRF / ColBERT-lite 检索 | 缺失 | 纯数值可移植 |
| `LLMSearch`（MCTS + LLM 策略） | 仅 Python 设计 | C README 声明 MCTS/ToT 未移植 |
| `mcts_search` / `tree_of_thought` | 仅 Python 设计 | 依赖可调用策略/价值函数 |
| `test_time_scaling` / `constitutional_ai` | 缺失 | 依赖生成/评分回调 |

### 2.4 领域模型（前向 + 训练）

| 能力 | 状态 | 备注 |
| --- | --- | --- |
| SASRec（序列推荐） | 缺失 | — |
| BERT4Rec（双向序列推荐） | 缺失 | — |
| AutoInt / DCN-v2 | 缺失 | — |
| DeepFM / DIN / NCF / WideAndDeep / MF | 缺失 | — |
| TrOCR（图像编码器 + 文本解码器） | 缺失 | 交叉注意力，C 未包含 |
| TimeSformer（时空分离注意力） | 缺失 | — |
| WhisperLite / Wav2Vec2Lite | 缺失 | 语音前端与训练亦未移植 |
| YOLOLite（单尺度检测训练循环） | 缺失 | C 无 YOLO 相关头文件 |
| NeRF-lite / GaussianSplatLite（体渲染） | 缺失 | — |
| PointNetDetector（3D 检测头） | 缺失 | — |
| MoleculeGNN（分子性质预测） | 缺失 | — |
| AlphaFold-lite（三角乘法 + IPA） | 缺失 | — |
| VideoMAE-lite（视频掩码自编码） | 缺失 | — |
| CRNN（OCR 编码器） | 缺失 | C 有 CTC beam，无 CRNN 类 |

### 2.5 数据 / 检索 / 可解释性 / 部署杂项

| 能力 | 状态 | 备注 |
| --- | --- | --- |
| MinHash / SimHash 去重 | 缺失 | 纯数值可移植 |
| logit lens / activation patching / steering | 缺失 | 依赖模型对象，可部分移植 |
| attention maps / ascii 火焰图 / HTML 评估报告 / lr_finder / profile_model | 缺失 | 依赖 Python 对象/输出格式 |
| watermark / StreamingLLM | 缺失 | — |
| Expert-Choice / BitNet（路由与量化模型） | 缺失 | — |
| SAM / 梯度中心化 | 缺失 | — |
| HMM / 生存分析 / GBDT / Decision Transformer / uplift | 缺失 | — |
| Matryoshka 表示 / KV offload | 缺失 | — |
| Safetensors 存档 | 缺失 | 依赖文件格式；C 有自有 JSON 存档 |
| VersionedStore / 数据版本 | 仅 Python 设计 | 需要文件系统 |
| dataset_version / load_dataset_version | 仅 Python 设计 | 需要文件系统 |
| 模型注册表 / 实验追踪 | 仅 Python 设计 | 附属组件，C README 明确未包含 |

### 2.6 结构性降级（无法等价对拍）

以下不是「没做」，而是 C 扁平数组接口在语义上无法表达，已在各模块 README/头注释中注明：

1. **结构校验类错误**：shape 不一致、各行不等长、非有限值位置等校验，Python 抛异常而 C 只有扁平入参，无法对拍错误路径。
2. **全局 `random`（seed=None）路径**：Python 用 OS 熵，不可复现；C 侧要求显式 seed（FIM、DARE、TTS、RWKV 采样等）。
3. **文件系统/回调依赖**：VersionedStore、MCTS/ToT、LLM-as-judge、Trainer 回调、benchmark 计时。
4. **Python 对象封装**：`Model`/`MLP` 包装、PEFT 包装、`unwrap_model` 等。

## 3. Web 同步状态

Web 仓库定位是**浏览器内可运行的单文件演示**，不引入训练循环、指标计算、C 引擎移植与
研究型组件。当前 README「与 Python 仓库的同步范围」表为 **6 行 ✅ / 9 行 ⛔**：

- ✅：可在 `index.html` / `tools/` 中核对到实现与校验证据的模块（推理/可视化类）；
- ⛔：训练、RL、指标、C 侧引擎、研究扩展等，按定位不进入 Web。

注意：✅ 表示「浏览器侧有实现且可核对」，**不代表与 Python 数值逐位等价**。
C 引擎是独立轨道，不在 Web 的同步范围内。

## 4. 建议的后续同步批次

按「风险低、纯数值、收益高」排序：

| 批次 | 内容 | 验收标准 |
| --- | --- | --- |
| C-S1 | FlashAttention 分块前向/反向（单头/多头、因果/非因果） | 与稠密实现逐位/FD 对拍；compare 全绿 |
| C-S2 | Prefix attention / PrefixTuning 参数容器 | 与 Python 参考逐位 |
| C-S3 | KV cache 量化生成、PagedKVCache 生成 | 固定 RNG 下逐 token 一致 |
| C-S4 | RLHF 偏好损失（DPO/IPO/SimPO/CPO/ORPO）+ 单步训练 | 损失与前向逐位；梯度 1e-12 容差 |
| C-S5 | PRM / Medusa 训练单步、`fit_mixed_precision` 单步 | 前向+反向逐位 |
| C-S6 | 推荐模型前向（SASRec/BERT4Rec/AutoInt/DCN-v2/DeepFM/DIN/NCF） | 前向逐位；反向 FD |
| C-S7 | 领域模型前向（TrOCR/TimeSformer/WhisperLite/Wav2Vec2Lite/YOLOLite/NeRF/GSplat/PointNetDetector/MoleculeGNN/AlphaFold-lite/VideoMAE） | 前向逐位；反向 FD |
| C-S8 | 纯函数杂项（MinHash/SimHash、BM25/RRF、attention map 数值部分） | 逐位 |
| C-S9 | Agent/RAG/MCTS（若 C 侧愿意引入回调协议） | 需要先定 C 侧接口，暂缓 |

每个批次的固定流程：

1. `c/tools/<module>_reference.py` 导出参考 JSON（固定 `random.seed`）；
2. `c/<module>.h` / `c/<module>.c` 实现；
3. `c/tests/compare_<module>.c` 对拍（逐位优先，1e-12 容差需在参考 JSON 里声明）；
4. 接入 `c/Makefile` 的 `check` 与 `clean`；
5. 更新 `c/README.md` 的计数句/结构树/功能对照；
6. 提交 `feat(<area>): 新增 <what>（同步 Python <python-head>）`，push 后确认 CI；
7. 运行两次 `make check` 确认工作区干净、参考 JSON 不被改写。

## 5. 方法与局限

- 本清单依据：C 仓库 65 个头文件与 `c/tests/compare_*.c`、`c/Makefile`、C README 的
  「与 Python 版本的功能对照 / 未包含」段落，以及本轮同步提交历史。
- 未做逐符号穷举匹配（尝试过按 AI.py 横幅分节 + 子串匹配，会大量误报，已放弃）；
  因此「已同步」按能力域归纳，「未同步」按模块组归纳，个别小函数可能被归到错误类别。
- 后续可把本文件作为同步看板：每完成一批，更新「已同步」表与对应行。

## 6. 变更记录

- 2026-11-13：初版。记录 C 侧已同步（含最近 FIM/QK-Norm/RoPE/线性注意力/RWKV/模型合并）
  与未同步分类清单、Web 同步范围说明、后续批次建议。

## 7. 2026-11-14 新增 Python 能力（尚未同步到 C / Web）

本段对应 Python 仓库 `e4aed9b..0715a5f` 的 41 个能力提交（测试 2602 → 2818）。按能力域归纳：

| 能力域 | 新增内容 |
| --- | --- |
| 注意力 / 状态空间 | 线性注意力手写反向、完整 MLA（Q 低秩 + 解耦 RoPE）、Hyena block、MoE capacity/token dropping/aux 反向、RetNet/Mamba2 chunkwise、S4D、S5、RG-LRU、Perceiver IO、FNet、Griffin block 与 Hawk 别名 |
| 大模型组合 | Mixtral 风格 MoE-LLM、DeepSeek-V3 风格 MLA+MoE+MTP、VMamba 完整分类器 |
| 训练循环 | KTO、GRPO/DAPO/GSPO、PPO-with-KL、通用 PEFT 入口 `finetune_adapter`（LoRA/QLoRA/DoRA/AdaLoRA/VeRA/ReFT）、GradScaler 动态缩放、通用梯度检查点 |
| 优化器 / 并行 | Sophia Hutchinson、Shampoo Newton 逆根与 grafting、GaLore、8-bit Adam、ZeRO-2/3、1F1B 调度、TP/SP 反向 |
| 推理 | KV cache 淘汰（H2O/SnapKV/StreamingLLM）、Prompt 前缀缓存管理、Lookahead decoding、EAGLE 草稿头训练 |
| 评估 / 安全 | Elo/Bradley-Terry、judge 校准、PII/canary/越狱扫描、工具调用评测、RULER、数据污染检测、数学等价 AST |
| 可解释性 | 归纳头/线性探针、tuned lens、ROME/MEMIT、causal tracing、ACDC |
| 工程质量 | 核心 API doctest 示例、MLP 属性测试报告、数值 fuzzing、单文件与拆包评估结论 |

同步建议：

- 优先级最高的是「纯函数 + 损失/指标」组（评估/安全、可解释性、KV 淘汰、Lookahead 解码），
  这些不依赖训练循环，可按 C-S8 的方式逐位对拍；
- 训练循环类（KTO/GRPO/DAPO/GSPO/PPO-KL/PEFT fit）需要 C 侧先补「回调 + 优化器状态」
  协议，建议放到后面统一设计（类似 C-S9 的 Agent 组）；
- 架构类（Mixtral/DeepSeek-V3/VMamba 分类器）体量大、依赖多，建议仅在需要时抽样同步前向。

## 8. 变更记录（补充）

- 2026-11-14：新增本节，记录 Python 批 1–41 的能力清单与同步建议；C/Web 侧尚未开始本批同步。
