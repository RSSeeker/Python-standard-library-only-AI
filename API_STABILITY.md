# API 稳定性约定

本表说明哪些 API 是「逐位一致」承诺、哪些只是教学实现。
逐位一致 = 固定输入/固定 seed 下跨平台数值相同（CI 的
`tools/numeric_regression.py --check` 对此有快照校验）。

## 数值逐位一致（序列化/导出）

| API | 约定 | 说明 |
|---|---|---|
| `safetensors_save` | `bit-exact` | 写入的 f32 字节可直接被读回 |
| `safetensors_load` | `bit-exact` | 读回与写入逐位相同 |
| `export_onnx_mlp` | `bit-exact` | MLP 的 ONNX 图与权重确定性导出 |
| `cross_entropy_loss` | `bit-exact` | 固定输入下结果确定 |
| `mse_loss` | `bit-exact` | 固定输入下结果确定 |
| `ScaledCosineAttention` | `bit-exact` | 固定 seed 初始化后前向确定 |

## 数值稳定但仍为教学实现

| API | 约定 | 说明 |
|---|---|---|
| `zero_adam_step` | `teaching-numeric` | 算法数值模拟，未接真实通信 |
| `tensor_parallel_linear_backward` | `teaching-numeric` | 分片数学正确，未接 NCCL |
| `pipeline_1f1b_schedule` | `teaching-numeric` | 调度顺序模拟 |
| `Shampoo` | `teaching-numeric` | Newton 逆根有限步迭代，非精确矩阵根 |
| `sophia_hutchinson_diag` | `teaching-numeric` | Hutchinson 估计有方差 |

## 仅教学/演示（不承诺 API 稳定）

| API | 约定 | 说明 |
|---|---|---|
| `rope_2d` | `demo-only` | 形状与数学约定可能随文档演进 |
| `mcts_search` | `demo-only` | 纯 Python 搜索，性能非目标 |
| `consistency_model_sample` | `demo-only` | 简化采样器 |

## 说明

- 未列出的名字默认按「教学实现」处理，不承诺跨版本签名稳定；
- 逐位一致的判定依赖 CI 快照（12 组固定计算，容差 1e-6）；
- 序列化格式（safetensors/ONNX）以本库写出→本库读回为准，
  不保证与第三方实现的字节级兼容。
