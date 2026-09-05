# 本地验证记录

日期：2026-09-04。验证机器为 macOS/Apple Silicon，Python 3.12.2，**没有CUDA GPU**。

本轮最终结果：**72项测试全部通过，0项跳过**，耗时约5.6秒；Ruff静态检查、格式检查、Python编译和shell语法检查均通过。该结果包含下述真实依赖接口测试，不包含GPU训练。

## 已执行

- Ruff静态检查、代码格式检查、Python编译检查。
- 单元测试：配置校验、随机数命名空间、相同情景共享/不同策略种子、乱序分组、LOO/mean/std、全零组、缺失组拒绝。
- 可枚举Bernoulli任务上的LOO期望梯度核验；不等于实际Slime优化器的无偏/降方差证明。
- 故障注入：局部操作计数、无关动作不移动另一操作的随机序列、边际概率、只读操作、相关故障块、重试预算、观测丢失下终局奖励保留。
- 轨迹：模型token与环境token分离、掩码、上下文/生成/工具预算、严格JSON动作、基础设施异常传播与资源释放。
- 实际Slime版本 `4c193f1f37509cca70f0e88807a9305b70f63f4e`：真实Sample token/logprob/mask接口、奖励回调、跨epoch采样和数据源恢复。
- 实际Qwen3-4B-Instruct-2507 tokenizer：多轮append-only角色边界、原始生成token保持。
- 实际ALFWorld 0.4.2 / TextWorld 1.7.0：从官方release取一个solvable游戏，验证reset/step、两个环境的状态隔离；仅在测试中启用planner，验证最终成功信号。训练代码不启用planner。
- 8卡训练启动命令dry-run、官方Qwen模型配置与rope值、checkpoint路径格式、shell语法。
- CLI冻结策略probe/独立评测的流程测试使用模拟生成客户端；它们不是实际LLM推理实验。

测试命令（路径替换成自己的依赖位置）：

```bash
SLIME_SOURCE_PATH=/path/to/pinned/slime \
ALFWORLD_TEST_GAME=/path/to/game.tw-pddl \
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
python -m compileall -q src tests scripts
bash -n scripts/train.sh scripts/convert_checkpoint.sh
```

如果不提供Slime源码、Qwen tokenizer缓存或ALFWorld游戏，可选接口测试会明确skip；skip不能视为这些接口已验证。

可用 `scripts/extract_smoke_game.py` 从官方 `json_2.1.3_tw-pddl.zip` 提取一个游戏，避免为接口检查展开全部数据。来源为 [ALFWorld 0.4.2 release资产](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip)。

## 尚未执行，不能声称通过

- 真实SGLang GPU HTTP生成、确定性GPU采样和并发下logprob返回。
- HF→Megatron GPU权重转换、8×A100联合启动、GPU显存/吞吐检查。
- Megatron反向传播、训练/rollout权重同步、完整训练checkpoint恢复。
- 多训练种子学习曲线、等token预算比较、统计显著性、跨环境迁移。
- BFCL接入及完整验证/幂等harness基线。

下一步是在服务器上先跑README中的2轮rollout-only检查，然后换新目录跑2轮完整训练和一次恢复。通过后再开始完整实验矩阵。
