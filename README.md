# Slime Noise RL

基于 **Slime + SGLang + Megatron** 的多轮智能体强化学习实验代码，用于研究：在相同 rollout 预算下，按共享故障情景比较轨迹，能否改善随机工具环境中的学习效率。

本项目是研究首版，不预设候选方法优于基线。已经实现 ALFWorld 文本环境接入、故障注入、分组采样、奖励后处理、训练启动、独立评测和冻结策略诊断。**没有训练出模型，也没有接入 BFCL；后者属于第二阶段跨环境验证。** 内置 `mini` 是 CPU 调试夹具，不是新 benchmark。

## 1. 兼容版本和已验证范围

- Slime：最初针对 [`4c193f1f37509cca70f0e88807a9305b70f63f4e`](https://github.com/THUDM/slime/tree/4c193f1f37509cca70f0e88807a9305b70f63f4e) 完成接口验证，但启动时不限制 commit。运行记录会保存实际 HEAD，并拒绝 tracked 文件有本地修改的 checkout；更换版本后应重新执行接口与两卡训练验证。
- 模型：`Qwen/Qwen3-4B-Instruct-2507`，非 Thinking 版本；使用对应的官方 Megatron 配置，包括 `rotary-base=5000000`。
- 本地验证：CPU 单元测试、真实 Slime `Sample` 接口、真实 Qwen tokenizer、ALFWorld 0.4.2 / TextWorld 1.7.0 的真实游戏交互与终局校验。
- **尚未验证：A100 上的 SGLang/Megatron 内核、权重同步、完整反向传播和训练收敛。** 8×A100 80GB 参数是保守起点，不是吞吐或显存保证。

上游当前快速开始主要说明 H/B 系列 GPU。A100 需要确认所用环境提供 SM80 可用的 FlashAttention/FlashInfer 与 Transformer Engine。不要直接假设任意最新版容器都兼容；确认后记录容器 digest 和依赖版本。这里使用 BF16，不启用 FP8/FlashAttention-3。

## 2. 代码结构

```text
src/noise_rl/
  sampling.py       稳定种子、独立采样和共享情景分组
  noise.py          操作局部计数的故障随机场；计费的保守重试 harness
  envs.py           ALFWorld 文本环境和 mini 调试夹具
  agent.py          append-only 多轮轨迹、SGLang 原生 token/logprob 请求
  advantages.py     prompt/情景分组、mean/LOO、可选标准差归一化
  data.py           严格任务清单、Slime 数据源与恢复状态
  slime_hooks.py    Slime 生成、奖励后处理、独立评测接口
  metrics.py        任务级统计、配对 bootstrap、嵌套采样方差诊断
  swanlab_bridge.py 不改Slime源码的分布式SwanLab指标转发
  launch.py         Slime状态记录、8卡参数、运行记录与安全启动
configs/            7组对照/消融配置
scripts/            训练启动、权重转换、单游戏提取工具
tests/              CPU 与可选真实依赖接口测试
docs/RESEARCH.md     数学定义、实验矩阵、限制与发表前检查
docs/VALIDATION.md   验证范围和运行记录
```

## 3. 先在 CPU 上验证代码

在项目根目录运行；Python 3.10–3.12 为建议范围。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
noise-rl demo --method matched --tasks 8 --output runs/demo-matched.jsonl
noise-rl demo --method independent --tasks 8 --output runs/demo-independent.jsonl
```

`demo` 使用脚本策略和字符 tokenizer，日志会明确标记 `SCRIPTED_FIXTURE_NOT_LLM`。它仅用于检查闭环，成功率不能用于论证 RL 方法有效。

可选：安装 CPU 接口测试依赖，在干净的 Slime 源码上运行真实 `Sample` 测试：

```bash
pip install -e '.[dev,contract,alfworld]'
SLIME_SOURCE_PATH=/path/to/slime pytest -q
```

`contract` extra 可能安装 PyTorch，**不要在已配置好的 GPU Slime 环境中重新安装这个 extra**。Qwen tokenizer 测试只使用本地缓存，不会在测试里自动联网。

## 4. GPU 服务器准备

首先按照 [Slime 官方安装说明](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/docs/en/get_started/quick_start.md) 配好兼容 A100 的 Slime/SGLang/Megatron 环境。以下代码不会帮你替换 CUDA、驱动或 GPU 运行库。

在一个新目录检出对应源码；不要对有修改的旧 checkout 强制切换版本：

```bash
git clone https://github.com/THUDM/slime.git /workspace/slime
git -C /workspace/slime checkout 4c193f1f37509cca70f0e88807a9305b70f63f4e
pip install -e /workspace/slime --no-deps
export SLIME_DIR=/workspace/slime
```

把本项目复制到服务器，例如 `/workspace/slime-noise-rl`。在已经配置好的 GPU 环境内安装项目及文本环境，保留 Slime 原有的 torch/transformers/sglang 版本：

```bash
cd /workspace/slime-noise-rl
pip install -e '.[dev,alfworld,tracking]'
# 可选：未对当前 Python 做 editable 安装时，告诉项目 Megatron-LM 源码位置。
export MEGATRON_LM_DIR=/workspace/Megatron-LM
```

`tracking` 只增加 SwanLab SDK，不替换 Slime 的 GPU 依赖。不使用 SwanLab 时可以省略该 extra。

如果安装器提示需要更换 GPU 核心依赖，先停止并在隔离环境解决，不要盲目升级。建议把 `pip freeze` 和容器 digest 保存在每组实验记录中。

### 两张GPU一键试跑

如果已经按照 Slime 文档配置好 CUDA、SGLang、Megatron、Ray 和一个干净的 Slime checkout，可以用下面的脚本完成其余依赖安装、本地资源校验、必要的本地权重转换、dry-run 以及两轮短程在线 RL。脚本不会下载模型或任务数据：

```bash
export SLIME_DIR=/workspace/slime
export MEGATRON_LM_DIR=/workspace/Megatron-LM
export HF_CHECKPOINT=/models/Qwen3-4B-Instruct-2507
export MEGATRON_CHECKPOINT=/models/Qwen3-4B-Instruct-2507_torch_dist
export ALFWORLD_ROOT=/datasets/alfworld/json_2.1.1
bash scripts/smoke_2gpu.sh
```

脚本默认只暴露 GPU 0、1，使用 `configs/smoke_2gpu.yaml`：训练 TP=2、一个双卡 rollout engine、每任务4条轨迹、最多8个模型回合，并执行2个完整 rollout（包含反向传播和checkpoint保存）。这是管线验证配置，不能作为论文主结果。

`HF_CHECKPOINT` 和 `ALFWORLD_ROOT` 必须指向已有本地目录。`MEGATRON_CHECKPOINT` 未设置时默认为 `${HF_CHECKPOINT}_torch_dist`；若不存在，脚本只从本地 HF checkpoint 转换。默认使用离线 SwanLab。常用方式：

```bash
# 也可以把本地路径保存在不会提交的配置文件中
cp configs/local_paths.env.example configs/local_paths.env
# 编辑路径后：
source configs/local_paths.env
bash scripts/smoke_2gpu.sh

# 只验证在线rollout，不反向传播
SLIME_DIR=/workspace/slime ROLLOUT_ONLY=1 bash scripts/smoke_2gpu.sh

# 上传到SwanLab；API Key仍通过登录状态或环境变量提供
SLIME_DIR=/workspace/slime SWANLAB_MODE=online bash scripts/smoke_2gpu.sh
```

可通过 `CUDA_VISIBLE_DEVICES=2,3` 选择其他两张卡，`NUM_ROLLOUT=1` 缩短运行，`USE_SWANLAB=0` 关闭实验记录，`SMOKE_OUTPUT=/path/to/new-run` 指定新输出目录。脚本不会安装 Slime、覆盖checkpoint、清空旧输出或停止已有Ray进程；Slime的GPU运行依赖缺失时会直接报错。

### 模型和初始化权重

项目不提供模型下载逻辑。已有 HF checkpoint 可按下面方式转换；转换脚本拒绝已经存在的输出路径，训练脚本不会执行 `pkill`、`ray stop` 或清空 checkpoint：

```bash
bash scripts/convert_checkpoint.sh \
  /models/Qwen3-4B-Instruct-2507 \
  /models/Qwen3-4B-Instruct-2507_torch_dist
```

### ALFWorld 任务清单

本项目只需要 ALFWorld 的 `traj_data.json` 和 `game.tw-pddl` 文本游戏。手工准备好本地数据后生成 manifest：

```bash
noise-rl prepare --environment alfworld \
  --root /datasets/alfworld/json_2.1.1 --split train \
  --output data/alfworld/train.jsonl
noise-rl prepare --environment alfworld \
  --root /datasets/alfworld/json_2.1.1 --split valid_unseen \
  --output data/alfworld/valid_unseen.jsonl
```

任务清单记录绝对游戏路径、官方 split、任务类型及游戏 SHA256。请在最终训练服务器上生成；更换数据位置应重新生成清单。读取时保留官方 solvable 筛选及 movable/sliced 排除规则。所需压缩包、目录结构和无需下载的视觉/预训练资源见 [本地模型与数据配置](docs/LOCAL_ASSETS.md)。

`--limit 128` 可用于小规模预实验，按固定hash选择任务，避免只取字典序最前的一种任务类型；正式实验应记录任务分布。清单生成拒绝覆盖已有文件。`valid_seen`、`valid_unseen` 不允许作为训练清单。

## 5. 训练：先检查，再短跑，再主实验

默认：单节点8卡、训练 TP=2、4个双卡 rollout engine、colocate、8K上下文、每条轨迹最多2048个模型生成 token、单次输出最多96 token、40个模型回合、50次环境调用。故障概率默认 action-drop=0.15、observation-loss=0.10。

`concurrency` 是全局 SGLang 请求容量：主配置设为 32，在 8 卡、每个 engine 2 卡时对应每个 rollout engine 8 条并发请求。`environment_workers` 只服务于普通的进程内同步环境调用。

ALFWorld 默认启用 `environment_processes: 32`。它在**每个 RolloutManager** 内创建最多 32 个 CPU-only runner；一条活跃轨迹从创建、`reset`、多次 `step` 到 `close` 始终独占其中一个 runner。这样，同一轨迹仍严格遵循“模型生成一个动作 → 环境执行一个动作 → 模型接收观察”的顺序，但不同轨迹的真实 TextWorld/ALFWorld step 可以跨进程并行，不会共享 Tatsu 的非线程安全解析器。故障注入、重试、工具调用计数和奖励仍留在 rollout 父进程：被 action-drop 的动作不会发送给 runner，observation-loss 发生在真实 step 返回后，因此实验语义不变。

`environment_processes` 是环境并行上限，不是 SGLang 的 `concurrency`。项目会自动把 ALFWorld RPC 的线程容量提高到该数值；不需要为了它再增大 `environment_workers`。这些 runner 是项目在 rollout worker 内启动的 CPU 子进程，不会被 Ray 的 GPU 调度自动计入 CPU 资源：配置 32 前应确认每个 RolloutManager 有至少约 32 个可用 CPU 核和足够内存。如果主机 CPU 或内存不足，可先改为 16；如果 `rollout/environment_runner_wait_seconds` 长期接近 0 而 GPU 仍空闲，瓶颈就不再是可用 runner 数，而更可能是模型请求并发、轨迹长度或训练/rollout 拓扑。

ALFWorld 的 FastDownward 与 Ray 默认都会使用 `/tmp`。长跑前应把它们移到有足够空间的本地盘，例如：

```bash
TMPDIR=/data/noise-rl-tmp \
NOISE_RL_RAY_TMPDIR=/tmp/nrl \
bash scripts/train_fully_async.sh ...
```

启动器会将 `TMPDIR` 传入 Ray worker 和 runner，并让本地 Ray session 写入 `NOISE_RL_RAY_TMPDIR/ray`。Ray 的 Unix socket 全路径不能超过 107 字节，因此项目目录很深时必须使用短路径；`/tmp/nrl` 可以是指向大容量盘目录的软链接。不要让它被 Python 解析为长的真实路径。若显式使用远程 Ray cluster，则由集群管理员配置 Ray 的临时目录，`TMPDIR` 仍会传给 ALFWorld runner。

先检查命令，不启动 Ray/GPU，也不创建实验输出目录：

```bash
bash scripts/train.sh \
  --config configs/matched_loo.yaml \
  --data data/alfworld/train.jsonl \
  --eval-data data/alfworld/valid_unseen.jsonl \
  --hf-checkpoint /models/Qwen3-4B-Instruct-2507 \
  --megatron-checkpoint /models/Qwen3-4B-Instruct-2507_torch_dist \
  --output runs/matched-s42 \
  --seed 42 --dry-run
```

然后换一个输出目录，只跑2轮、每轮2个任务，先检查 rollout：

```bash
bash scripts/train.sh \
  --config configs/matched_loo.yaml \
  --data data/alfworld/train.jsonl \
  --hf-checkpoint /models/Qwen3-4B-Instruct-2507 \
  --megatron-checkpoint /models/Qwen3-4B-Instruct-2507_torch_dist \
  --output runs/rollout-smoke-s42 \
  --seed 42 --batch-size 2 --num-rollout 2 --debug-rollout-only
```

确认返回 token、reward、loss mask 和实际显存都正常后，在**另一个新输出目录**去掉 `--debug-rollout-only`，用相同小规模参数验证一次完整训练与保存/恢复。不要直接把 rollout-only 目录当训练 checkpoint 恢复。

主实验去掉 `--dry-run`，默认300轮、16个任务/轮、8条轨迹/任务，即 **38,400条轨迹/训练种子**。生成 token 和工具调用成本会因策略而不同，必须同时报告。`--num-rollout` 是总轮数，不是恢复后额外增加的轮数。

### Fully-async rollout：独立、非 colocate 的实验路径

当同步 rollout 被长尾 agent 轨迹阻塞时，可使用独立的 `scripts/train_fully_async.sh`。它通过 Slime 固定版本提供的 `train_async.py` 和 `slime.rollout.fully_async_rollout.generate_rollout_fully_async`，持续保留固定数量的 in-flight group；每次训练只取已经完成的 group，不必等待同批次最慢的轨迹。项目继续使用原有的多轮 `noise_rl.slime_hooks.generate` 和奖励后处理，不修改 Slime 源码。

这不是同步 colocate 配置的开关：Slime 的 async driver 要求训练与 rollout 使用**互不重叠**的 GPU。因此它使用 `--actor-gpus`（Megatron 训练）和 `--rollout-gpus`（SGLang），两者之和必须等于 `--gpus`。默认的 8 卡配方为 4 张训练卡（TP=2、DP=2）和 4 张 rollout 卡（两个 TP=2 engine）。新配置的 `concurrency: 16` 因而会传成每 engine 8 条请求，Slime worker 全局维持 16 个 in-flight group。

先用新输出目录短跑；fully-async 不支持 `--eval-data`、在线 evaluation 或 `--resume`，因为其完成队列并不随 checkpoint 保存：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_fully_async.sh \
  --config configs/matched_loo_fully_async.yaml \
  --data data/alfworld/train.jsonl \
  --hf-checkpoint /models/Qwen3-4B-Instruct-2507 \
  --megatron-checkpoint /models/Qwen3-4B-Instruct-2507_torch_dist \
  --output runs/matched-async-smoke-s42 \
  --seed 42 --gpus 8 --actor-gpus 4 --rollout-gpus 4 \
  --tensor-parallel 2 --engine-gpus 2 \
  --batch-size 2 --num-rollout 3 --save-interval 1 \
  --max-tokens-per-gpu 9216 --use-swanlab \
  --swanlab-project agentic-noise-rl \
  --swanlab-experiment-name matched-async-smoke-s42
```

短跑无误后，建议使用 `--batch-size 8 --num-rollout 600 --save-interval 50`：每个训练 DP rank 的样本数仍为 32，总采样数也与默认同步配方相同。fully-async 改变了样本完成与权重更新的时序，不能把它与同步结果视为逐步等价；论文比较中应将它作为单独训练系统，并对所有方法使用同一 async 拓扑。

评估应在训练任务结束或释放 rollout GPU 后，以保存的 `hf/rollout_<id>/` checkpoint 启动独立的、兼容 `/generate` 的 SGLang server，再运行已有评估工具。例如：

```bash
noise-rl evaluate \
  --config configs/matched_loo_fully_async.yaml \
  --data data/alfworld/valid_unseen.jsonl \
  --model runs/matched-async-s42/hf/rollout_49 \
  --url http://127.0.0.1:30000 \
  --repeats 4 --seed 20260904 \
  --output runs/matched-async-s42/eval/rollout_49.jsonl
```

### SwanLab实验记录（可选）

先通过 `swanlab login` 登录，或只在运行环境中设置 `SWANLAB_API_KEY`；不要把密钥写入启动命令或配置文件。在线记录示例：

```bash
bash scripts/train.sh \
  --config configs/matched_loo.yaml \
  --data data/alfworld/train.jsonl \
  --eval-data data/alfworld/valid_unseen.jsonl \
  --hf-checkpoint /models/Qwen3-4B-Instruct-2507 \
  --megatron-checkpoint /models/Qwen3-4B-Instruct-2507_torch_dist \
  --output runs/matched-s42 \
  --seed 42 --use-swanlab \
  --swanlab-project agentic-noise-rl \
  --swanlab-experiment-name matched-loo-s42 \
  --swanlab-group matched-loo \
  --swanlab-tags alfworld qwen3-4b seed-42
```

可用 `--swanlab-mode offline` 仅保存本地记录，或者用 `--swanlab-mode local` 配合本地看板；默认日志目录是运行目录下的 `swanlab/`，可通过 `--swanlab-logdir` 改写。`SWANLAB_API_HOST` 可指定私有部署地址。

接入层不修改 Slime 源码：Ray 的 worker setup hook 在每个训练进程中保留 Slime 原日志调用，并把标量指标转发给单一 SwanLab logger actor。SwanLab SDK 自动维护全局事件 step（包括断点恢复），桥接层同时保留 `train/step`、`rollout/step`、`eval/step` 等 Slime 原生计数器。每个完整 rollout 还会从逐轨迹 trace 汇总并记录 `rollout/success_rate`、token、工具调用、轨迹耗时、模型请求耗时、环境 step/排队耗时、worker 内 in-flight 轨迹数、终止原因和实际噪声触发率；评估会产生对应的 `eval/<dataset>/*` 指标。`--resume` 会复用 `run.json` 中保存的 SwanLab run ID；恢复时保持 SwanLab 项目、实验名和日志目录不变。

启用 SwanLab 后，项目还会将每个 Ray worker 的 Python `INFO`、`WARNING`、`ERROR` 日志非阻塞镜像到 SwanLab 的 Logs 页；原始 Ray 日志仍保留在 Ray session 目录。本地 `swanlab/forwarded_logs.jsonl` 是镜像日志的完整审计副本，`metric_events.jsonl` 则保存标量指标。为减少噪声或上传量，可在启动前设置 `NOISE_RL_SWANLAB_LOG_LEVEL=WARNING`（默认 `INFO`）。`SWANLAB_API_KEY`、`HF_TOKEN` 和 `HUGGING_FACE_HUB_TOKEN` 出现在日志文本时会被脱敏。

运行目录包含：

- `run.json`、`runtime_config.json`：启动命令、Slime版本、完整实验参数。
- `swanlab/`：启用 SwanLab 时的本地SDK记录；`metric_events.jsonl` 是所有实际提交给 SDK 的标量审计日志，run ID同时保存在 `run.json`。
- `checkpoints/`：Slime训练checkpoint和采样计数器状态。
- `hf/rollout_<id>/`：可用于独立部署评估的HF权重。
- `traces/train/`、`traces/eval_<id>/`：逐轨迹动作、故障审计、token成本和成功标记。
- `traces/advantages/`：原始奖励、分组标识和实际传给Slime的标量优势。
- `traces/metrics/`：每个训练 rollout 与每次评估从 trace 汇总出的 SwanLab 指标副本，可用于离线复核。

恢复时加 `--resume` 并保持原参数、数据和路径；可增大 `--num-rollout`。数据源状态丢失、数据内容或采样配置变化都会拒绝恢复，避免悄悄重置环境随机情景。

默认可复现的是数据顺序和故障随机场，不承诺GPU结果逐位一致。确认内核支持后，可以所有对照统一添加 `--deterministic`，启用上游的确定性推理/训练开关及对应环境变量。它可能影响速度或要求调整GPU依赖，应先短跑验证，不能只给候选方法开启。

## 6. 对照配置

| 配置 | 环境随机性 | 优势比较范围 | 基线/标准差 |
|---|---|---|---|
| `independent_loo` | 每条轨迹独立 | 同任务8条 | LOO / 不除标准差 |
| `matched_loo` | 2情景×4策略采样 | 同任务、同情景4条 | LOO / 不除标准差 |
| `coupled_prompt_loo` | 2情景×4策略采样 | 同任务8条 | LOO / 不除标准差 |
| `independent_grpo` | 每条轨迹独立 | 同任务8条 | mean / sample std |
| `matched_grpo` | 2情景×4策略采样 | 同任务、同情景4条 | mean / sample std |
| `clean_grpo` | 无故障 | 同任务8条 | mean / sample std |
| `retry_loo` | 每条轨迹独立＋最多2次规则重试 | 同任务8条 | LOO / 不除标准差 |

主要算法比较应是 **`matched_loo` vs `independent_loo`**。不要只比较 matched-LOO 和标准GRPO后，把全部收益归因于情景分组。建议训练种子42/43/44，固定任务和预算。

`retry_loo` 只自动重试“明确报告未执行”的请求，所有重试计入工具调用预算；不实现通用事务幂等或完整 postcondition verification。训练时奖励只取环境终局成功0/1，不额外奖赏重试行为，不过滤全0/全1分组。

## 7. 固定模型预实验与独立评测

部署一个独立SGLang实例，加载要评估的**确切HF checkpoint**，不要同时在同一批卡上运行主训练。部署参数按所用SGLang版本/A100环境配置；接口必须支持 `/generate`、输入token、输出token logprobs。

冻结模型、16个故障情景×4个策略采样，对16个任务先测方差结构：

```bash
noise-rl probe --config configs/matched_loo.yaml \
  --data data/alfworld/valid_unseen.jsonl \
  --model /models/Qwen3-4B-Instruct-2507 --url http://127.0.0.1:30000 \
  --tasks 16 --scenarios 16 --policy-samples 4 --output runs/probe.jsonl
```

这会生成实际模型轨迹与 `.probe.json`，**不更新模型**。结果是奖励方差诊断，不是梯度方差，也不是不可学习噪声的因果估计。

独立评测示例：

```bash
noise-rl evaluate --config configs/independent_loo.yaml \
  --data data/alfworld/valid_unseen.jsonl \
  --model runs/matched-s42/hf/rollout_299 --url http://127.0.0.1:30000 \
  --repeats 4 --seed 20260904 --output runs/matched-eval.jsonl
noise-rl summarize runs/matched-eval.jsonl
noise-rl compare runs/independent-eval.jsonl runs/matched-eval.jsonl
```

`--model` 提供与已部署权重匹配的 tokenizer 路径，不能替你自动切换服务端权重；请为每次部署记录 checkpoint。所有待比较模型使用同一个评测config、任务顺序、seed、重复数与harness。评测不使用训练时的共享故障情景；训练种子变化也不会改变默认评测种子。`--clean` 测干净环境；未见强度/相关故障需单独评测config。

`compare` 检查任务/重复/种子/预算与故障条件一致，并按任务做配对bootstrap。该置信区间不能替代多个训练种子。推理输入token统计是每次请求的完整输入长度之和，包含可能命中缓存的token，不等于实际FLOPs。

## 8. 研究边界

- ALFWorld扰动结果应单独命名，不能作为标准ALFWorld排行榜分数。
- hash随机场只规范化大小写与空白，不声称理解任意自然语言同义操作。
- 只支持Qwen3 Instruct的append-only chat framing，不支持上下文压缩、Thinking模板、视觉、多智能体或partial rollout。
- 故障在调用执行前/返回观测后注入，不模拟真实时间、并发事务、延迟提交或幂等性。
- 终局成功会结束episode；即使终局观测丢失，episode终止仍能体现成功，不是完整“隐藏提交结果”基准。
- 共享随机数、LOO都不是本项目的新发明。能否形成论文取决于机制证据、等预算提升、第二环境验证与进一步文献查重。

更多定义、实验设计和后续清单见 [研究说明](docs/RESEARCH.md)。
