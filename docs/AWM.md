# AWM 环境

通过 manifest 中的 `environment: awm` 选择 AWM；ALFWorld 配置与数据继续使用原启动方式。同步和 fully-async 均可使用 `configs/matched_loo_awm.yaml`。新增功能不修改 Slime，也不自动下载模型或数据；默认连接已启动的环境服务器。

## 隔离的依赖和服务

AWM 服务端与 Slime/Megatron **必须使用两个 Python 环境**。当前 AWM 的
`mcp-agent==0.2.6` 依赖 NumPy 2.x，而 Megatron 不支持 NumPy 2.x。训练端现在
使用 AWM 已公开的持久 `/ws` 协议，只依赖轻量 `websockets` 包，**不导入、不安装**
`openenv`、`agent_world_model_env` 或 `mcp-agent`。

服务端环境准备好 OpenEnv/AWM 后，单独启动本地服务：

```bash
AWM_SERVER_PYTHON_BIN=/opt/conda/envs/awm-server/bin/python \
OPENENV_DIR=/workspace/OpenEnv \
AWM_DATA_DIR=/datasets/AgentWorldModel-1K \
RUNTIME_TMPDIR=/data/awm-tmp \
bash scripts/start_awm_server.sh
```

首次在**服务端环境**安装依赖时，额外设置 `AWM_SERVER_INSTALL_DEPS=1`；它只会在
该解释器中安装 OpenEnv/AWM。训练环境绝不能设置此变量或把 OpenEnv 加到 `PYTHONPATH`。
服务端和数据均在本地，脚本不会下载模型或任务数据。

训练端只需安装本项目（它会安装 `websockets`），并连接服务的 URL：

```bash
PYTHON_BIN=/opt/conda/envs/slime-train/bin/python \
INSTALL_DEPS=1 START_AWM_SERVER=0 \
AWM_URL=http://127.0.0.1:8899 \
bash scripts/train_awm_4gpu.sh
```

训练脚本会拒绝 NumPy 2.x；服务脚本会拒绝 NumPy 1.x，从启动前就阻止两套依赖意外混用。
`START_AWM_SERVER=1` 仍可作为便利模式：它调用独立服务脚本，因而仍须提供
`AWM_SERVER_PYTHON_BIN`、`OPENENV_DIR` 和 `AWM_DATA_DIR`，且只会停止它自己启动的服务。

### 服务端错误镜像到 SwanLab

训练端和 AWM 服务端是不同进程，Uvicorn/MCP 的 traceback 不会自然出现在 Ray worker
日志中。若训练脚本以 `START_AWM_SERVER=1` 启动服务，它会自动增量读取该服务的 log，仅将
`ERROR`、HTTP 5xx、`Traceback` 及有限上下文转发到训练日志和 SwanLab Logs，并在
`runs/.../awm_server_diagnostics.log` 留下可持久化的副本。

若服务由单独终端预先启动（`START_AWM_SERVER=0`），训练命令必须显式给出该终端实际写入的
日志文件，才能镜像。先以后台日志模式启动服务：

```bash
AWM_SERVER_PYTHON_BIN=/opt/conda/envs/awm-server/bin/python \
OPENENV_DIR=/workspace/OpenEnv \
AWM_DATA_DIR=/datasets/AgentWorldModel-1K \
RUNTIME_TMPDIR=/data/awm-tmp \
AWM_SERVER_BACKGROUND=1 \
AWM_SERVER_LOG=/data/awm-tmp/awm-server-8899.log \
bash scripts/start_awm_server.sh
```

再在训练命令中传入同一路径：

```bash
AWM_SERVER_LOG=/data/awm-tmp/awm-server-8899.log \
START_AWM_SERVER=0 AWM_URL=http://127.0.0.1:8899 \
bash scripts/train_awm_4gpu.sh
```

默认每秒读取一次，首次仅检查末尾 64 KiB，避免重传整个历史日志。可用
`NOISE_RL_AWM_LOG_MIRROR=0` 禁用；或用 `NOISE_RL_AWM_LOG_POLL_SECONDS`、
`NOISE_RL_AWM_LOG_CONTEXT_LINES`、`NOISE_RL_AWM_LOG_FOLLOWUP_LINES` 调整诊断开销。
`AWM_SERVER_LOG` 应为服务器的专用输出文件，不能重定向为训练进程的 stdout/stderr，以免造成
日志回环；只在训练命令设置该变量不会为已经运行的前台服务器追溯生成日志。

服务启动器还会在**不修改 OpenEnv 工作树文件**的前提下，使用项目的运行时入口包装 AWM。
当生成 scenario 子进程的 MCP 工具返回 500 时，包装器会在 session 清理前读取该子进程
`server.log` 的末尾，连同 scenario、task、工具参数、原始错误写进外层服务日志。因此它也会被
上面的 SwanLab 镜像捕获，并在
`${AWM_DIAGNOSTICS_DIR:-$RUNTIME_TMPDIR/awm-session-diagnostics}` 留下 JSON 文件。默认每次保留
末尾 24 KiB，可用 `NOISE_RL_AWM_SUBPROCESS_LOG_TAIL_BYTES` 调整，上限为 256 KiB。

### Code verifier `others` 证据包

OpenEnv 的 **code** verifier 对未通过的判定返回 `others`（它不是 verifier 进程异常）。在未确认
它是普通策略失败、环境 API 问题还是 verifier 条件错误前，不能仅依赖这一标签。每次 `others` 都会
在外层日志写入一条紧凑的 `NOISE_RL_AWM_VERIFIER_EVIDENCE` 标记；训练端将其汇总到 SwanLab：
此诊断机制**不改变**训练客户端对 `others` 的处理，也不会把它悄悄折算成零奖励；完成证据分析后再
决定是否修改训练语义。

- `awm/verifier/noncomplete/total`：code-verifier 未通过次数；不是总验证次数，不能直接当作失败率。
- `awm/verifier/noncomplete/unique_tasks`：已观察到的不同 `(scenario, task_idx)` 数。
- `awm/verifier/evidence/{saved,skipped,disabled,error}_total`：证据包落盘状态。
- `awm/verifier/evidence/db_backups_saved_total`：已保存的 SQLite 快照数。

为避免高并发训练时产生大量临时文件，默认仅保存最多 64 个 bundle，且每个任务最多一个。每个已保存
bundle 目录为 `${AWM_DIAGNOSTICS_DIR}/awm-evidence-*/`，包含：

- `evidence.json`：任务文本、code verifier 源码及 SHA-256、完整（有上限）MCP 调用轨迹、原始
  verifier 返回、子环境日志尾部，以及 SQLite 表级初始/最终摘要与 diff；
- `initial.sqlite`、`final.sqlite`：通过 SQLite backup API 保存的独立一致性快照（包含 WAL 状态）。

先读 `evidence.json` 的 `database.diff` 和 `trajectory.entries`：任务状态已满足而 verifier 仍为
`others` 时，优先检查 verifier/数据；工具调用返回 5xx 或未改变数据库时，检查环境；状态未满足且调用
正常时，才属于普通策略失败。默认限制可按一次调试运行临时覆盖：

```bash
NOISE_RL_AWM_EVIDENCE_MAX_BUNDLES=100 \
NOISE_RL_AWM_EVIDENCE_PER_TASK=1 \
NOISE_RL_AWM_EVIDENCE_MAX_DB_BYTES=$((32 * 1024 * 1024)) \
NOISE_RL_AWM_OTHERS_LOG_TAIL_BYTES=$((8 * 1024)) \
bash scripts/start_awm_server.sh
```

将 `NOISE_RL_AWM_EVIDENCE_MAX_BUNDLES=0` 设为禁用落盘；计数仍会上报。单条轨迹与 verifier 源码的
保留上限可用 `NOISE_RL_AWM_EVIDENCE_MAX_TRAJECTORY_ENTRIES`、
`NOISE_RL_AWM_EVIDENCE_MAX_TRAJECTORY_BYTES`、`NOISE_RL_AWM_EVIDENCE_MAX_VERIFIER_BYTES` 调整。
这些变量必须在启动 AWM 服务前设置；外部服务训练时也需将同一个 `AWM_SERVER_LOG` 传给训练脚本，
才能看到 SwanLab 图表和证据路径。

每条轨迹创建独立 WebSocket session；网络操作通过后台 asyncio loop 执行，环境接口通过现有有界线程池等待返回。AWM 不使用 ALFWorld 的进程池。`environment_workers` 控制同时进行的环境请求数量，`concurrency` 控制模型侧容量，服务器 session 上限必须覆盖全部活跃轨迹（包含正在等待模型的轨迹）。先以 32/64/128 活跃轨迹逐级压测，不能把最大连接数当作实际吞吐。

## 数据

使用 `scripts/build_awm_manifest.sh` 从本地 AgentWorldModel-1K 的七个 JSONL
文件生成训练和 `valid_unseen` manifest。它复现 OpenEnv 的场景名归一化，只生成具有
至少一个 pure-code verifier 的任务，并以**完整 scenario** 为单位做确定性切分；不会下载
数据、导入 OpenEnv 或联系 AWM 服务：

```bash
AWM_DATA_DIR=/datasets/AgentWorldModel-1K \
AWM_TRAIN_MANIFEST=data/awm/train.jsonl \
AWM_VALID_UNSEEN_MANIFEST=data/awm/valid_unseen.jsonl \
AWM_MANIFEST_REPORT=data/awm/split-report.json \
bash scripts/build_awm_manifest.sh
```

默认以 seed `20260910` 留出 20% scenario。固定 `AWM_SPLIT_SEED` 和
`AWM_VALID_SCENARIO_FRACTION`，并把生成的 `split-report.json` 与论文实验记录一同保存。
报告会列出被排除的无 pure-code verifier 任务；脚本拒绝覆盖已有输出，以免意外改变数据划分。

### 初始上下文审计（必做）

AWM 的初始 observation 包含任务和完整工具 schema，其 token 数不能只从原始 JSONL
推断，必须由已启动的本地 AWM 服务实际 reset 后测量。训练前运行：

```bash
PYTHON_BIN=/opt/conda/envs/slime-train/bin/python \
AWM_MANIFEST=data/awm/train.jsonl \
AWM_CONTEXT_MANIFEST=data/awm/train.ctx16k.jsonl \
AWM_CONTEXT_REPORT=data/awm/train.ctx16k.report.json \
HF_CHECKPOINT=/models/Qwen3-4B-Instruct-2507 \
AWM_URL=http://127.0.0.1:8899 \
MAX_CONTEXT_TOKENS=16384 \
bash scripts/audit_awm_context.sh
```

该工具使用 rollout 相同的 WebSocket reset、code-verifier 检查、保留工具列表、AWM
system prompt 和 Qwen chat template。它保留满足严格条件
`initial_prompt_tokens < MAX_CONTEXT_TOKENS` 的原始 manifest 行，并将所有排除任务及
token 统计写进 report。输出和 report 都不可覆盖。任何 AWM 服务/协议错误都会终止审计且
不写输出，避免因暂时性服务故障悄悄改变训练分布。训练时把 `AWM_MANIFEST` 指向该新文件；
它不替代训练中后续多轮 context budget 的正常截断统计。

可选的 `AWM_READ_ONLY_TOOLS` 是一个人工审计的 JSON 映射；`_default` 为所有未单独
列出的 scenario 提供显式默认值。例如：

```json
{
  "_default": [],
  "e_commerce_33": ["search_products"]
}
```

未提供该文件时，所有工具均视为可能写入（保守但可运行）。生成后的 JSONL 每行例如
（场景和任务编号须存在于本地 AWM 服务）：

```json
{"prompt":"Complete the tool-use task.","metadata":{"task":{"id":"awm/e_commerce_33/0","environment":"awm","split":"train","scenario":"e_commerce_33","task_idx":0,"read_only_tools":["search_products"]}}}
```

`read_only_tools` 必须人工核对真实工具行为；未列出的工具按可能写入处理。不要根据 search/get 等名称自动推断。任务 ID 应由场景和任务编号唯一确定，不应把 split 加进 ID 来掩盖训练/评估重叠。评估建议按整个 scenario 留出，使用 `valid_unseen` split。

## 启动

沿用原来的训练命令及本地模型路径，把 `--config` 改为 `configs/matched_loo_awm.yaml`，`--data` 改为 AWM manifest，并选择新的输出目录即可。fully-async 继续使用 `scripts/train_fully_async.sh`，GPU 分配参数保持原设置。

四卡 fully-async 可使用 `scripts/train_awm_4gpu.sh`。它只安装训练端项目/SwanLab
依赖，校验四张可见 GPU、两个本地 checkpoint、manifest、NumPy 1.x 和 AWM 服务可达性；
不会安装 OpenEnv/AWM 或修改 Slime 的 CUDA 运行栈。服务端由
`scripts/start_awm_server.sh` 单独负责。完整环境变量示例见脚本开头和 README。

八卡使用独立的 `scripts/train_awm_8gpu.sh`，固定拓扑为 **2 actor + 6 rollout**：actor
为 TP=2，rollout 为三个 TP=2 SGLang engine。脚本默认总 `CONCURRENCY=48`，启动器将其
均分为每 engine 16；`environment_workers` 仍先保持 32，因为应先依据
`rollout/environment_queue_seconds` 判断环境是否真有瓶颈。请先完成上面的 16K context
审计，再执行：

```bash
AWM_MANIFEST=data/awm/train.ctx16k.jsonl \
MAX_CONTEXT_TOKENS=16384 MAX_TOKENS_PER_GPU=16384 \
bash scripts/train_awm_8gpu.sh
```

`MAX_CONTEXT_TOKENS`、`CONCURRENCY` 和 `ENVIRONMENT_WORKERS` 可在两个训练脚本中直接
覆盖，无须复制 YAML；它们会被写入运行时配置和 SwanLab 元数据。`MAX_TOKENS_PER_GPU`
必须不小于实际 context 上限。7 张 rollout 卡加 2 张 actor 卡需要 9 张物理 GPU，且 7
不能被当前每 engine 2 卡的布局整除。

## 动作、奖励与实验条件

模型输出 `{"tool_name":"...","arguments":{...}}`，参数的大小写和内部空格保持原样。噪声以排序后的完整 JSON 调用为键；action-drop 在发送前发生，observation-loss 在收到结果后发生。ALFWorld 仍使用原来的字符串动作格式。

模型用 `{"tool_name":"done","arguments":{}}` 提交。项目以 `verifier_mode: code` 调用隐藏 verifier，将 complete/incomplete 映射为 1/0 后结束轨迹；不向模型开放 verify 或全局场景查询工具。预算耗尽而未提交时奖励为 0。结束提交免 action-drop；普通工具的只读分类由 manifest 决定。Verifier、工具 schema 和会话存储路径不会作为奖励提示泄露，模型仅看到任务、可用工具和工具结果。

基础设施错误直接抛出，不当作奖励 0。会话在 finally 关闭，不自动重试可能已经生效的远程写操作。默认不保留服务端数据库快照，项目 traces 继续保存动作、观察和噪声审计。

这是新任务域，不应将其成功率与 ALFWorld 直接比较。正式 matched-LOO 实验前需核对同一 scenario/task_idx 的初始数据库及 code verifier 可复现性；当前客户端不提供数据库快照一致性认证。固定服务端代码和数据版本，并对两种算法使用同一任务划分与服务配置。

协议参考：https://github.com/huggingface/OpenEnv/tree/main/envs/agent_world_model_env
