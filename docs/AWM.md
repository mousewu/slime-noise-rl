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

## 动作、奖励与实验条件

模型输出 `{"tool_name":"...","arguments":{...}}`，参数的大小写和内部空格保持原样。噪声以排序后的完整 JSON 调用为键；action-drop 在发送前发生，observation-loss 在收到结果后发生。ALFWorld 仍使用原来的字符串动作格式。

模型用 `{"tool_name":"done","arguments":{}}` 提交。项目以 `verifier_mode: code` 调用隐藏 verifier，将 complete/incomplete 映射为 1/0 后结束轨迹；不向模型开放 verify 或全局场景查询工具。预算耗尽而未提交时奖励为 0。结束提交免 action-drop；普通工具的只读分类由 manifest 决定。Verifier、工具 schema 和会话存储路径不会作为奖励提示泄露，模型仅看到任务、可用工具和工具结果。

基础设施错误直接抛出，不当作奖励 0。会话在 finally 关闭，不自动重试可能已经生效的远程写操作。默认不保留服务端数据库快照，项目 traces 继续保存动作、观察和噪声审计。

这是新任务域，不应将其成功率与 ALFWorld 直接比较。正式 matched-LOO 实验前需核对同一 scenario/task_idx 的初始数据库及 code verifier 可复现性；当前客户端不提供数据库快照一致性认证。固定服务端代码和数据版本，并对两种算法使用同一任务划分与服务配置。

协议参考：https://github.com/huggingface/OpenEnv/tree/main/envs/agent_world_model_env
