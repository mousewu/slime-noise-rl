# AWM 环境

通过 manifest 中的 `environment: awm` 选择 AWM；ALFWorld 配置与数据继续使用原启动方式。同步和 fully-async 均可使用 `configs/matched_loo_awm.yaml`。新增功能不修改 Slime，也不自动下载模型、数据或启动环境服务器。

## 依赖和服务

在已有 OpenEnv 源码及其依赖准备好的环境中，把 OpenEnv 的 `src` 和 `envs` 目录加入 `PYTHONPATH`，使 `openenv` 和 `agent_world_model_env` 可导入。项目训练启动器会保留并传播 PYTHONPATH。

```bash
export PYTHONPATH=/workspace/OpenEnv/src:/workspace/OpenEnv/envs:${PYTHONPATH:-}
```

单独按 OpenEnv 的 AWM 服务说明准备本地数据并启动服务器，然后在 YAML 设置 `noise_rl.awm_url`。本项目只连接该服务，不调用 from_hub 等下载接口。AWM 服务的数据加载、磁盘目录和会话并发容量由其部署配置负责。

每条轨迹创建独立 WebSocket session；网络操作通过后台 asyncio loop 执行，环境接口通过现有有界线程池等待返回。AWM 不使用 ALFWorld 的进程池。`environment_workers` 控制同时进行的环境请求数量，`concurrency` 控制模型侧容量，服务器 session 上限必须覆盖全部活跃轨迹（包含正在等待模型的轨迹）。先以 32/64/128 活跃轨迹逐级压测，不能把最大连接数当作实际吞吐。

## 数据

使用 `scripts/build_awm_manifest.sh` 从本地 AgentWorldModel-1K 的七个 JSONL
文件生成训练和 `valid_unseen` manifest。它复现 OpenEnv 的场景名归一化，要求每个
任务都有 pure-code verifier，并以**完整 scenario** 为单位做确定性切分；不会下载
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
脚本拒绝覆盖已有输出，以免意外改变数据划分。

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

四卡 fully-async 可使用 `scripts/train_awm_4gpu.sh`。脚本安装本项目、OpenEnv/AWM 和 SwanLab 的 Python 依赖，校验四张可见 GPU、两个本地 checkpoint、七个本地 AWM 数据文件及 manifest，并可启动/停止本地 AWM 服务。它不会下载模型或任务数据，也不会安装或修改 Slime 的 CUDA 运行栈。完整环境变量示例见脚本开头和 README。

## 动作、奖励与实验条件

模型输出 `{"tool_name":"...","arguments":{...}}`，参数的大小写和内部空格保持原样。噪声以排序后的完整 JSON 调用为键；action-drop 在发送前发生，observation-loss 在收到结果后发生。ALFWorld 仍使用原来的字符串动作格式。

模型用 `{"tool_name":"done","arguments":{}}` 提交。项目以 `verifier_mode: code` 调用隐藏 verifier，将 complete/incomplete 映射为 1/0 后结束轨迹；不向模型开放 verify 或全局场景查询工具。预算耗尽而未提交时奖励为 0。结束提交免 action-drop；普通工具的只读分类由 manifest 决定。Verifier、工具 schema 和会话存储路径不会作为奖励提示泄露，模型仅看到任务、可用工具和工具结果。

基础设施错误直接抛出，不当作奖励 0。会话在 finally 关闭，不自动重试可能已经生效的远程写操作。默认不保留服务端数据库快照，项目 traces 继续保存动作、观察和噪声审计。

这是新任务域，不应将其成功率与 ALFWorld 直接比较。正式 matched-LOO 实验前需核对同一 scenario/task_idx 的初始数据库及 code verifier 可复现性；当前客户端不提供数据库快照一致性认证。固定服务端代码和数据版本，并对两种算法使用同一任务划分与服务配置。

协议参考：https://github.com/huggingface/OpenEnv/tree/main/envs/agent_world_model_env
