# ALFWorld Noise RL 独立测评

`scripts/evaluate_alfworld_noise_matrix.sh` 对多个本地 HF checkpoint 依次执行相同的
`valid_unseen` 测评矩阵。脚本不下载模型或数据，也不做训练。

默认条件为：训练匹配噪声（action drop 0.15、observation loss 0.10）、clean、仅 action
drop、仅 observation loss，以及更强噪声（0.25、0.20）。设置
`EVAL_CONDITIONS=matched,clean,action-only,observation-only,strong,burst` 可额外加入 burst
条件。每项均使用相同任务、重复次数和评测 seed。

## Checkpoint 清单

复制 `configs/alfworld_eval_checkpoints.example.tsv`，每行填写：

```text
方法<TAB>训练随机种子<TAB>本地HF checkpoint绝对路径
```

可以把 4 种方法 × 3 个训练 seed 全部列入同一文件。路径必须是已经转换完成的 HF
checkpoint；本脚本不会隐式转换 Megatron 权重。

## 推荐启动方式（4 × A100 80GB）

```bash
cd /workspace/slime-noise-rl

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHON_BIN=/path/to/slime-env/bin/python \
SGLANG_PYTHON_BIN=/path/to/slime-env/bin/python \
ALFWORLD_EVAL_DATA=data/alfworld/valid_unseen.jsonl \
ALFWORLD_EVAL_CHECKPOINTS_TSV=configs/alfworld_eval_checkpoints.tsv \
ALFWORLD_EVAL_OUTPUT=runs/alfworld-noise-eval-main \
EVAL_GPUS=4 \
EVAL_REPEATS=4 \
EVAL_CONCURRENCY_PER_GPU=64 \
EVAL_ENVIRONMENT_PROCESSES_PER_GPU=16 \
EVAL_HEARTBEAT_INTERVAL_SECONDS=30 \
SGLANG_STALL_TIMEOUT_SECONDS=600 \
SGLANG_MEM_FRACTION_STATIC=0.80 \
GPU_UTILIZATION_MIN=70 \
bash scripts/evaluate_alfworld_noise_matrix.sh
```

## 单 checkpoint、单 GPU 测评

先用一个 TP=1 SGLang 实例验证 checkpoint 和 CUDA 运行时，可执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_CHECKPOINT=/absolute/path/to/local-hf-checkpoint \
ALFWORLD_EVAL_DATA=data/alfworld/valid_unseen.jsonl \
ALFWORLD_EVAL_OUTPUT=runs/alfworld-single-checkpoint-eval \
EVAL_METHOD=matched \
EVAL_TRAIN_SEED=42 \
EVAL_CONDITIONS=matched,clean,action-only,observation-only,strong \
bash scripts/evaluate_alfworld_single_checkpoint.sh
```

也可用 `EVAL_GPU_ID=3` 选择物理 GPU。该脚本不会安装或下载任何依赖、模型和数据，默认
`concurrency=64`、`environment_processes=16`，并复用矩阵脚本的心跳、SGLang stall 检测、
环境 RPC 超时和 GPU 利用率统计。单卡诊断默认不因利用率低于 70% 而中止；仍会写出
`gpu-utilization.json`。如需只验证一个条件，可设置 `EVAL_CONDITIONS=matched`。

每张 GPU 启动一个 TP=1 SGLang 实例，并加载一个不同 checkpoint；同一 wave 中四张卡分别
评估四个模型，每个模型都运行完整的 134 个任务 × 4 repeats，不再把一个小评测集切成四份。
12 个 checkpoint 会组成 3 个 wave。checkpoint 数必须能被 `EVAL_GPUS` 整除，以免最后一个
wave 留下空闲 GPU。模型加载阶段不计入脚本的条件利用率。每个 checkpoint/噪声条件都有
独立的 `gpu-utilization.json`；平均值低于门槛时脚本保留已完成结果并停止后续测评。

默认每卡推理并发为 64、环境进程为 16，避免 128 个 ALFWorld spawn worker 与四份模型共同
触发主机/cgroup OOM。若门槛未通过，
可单独将 `EVAL_CONCURRENCY_PER_GPU` 从 64 调到 96，但不要先大幅增加环境进程：当
`environment_queue_seconds` 已很低时，它只会增加 CPU 与内存压力。只有明确接受低利用率
继续完成矩阵时，才设置 `ALLOW_LOW_GPU_UTILIZATION=1`。

评测等待期间每 30 秒向主日志打印各 checkpoint 的存活状态；如果某个 evaluator 仍在运行，
但对应 SGLang 日志连续 600 秒没有任何进展，脚本会报告具体 checkpoint/condition、打印服务
日志尾部并停止该 wave。可通过 `EVAL_HEARTBEAT_INTERVAL_SECONDS` 和
`SGLANG_STALL_TIMEOUT_SECONDS` 调整。ALFWorld runner RPC 使用配置中的
`request_timeout` 作为硬超时；超时 runner 会被终止并重建，评测清理也不会再无限等待。

checkpoint 切换时，脚本会同步回收 ALFWorld 子进程，并以独立进程组关闭 SGLang 的
launcher、scheduler 和 tokenizer workers。默认将四个 SGLang 实例的启动错开 20 秒，并在
加载权重后释放文件页缓存，避免同时加载多份模型产生主机内存峰值。可用
`SGLANG_STARTUP_STAGGER_SECONDS`、`SGLANG_CLEANUP_TIMEOUT` 和
`CHECKPOINT_CLEANUP_SECONDS` 调整。若服务仍被 `SIGKILL`，主日志会同时打印 cgroup 的
`memory.events`；其中 `oom_kill` 增长即可确认是容器内存上限触发，而不是评测代码主动退出。

最终结果位于：

- `summary.tsv`：模型 × 噪声条件主表；
- `summary.json`：同一主表的结构化版本；
- `<checkpoint>/<condition>/episodes.jsonl`：完整轨迹；
- `<checkpoint>/<condition>/episodes.jsonl.summary.json`：成功率、按任务 bootstrap CI、长度与工具调用；
- `<checkpoint>/<condition>/gpu-utilization.json`：整体及逐卡利用率。
