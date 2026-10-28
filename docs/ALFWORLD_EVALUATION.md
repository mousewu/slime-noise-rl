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
EVAL_CONCURRENCY_PER_GPU=48 \
EVAL_ENVIRONMENT_PROCESSES_PER_GPU=24 \
SGLANG_MEM_FRACTION_STATIC=0.80 \
GPU_UTILIZATION_MIN=70 \
bash scripts/evaluate_alfworld_noise_matrix.sh
```

每张 GPU 启动一个 TP=1 SGLang 实例；134 个任务稳定分成 4 个互斥分片。模型加载阶段
不计入利用率。每个 checkpoint/噪声条件都有独立的 `gpu-utilization.json`；平均值低于
70% 时脚本保留已完成结果并停止后续测评。这是运行时门槛，而不是仅凭参数预测利用率。

若门槛未通过，先将 `EVAL_GPUS` 降为 2（同时只暴露两张卡），使单卡获得更多轨迹；其次
把 `EVAL_CONCURRENCY_PER_GPU` 从 48 调到 64。不要先增加环境进程：当
`environment_queue_seconds` 已很低时，它只会增加 CPU 与内存压力。只有明确接受低利用率
继续完成矩阵时，才设置 `ALLOW_LOW_GPU_UTILIZATION=1`。

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
