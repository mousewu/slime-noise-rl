# 本地模型与数据配置

本项目不下载模型或任务数据。训练、冻结模型预实验和独立评测都只接受显式的本地路径，并强制设置 Hugging Face/Transformers/Datasets 离线模式；缺少文件时会直接退出。

## 1. 本地模型

固定基座是 `Qwen/Qwen3-4B-Instruct-2507`（非 Thinking 版本）。需要准备两种本地表示：

1. Hugging Face checkpoint：供 tokenizer 和 SGLang rollout 使用。
2. Megatron torch-dist checkpoint：供 Slime 训练 actor/reference 权重使用。

训练还需要完整的 Megatron-LM Python 源码。若它没有通过 `pip install -e .` 安装到当前 Python 环境，在配置文件中设置 `MEGATRON_LM_DIR=/path/to/Megatron-LM`；项目会将该目录前置加入 `PYTHONPATH`，并在第 2 步验证 `megatron.training` 可导入。

Hugging Face 目录至少应包含：

```text
/models/Qwen3-4B-Instruct-2507/
  config.json
  tokenizer_config.json
  tokenizer.json
  model.safetensors.index.json
  model-00001-of-*.safetensors
  ...
```

如果已有 Hugging Face checkpoint、但还没有 Megatron checkpoint，可以做一次纯本地转换：

```bash
export SLIME_DIR=/workspace/slime
bash scripts/convert_checkpoint.sh \
  /models/Qwen3-4B-Instruct-2507 \
  /models/Qwen3-4B-Instruct-2507_torch_dist
```

转换脚本不会联网，也不会覆盖已有目标目录。

## 2. ALFWorld 数据

实验使用 ALFWorld 的 TextWorld/PDDL 文本环境。生成训练和评测 manifest 的最小数据是：

- [`json_2.1.1_json.zip`](https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip)：提供每个任务的 `traj_data.json` 元数据。
- [`json_2.1.3_tw-pddl.zip`](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip)：提供可直接执行的 `game.tw-pddl` 文本游戏；这是当前 ALFWorld 主分支指向的 0.4.2 release 版本。

将两个压缩包解压、合并到同一个数据父目录后，目标结构应为：

```bash
unzip /path/to/json_2.1.1_json.zip -d /datasets/alfworld
unzip /path/to/json_2.1.3_tw-pddl.zip -d /datasets/alfworld
```

```text
/datasets/alfworld/json_2.1.1/
  train/<task>/<trial>/traj_data.json
  train/<task>/<trial>/game.tw-pddl
  valid_seen/<task>/<trial>/traj_data.json
  valid_seen/<task>/<trial>/game.tw-pddl
  valid_unseen/<task>/<trial>/traj_data.json
  valid_unseen/<task>/<trial>/game.tw-pddl
```

本项目不需要以下 ALFWorld 资源：

- Mask R-CNN 权重和 THOR 视觉数据；
- `pretrained_checkpoints.zip`；
- `seq2seq_data.zip`；
- 单独的 `json_2.1.1_pddl.zip`（本项目直接执行已生成的 `game.tw-pddl`）。

### 用于专家 SFT 的本地轨迹

不需要额外下载 `seq2seq_data.zip` 或从 ALFRED 另行抽取动作。每个由官方 planner 生成、且 `solvable: true` 的 `game.tw-pddl` 已包含 `walkthrough`：这是该 PDDL game 的可执行专家 command 序列。本项目的 `scripts/build_alfworld_expert_sft.sh` 只读取该字段，再通过本地 TextWorld 游戏回放；回放成功后才写入一个新的 Slime SFT JSONL。因此除了上面列出的 `traj_data.json` 和 `game.tw-pddl` 外，没有新增数据下载要求。

默认应保持 `SFT_PLANNER_FALLBACK=0`。若某些游戏的 `walkthrough` 缺失、且你确认本机已有完整的 ALFWorld/TextWorld planner 依赖，可显式设为 `1`：它只在内存中查询本地 PDDL planner，不改写原始 game，但会增加 CPU/临时目录开销。无论哪种来源，构建报告都会记录各来源数量及跳过原因；训练只接受 `train` split 的记录。

不要把原始数据提交到本仓库。manifest 保存绝对路径和游戏文件 SHA256，所以应在最终训练机器上生成；如果移动了数据目录，请重新生成 manifest。

```bash
noise-rl prepare --environment alfworld \
  --root /datasets/alfworld/json_2.1.1 \
  --split train \
  --output data/alfworld/train.jsonl

noise-rl prepare --environment alfworld \
  --root /datasets/alfworld/json_2.1.1 \
  --split valid_unseen \
  --output data/alfworld/valid_unseen.jsonl
```

训练只允许 `train`；主评测建议使用 `valid_unseen`，`valid_seen` 可作为补充评测。官方数据规模是 3,553 个训练游戏、140 个 seen validation 游戏和 134 个 unseen validation 游戏；项目还会沿用官方 loader 的 solvable 筛选，并排除当前文本环境不支持的 movable/Sliced 任务。

## 3. 两卡试跑配置

复制示例并将路径改为训练服务器上的实际位置：

```bash
cp configs/local_paths.env.example configs/local_paths.env
# 编辑 configs/local_paths.env
source configs/local_paths.env
bash scripts/smoke_2gpu.sh
```

`HF_CHECKPOINT` 和 `ALFWORLD_ROOT` 是必填项。`MEGATRON_CHECKPOINT` 未设置时，默认是 `${HF_CHECKPOINT}_torch_dist`；若该目录不存在，试跑脚本会从本地 Hugging Face checkpoint 转换。试跑脚本只会创建派生的 manifest、Megatron checkpoint（需要时）和运行输出，不会下载模型或数据。

## 4. 主训练配置

训练命令直接传入本地路径：

```bash
bash scripts/train.sh \
  --config configs/matched_loo.yaml \
  --data data/alfworld/train.jsonl \
  --eval-data data/alfworld/valid_unseen.jsonl \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --megatron-checkpoint "${MEGATRON_CHECKPOINT}" \
  --output runs/matched-s42 \
  --seed 42 --dry-run
```

`--dry-run` 也会验证本地模型结构、Megatron checkpoint、manifest 中每个游戏文件的存在性和 SHA256，不会启动 GPU 训练或创建运行目录。
