# ALFWorld Noise-RL experiment plan

## Objective

Use ALFWorld as the controlled primary environment for testing whether matched
noise scenarios and LOO reward centering improve robustness, learning signal,
and stability. AWM remains the external-validity environment rather than the
source of the main causal claim.

## Compute topology

The primary recipe is synchronous colocate on eight A100s:

- actor: TP=2, DP=4, all eight GPUs;
- rollout: eight independent TP=1 SGLang engines, one per GPU;
- rollout batch: 16 tasks x 8 samples = 128 trajectories;
- global SGLang concurrency: 128, or 16 ready requests per engine;
- ALFWorld process runners: 32, increased only when
  `environment_runner_wait_seconds` is material;
- one optimizer step per rollout.

This topology is intentionally different from 2-actor + 6-rollout fully async.
Every GPU performs inference during rollout and training during the optimizer
phase, eliminating role-reservation idle time. It may still miss 70% when CPU
environment latency or the synchronous long-tail dominates, so a 20-rollout
utilization pilot is a mandatory gate.

## Experiment stages

1. **Utilization pilot**: matched LOO, seed 42, 20 rollouts. Require aggregate
   GPU mean >=70%, environment error rate near zero, and no growing runner wait.
2. **Core methods**: SFT-only evaluation, independent LOO, matched LOO, and
   coupled-prompt LOO. Run seeds 42/43/44 with identical sample budgets.
3. **Estimator ablation**: independent GRPO and matched GRPO. This separates
   scenario matching from LOO centering and standard-deviation normalization.
4. **Noise evaluation grid**: clean; action-only; observation-only;
   train-matched 0.15/0.10; stronger 0.25/0.20; and burst noise. Evaluate every
   selected checkpoint with identical tasks and seeds.
5. **Checkpoint discipline**: select checkpoints on a validation set only;
   report final numbers once on `valid_unseen`. Include early, best-validation,
   and final checkpoints to expose late collapse.

For each run report success rate, bootstrap CI, clean-to-noisy degradation,
generated/input tokens, tool calls, zero-group fraction, advantage absolute
mean/second moment, environment-error rate, actor time, rollout time, and GPU
utilization. Compare methods at equal trajectories and generated-token cost.

## Commands

Run the 20-rollout utilization gate first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SLIME_DIR=/workspace/slime \
MEGATRON_LM_DIR=/workspace/Megatron-LM \
HF_CHECKPOINT=/models/Qwen3-4B-Instruct-2507 \
MEGATRON_CHECKPOINT=/models/qwen-alfworld-sft-torch-dist \
ALFWORLD_MANIFEST=data/alfworld/train.jsonl \
CONFIG=configs/matched_loo.yaml \
SEED=42 NUM_ROLLOUT=20 SAVE_INTERVAL=10 \
BATCH_SIZE=16 CONCURRENCY=128 \
TMPDIR=/data/noise-rl-tmp NOISE_RL_RAY_TMPDIR=/tmp/nrl \
RUN_NAME=alfworld-matched-loo-colocate8-util-pilot-s42 \
bash scripts/train_alfworld_8gpu_high_util.sh
```

If the pilot is stable and reaches the utilization target, run a core method:

```bash
CONFIG=configs/matched_loo.yaml SEED=42 NUM_ROLLOUT=300 SAVE_INTERVAL=25 \
RUN_NAME=alfworld-matched-loo-colocate8-s42 \
bash scripts/train_alfworld_8gpu_high_util.sh
```

Repeat with `independent_loo.yaml`, `coupled_prompt_loo.yaml`, seeds 42/43/44,
and unique output names. Keep batch, concurrency, model initialization, and
rollout count fixed. Do not tune each method independently on `valid_unseen`.

## Utilization tuning gate

The script records a five-second `nvidia-smi` series and prints aggregate and
per-GPU means on exit. Tune in this order:

1. Start at batch 16 / concurrency 128.
2. If SGLang queues are empty but memory permits, use batch 24 / concurrency
   192. This changes the trajectories per update, so use the selected setting
   for every method and compare equal total trajectories.
3. Increase environment processes from 32 to 48 only when runner-wait time is
   non-trivial and the host has enough CPU cores and RAM.
4. If the synchronous tail remains dominant, compare a four-GPU colocated
   pilot instead of adding rollout GPUs. Do not claim equal training systems
   when comparing synchronous and fully-async results.

No configuration can guarantee 70% before measurement: ALFWorld alternates
short model generations with CPU environment steps. The utilization gate is
therefore an operational requirement, not a scientific outcome metric.
