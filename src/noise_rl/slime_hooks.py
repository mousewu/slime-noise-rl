import asyncio
import copy
import logging
from dataclasses import replace
from pathlib import Path

from .advantages import group_advantages
from .agent import SGLangAbort, SGLangClient, run_episode
from .config import NoiseConfig, config_from_args
from .data import atomic_json, read_records
from .metrics import (
    episode_record,
    is_environment_error_record,
    percentile95,
    summarize,
    trace_metrics,
)
from .sampling import SamplingPlan, plan_sample
from .swanlab_bridge import report_metrics, report_metrics_nonblocking, report_rollout_timing

logger = logging.getLogger(__name__)


def validate_slime_args(args):
    config = config_from_args(args)
    if args.n_samples_per_prompt != config.group_size:
        raise ValueError("Slime n_samples_per_prompt must equal noise_rl.group_size")
    if getattr(args, "advantage_estimator", "grpo") != "grpo":
        raise ValueError("This adapter currently supports only Slime's GRPO loss path")
    for option in ("partial_rollout", "group_rm", "normalize_advantages", "use_rollout_routing_replay"):
        if getattr(args, option, False):
            raise ValueError(f"{option} is not supported in this controlled experiment")
    for option in (
        "dynamic_sampling_filter_path",
        "rollout_sample_filter_path",
        "custom_advantage_function_path",
    ):
        if getattr(args, option, None):
            raise ValueError(f"{option} would change group membership/advantages; disable it")
    converter = getattr(args, "custom_convert_samples_to_train_data_path", None)
    expected_converter = "noise_rl.slime_hooks.convert_samples_to_windowed_train_data"
    if converter not in {None, expected_converter}:
        raise ValueError("Only the project-owned history-window converter is supported")
    if config.history_window and converter != expected_converter:
        raise ValueError(
            f"history_window requires --custom-convert-samples-to-train-data-path {expected_converter}"
        )
    if not config.history_window and converter:
        raise ValueError("Disable the history-window converter when history_window=0")
    if getattr(args, "over_sampling_batch_size", args.rollout_batch_size) != args.rollout_batch_size:
        raise ValueError("Set --over-sampling-batch-size equal to --rollout-batch-size for exact budgets")
    if not getattr(args, "rollout_global_dataset", True):
        raise ValueError("Slime's global dataset mode is required")
    return config


def fill_sample(args, sample, trajectory):
    """Use real Slime Sample APIs to preserve token/logprob/mask and metadata alignment."""
    sample.prompt = trajectory.prompt_text
    sample.tokens = list(trajectory.prompt_tokens)
    sample.response, sample.response_length = "", 0
    sample.loss_mask, sample.rollout_log_probs = [], []
    for segment in trajectory.segments:
        sample.append_response_tokens(
            args,
            tokens=segment.tokens,
            log_probs=segment.log_probs,
            text=segment.text,
            trainable=segment.trainable,
            meta_info=segment.meta_info,
            update_terminal_info=False,
        )
    if not sample.response_length or not any(sample.loss_mask):
        raise ValueError("An episode without model tokens cannot be trained")
    sample.reward = float(trajectory.success)
    complete = trajectory.termination in {"success", "environment_terminal"}
    sample.status = sample.Status.COMPLETED if complete else sample.Status.TRUNCATED
    if len(sample.tokens) - sample.response_length != len(trajectory.prompt_tokens):
        raise ValueError("Prompt/response boundary changed")
    sample._validate_response_metadata_lengths()
    if trajectory.history_window:
        sample.metadata["windowed_train_sequences"] = windowed_train_sequences(
            trajectory, trajectory.history_window
        )
    return sample


def windowed_train_sequences(trajectory, history_window):
    """Split one rollout into per-action samples with inference-identical context.

    Each result keeps the initial task/tool prompt, up to ``history_window``
    completed action-observation pairs, and the current action. Only the
    current action has a nonzero loss mask. This is the history-aware training
    construction described by AWM while preserving the original rollout as the
    GRPO normalization unit.
    """
    if type(history_window) is not int or history_window < 1:
        raise ValueError("history_window must be a positive integer")
    history = []
    sequences = []
    segments = list(trajectory.segments)
    index = 0
    while index < len(segments):
        action = segments[index]
        if not action.trainable:
            raise ValueError("Expected every interaction to start with a trainable action")
        if action.log_probs is None or len(action.log_probs) != len(action.tokens):
            raise ValueError("Windowed action tokens require aligned rollout log probabilities")
        prior = history[-history_window:]
        prior_tokens = [token for interaction in prior for token in interaction]
        tokens = list(trajectory.prompt_tokens) + prior_tokens + list(action.tokens)
        response_length = len(prior_tokens) + len(action.tokens)
        sequences.append(
            {
                "tokens": tokens,
                "response_length": response_length,
                "loss_mask": [0] * len(prior_tokens) + [1] * len(action.tokens),
                "rollout_log_probs": [0.0] * len(prior_tokens) + list(action.log_probs),
            }
        )
        index += 1
        if index < len(segments) and not segments[index].trainable:
            observation = segments[index]
            history.append(list(action.tokens) + list(observation.tokens))
            index += 1
    if not sequences:
        raise ValueError("A rollout without trainable actions cannot be windowed")
    return sequences


def convert_samples_to_windowed_train_data(args, samples):
    """Convert complete rollouts into history-windowed per-action train data."""
    config = validate_slime_args(args)
    if not samples or isinstance(samples[0], list):
        raise ValueError("Expected a flat Sample list before Slime DP sharding")
    if getattr(args, "rollout_top_p", 1.0) != 1.0:
        raise ValueError("Windowed training currently requires rollout_top_p=1")
    raw_rewards, advantages = reward_postprocess(args, samples)
    data = {
        "tokens": [],
        "response_lengths": [],
        "rewards": [],
        "raw_reward": [],
        "truncated": [],
        "sample_indices": [],
        "rollout_ids": [],
        "loss_masks": [],
        "rollout_log_probs": [],
    }
    status_type = samples[0].Status
    for position, (sample, raw_reward, advantage) in enumerate(
        zip(samples, raw_rewards, advantages, strict=True)
    ):
        sequences = sample.metadata.get("windowed_train_sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("Missing windowed train sequences on completed rollout")
        rollout_id = sample.rollout_id
        if rollout_id is None:
            rollout_id = sample.index if sample.index is not None else position
        for sequence in sequences:
            if len(sequence["tokens"]) > config.max_context_tokens:
                raise ValueError(
                    "Windowed train sequence exceeds max_context_tokens: "
                    f"{len(sequence['tokens'])} > {config.max_context_tokens}"
                )
            data["tokens"].append(sequence["tokens"])
            data["response_lengths"].append(sequence["response_length"])
            data["rewards"].append(advantage)
            data["raw_reward"].append(raw_reward)
            data["truncated"].append(1 if sample.status == status_type.TRUNCATED else 0)
            data["sample_indices"].append(sample.index)
            data["rollout_ids"].append(rollout_id)
            data["loss_masks"].append(sequence["loss_mask"])
            data["rollout_log_probs"].append(sequence["rollout_log_probs"])
    rollout_mask_totals = {}
    for rollout_id, mask in zip(data["rollout_ids"], data["loss_masks"], strict=True):
        rollout_mask_totals[rollout_id] = rollout_mask_totals.get(rollout_id, 0) + sum(mask)
    data["rollout_mask_sums"] = [rollout_mask_totals[value] for value in data["rollout_ids"]]
    lengths = [len(tokens) for tokens in data["tokens"]]
    report_metrics(
        {
            "train_tokens": sum(lengths),
            "train_sequence_length_mean": sum(lengths) / len(lengths),
            "train_sequence_length_p95": percentile95(lengths),
            "train/action_tokens": sum(sum(mask) for mask in data["loss_masks"]),
            "train/subtrajectories": len(lengths),
            "train/original_rollouts": len(samples),
        }
    )
    return data


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState

    config = validate_slime_args(args)
    metadata = sample.metadata
    plan = SamplingPlan(**metadata["noise_plan"])
    if plan.evaluation != evaluation or plan.task_id != metadata["task"]["id"]:
        raise ValueError("Plan/task/evaluation mismatch")
    if "evaluation_noise" in metadata:
        if not evaluation:
            raise ValueError("Evaluation-only noise override found in training data")
        config = replace(config, noise=NoiseConfig(**metadata["evaluation_noise"]))
    params = dict(sampling_params)
    config = replace(config, max_generated_tokens=min(config.max_generated_tokens, params["max_new_tokens"]))
    tokenizer = GenerateState(args).tokenizer
    host = args.sglang_router_ip
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    headers = None
    if getattr(args, "router_policy", None) == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}
    client = SGLangClient(f"http://{host}:{args.sglang_router_port}", config.request_timeout, headers)
    try:
        trajectory = await run_episode(metadata["task"], plan, config, tokenizer, client, params)
    except SGLangAbort as exc:
        # Slime's fully-async worker detects any ABORTED member and requeues the
        # original group.  Do not fill a partial trace, assign reward zero, or
        # retry this individual member: each would corrupt a matched LOO group.
        sample.status = sample.Status.ABORTED
        # Weight-sync aborts are expected in fully-async mode. Keep their
        # scalar counter, but never hold a requeued group behind SwanLab I/O.
        report_metrics_nonblocking(
            {
                "rollout/sglang_abort/count": 1,
                "rollout/sglang_abort/group_id": plan.group_id,
                "rollout/sglang_abort/turn": exc.diagnostics["turn"],
                "rollout/sglang_abort/in_flight_episodes": exc.diagnostics[
                    "episode"
                ]["in_flight_episodes_at_abort"],
            }
        )
        logger.info(
            "Marked sample as ABORTED for Slime full-group requeue: group=%s rank=%s",
            plan.group_id,
            plan.rank,
        )
        return sample
    finally:
        await client.close()
    fill_sample(args, sample, trajectory)
    record = episode_record(trajectory, plan, config, checkpoint=metadata.get("evaluation_checkpoint"))
    sample.metadata["noise_result"] = record
    if not evaluation:
        # Fully-async Slime can decouple completed trajectories from the later
        # reward-postprocess callback.  Emit a bounded, nonblocking timing
        # summary here so ALFWorld runner lease waits always reach SwanLab.
        report_rollout_timing(record, group_id=plan.group_id, group_size=config.group_size)
    if config.trace_dir:
        domain = f"eval_{metadata.get('evaluation_checkpoint', 0)}" if evaluation else "train"
        dataset = str(metadata.get("evaluation_dataset", "train"))
        from .sampling import stable_seed

        name = f"{stable_seed(dataset, plan.task_id):016x}_{plan.group_id}_{plan.rank}.json"
        atomic_json(Path(config.trace_dir) / domain / name, record)
    return sample


def reward_postprocess(args, samples):
    config = validate_slime_args(args)
    if not samples or isinstance(samples[0], list):
        raise ValueError("Expected flat Sample list before Slime DP sharding")
    plans = [SamplingPlan(**s.metadata["noise_plan"]) for s in samples]
    rewards = [float(s.reward) for s in samples]
    if any(s.remove_sample for s in samples):
        raise ValueError("Do not drop individual members of a comparison group")
    normalized, statistics = group_advantages(rewards, plans, config)
    logger.info("noise_rl advantages: %s", statistics)
    records = [sample.metadata.get("noise_result") for sample in samples]
    if all(isinstance(record, dict) for record in records):
        rollout_id = max(plan.group_id for plan in plans)
        metrics = trace_metrics(records, prefix="rollout", step=rollout_id)
        metrics.update(
            {
                "rollout/reward/mean": sum(rewards) / len(rewards),
                "rollout/reward/total": sum(rewards),
                "rollout/advantages/mean": sum(normalized) / len(normalized),
                "rollout/advantages/abs_mean": sum(abs(value) for value in normalized) / len(normalized),
                "rollout/advantages/zero_group_fraction": statistics[
                    "advantage_zero_group_fraction"
                ],
                "rollout/advantages/second_moment": statistics[
                    "advantage_second_moment"
                ],
                "rollout/groups": len({plan.group_id for plan in plans}),
                "environment_error_rate": sum(
                    float(is_environment_error_record(record)) for record in records
                ) / len(records),
            }
        )
        logger.info("noise_rl rollout_metrics: %s", metrics)
        report_metrics(metrics)
        if config.trace_dir:
            atomic_json(Path(config.trace_dir) / "metrics" / f"rollout_{rollout_id}.json", metrics)
    if config.trace_dir:
        atomic_json(
            Path(config.trace_dir) / "advantages" / f"group_{min(p.group_id for p in plans)}.json",
            {
                "raw_rewards": rewards,
                "advantages": normalized,
                "plans": [p.to_dict() for p in plans],
                "statistics": statistics,
            },
        )
    # Slime consumes the second array as its scalar GRPO return; no second normalization.
    return rewards, normalized


def evaluate_rollout(args, rollout_id, data_source, evaluation=False):
    """Independent evaluation seeds, fixed across checkpoints; no training group normalization."""
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.rollout.sglang_rollout import generate_and_rm
    from slime.utils.async_utils import run
    from slime.utils.types import Sample

    if not evaluation:
        raise ValueError("Use this hook only with --eval-function-path")
    config = validate_slime_args(args)

    async def evaluate():
        output = {}
        for dataset in args.eval_datasets:
            records = read_records(dataset.path)
            tasks = []
            for group_id, record in enumerate(records):
                if record["metadata"]["task"].get("split") == "train":
                    raise ValueError("Evaluation manifest must not contain training tasks")
                for rank in range(dataset.n_samples_per_eval_prompt):
                    metadata = copy.deepcopy(record["metadata"])
                    plan = plan_sample(config, metadata["task"]["id"], group_id, rank, evaluation=True)
                    metadata.update(
                        noise_plan=plan.to_dict(),
                        evaluation_checkpoint=rollout_id,
                        evaluation_dataset=dataset.name,
                    )
                    sample = Sample(
                        prompt=record.get("prompt", ""),
                        metadata=metadata,
                        group_index=group_id,
                        index=group_id * dataset.n_samples_per_eval_prompt + rank,
                    )
                    params = {
                        "temperature": dataset.temperature,
                        "top_p": dataset.top_p,
                        "top_k": dataset.top_k,
                        "max_new_tokens": dataset.max_response_len,
                    }
                    tasks.append(generate_and_rm(args, sample, params, evaluation=True))
            samples = await asyncio.gather(*tasks)
            result_records = [s.metadata["noise_result"] for s in samples]
            summary = summarize(result_records)
            metrics = trace_metrics(result_records, prefix=f"eval/{dataset.name}", step=rollout_id)
            logger.info("noise_rl evaluation %s: %s", dataset.name, summary)
            logger.info("noise_rl evaluation_metrics: %s", metrics)
            report_metrics(metrics)
            if config.trace_dir:
                atomic_json(Path(config.trace_dir) / "metrics" / f"eval_{rollout_id}_{dataset.name}.json", metrics)
            output[dataset.name] = {
                "rewards": [s.reward for s in samples],
                "samples": samples,
                "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
            }
        return RolloutFnEvalOutput(data=output)

    return run(evaluate())
