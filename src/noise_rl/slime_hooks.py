import asyncio
import copy
import logging
from dataclasses import replace
from pathlib import Path

from .advantages import group_advantages
from .agent import SGLangClient, run_episode
from .config import NoiseConfig, config_from_args
from .data import atomic_json, read_records
from .metrics import episode_record, summarize
from .sampling import SamplingPlan, plan_sample

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
        "custom_convert_samples_to_train_data_path",
    ):
        if getattr(args, option, None):
            raise ValueError(f"{option} would change group membership/advantages; disable it")
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
    return sample


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
    finally:
        await client.close()
    fill_sample(args, sample, trajectory)
    record = episode_record(trajectory, plan, config, checkpoint=metadata.get("evaluation_checkpoint"))
    sample.metadata["noise_result"] = record
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
            logger.info("noise_rl evaluation %s: %s", dataset.name, summarize(result_records))
            output[dataset.name] = {
                "rewards": [s.reward for s in samples],
                "samples": samples,
                "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
            }
        return RolloutFnEvalOutput(data=output)

    return run(evaluate())
