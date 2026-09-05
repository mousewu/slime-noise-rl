import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from .advantages import group_advantages
from .agent import SGLangClient, run_episode
from .config import ExperimentConfig, NoiseConfig, load_config
from .data import alfworld_records, atomic_json, mini_records, read_records, write_records
from .fixtures import ByteTokenizer, ScriptedClient
from .metrics import balanced_variance_components, episode_record, paired_comparison, summarize
from .sampling import plan_sample


def load_episode_records(path):
    path = Path(path)
    if path.is_dir():
        values = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]
        return [v for v in values if "success" in v and "plan" in v]
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


async def run_evaluation(args):
    from transformers import AutoTokenizer

    if Path(args.output).exists():
        raise FileExistsError("Evaluation output exists; choose a new file before starting inference")
    config = load_config(args.config)
    config = replace(config, eval_seed=args.seed)
    if args.clean:
        config = replace(config, noise=NoiseConfig(action_drop=0, observation_loss=0))
    records = read_records(args.data)
    if any(r["metadata"]["task"].get("split") == "train" for r in records):
        raise ValueError("Use a held-out manifest for evaluation")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    client = SGLangClient(args.url, config.request_timeout)
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(group_id, record, rank):
        async with semaphore:
            task = record["metadata"]["task"]
            plan = plan_sample(config, task["id"], group_id, rank, evaluation=True)
            trajectory = await run_episode(
                task,
                plan,
                config,
                tokenizer,
                client,
                {"temperature": args.temperature, "top_p": 1.0, "top_k": -1},
            )
            return episode_record(trajectory, plan, config, checkpoint=args.model)

    try:
        output = await asyncio.gather(
            *(one(i, r, rank) for i, r in enumerate(records) for rank in range(args.repeats))
        )
    finally:
        await client.close()
    write_records(args.output, output)
    report = summarize(output)
    atomic_json(str(args.output) + ".summary.json", report)
    return report


async def run_demo(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    config = ExperimentConfig(method=args.method, max_turns=8, max_generated_tokens=1024)
    records = []
    plans, rewards = [], []
    for group_id, row in enumerate(mini_records(args.tasks, "fixture")):
        task = row["metadata"]["task"]
        for rank in range(config.group_size):
            plan = plan_sample(config, task["id"], group_id, rank)
            client = ScriptedClient(task, plan.policy_seed, competence=0.7)
            trajectory = await run_episode(task, plan, config, ByteTokenizer(), client)
            records.append(episode_record(trajectory, plan, config, checkpoint="SCRIPTED_FIXTURE_NOT_LLM"))
            plans.append(plan)
            rewards.append(float(trajectory.success))
    _, diagnostics = group_advantages(rewards, plans, config)
    write_records(args.output, records)
    return {
        "warning": "CPU scripted fixture only. No language model was trained or evaluated.",
        **summarize(records),
        **diagnostics,
    }


async def run_probe(args):
    """Frozen-policy nested sampling. Reports outcome variance, NOT gradient variance."""
    from transformers import AutoTokenizer

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    config = replace(
        load_config(args.config),
        method="matched",
        scenarios=args.scenarios,
        group_size=args.scenarios * args.policy_samples,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    records = read_records(args.data)[: args.tasks]
    client = SGLangClient(args.url, config.request_timeout)
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(group_id, record, rank):
        async with semaphore:
            task = record["metadata"]["task"]
            # Separate task namespace prevents accidental reuse of training streams.
            plan = plan_sample(config, "probe/" + task["id"], group_id, rank)
            trajectory = await run_episode(task, plan, config, tokenizer, client)
            row = episode_record(trajectory, plan, config, checkpoint=args.model)
            row["phase"] = "frozen_policy_probe"
            return row

    try:
        rows = await asyncio.gather(
            *(one(g, r, i) for g, r in enumerate(records) for i in range(config.group_size))
        )
    finally:
        await client.close()
    write_records(args.output, rows)
    diagnostic = {}
    for group_id, record in enumerate(records):
        block = rows[group_id * config.group_size : (group_id + 1) * config.group_size]
        matrix = [
            [float(r["success"]) for r in block[s * args.policy_samples : (s + 1) * args.policy_samples]]
            for s in range(args.scenarios)
        ]
        diagnostic[record["metadata"]["task"]["id"]] = {
            "rewards": matrix,
            **balanced_variance_components(matrix),
        }
    report = {
        "summary": summarize(rows),
        "per_task": diagnostic,
        "warning": "Frozen-policy outcome variance only; no model update or gradient variance estimate.",
    }
    atomic_json(str(args.output) + ".probe.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Slime stochastic-environment RL research tools")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create strict, non-overwriting task manifests")
    prepare.add_argument("--environment", choices=["mini", "alfworld"], required=True)
    prepare.add_argument("--root", help="ALFWorld json_2.1.1 directory")
    prepare.add_argument("--split", choices=["train", "valid_seen", "valid_unseen"], required=True)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--output", required=True)
    demo = commands.add_parser("demo", help="CPU fixture; does NOT run or train a language model")
    demo.add_argument("--method", choices=["independent", "matched", "coupled_prompt"], default="matched")
    demo.add_argument("--tasks", type=int, default=8)
    demo.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate a model served by SGLang /generate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", required=True, help="Tokenizer directory matching the served model")
    evaluate.add_argument("--url", required=True, help="SGLang URL, e.g. http://127.0.0.1:30000")
    evaluate.add_argument("--repeats", type=int, default=4)
    evaluate.add_argument("--seed", type=int, default=20260904)
    evaluate.add_argument("--temperature", type=float, default=0.8)
    evaluate.add_argument("--clean", action="store_true")
    evaluate.add_argument("--output", required=True)
    probe = commands.add_parser("probe", help="Frozen-model environment x policy nested-sampling pilot")
    probe.add_argument("--config", required=True)
    probe.add_argument("--data", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--url", required=True)
    probe.add_argument("--seed", type=int, default=71237)
    probe.add_argument("--tasks", type=int, default=16)
    probe.add_argument("--scenarios", type=int, default=16)
    probe.add_argument("--policy-samples", type=int, default=4)
    probe.add_argument("--output", required=True)
    report = commands.add_parser("summarize")
    report.add_argument("path")
    compare = commands.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    variance = commands.add_parser(
        "variance", help="ANOVA diagnostic for an environment x policy reward matrix"
    )
    variance.add_argument("matrix", help="JSON file containing a balanced list of lists")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.environment == "alfworld" and not args.root:
            parser.error("ALFWorld requires --root")
        rows = (
            alfworld_records(args.root, args.split, args.limit)
            if args.environment == "alfworld"
            else mini_records(args.limit or 32, args.split)
        )
        write_records(args.output, rows)
        result = {"tasks": len(rows), "output": str(Path(args.output).resolve())}
    elif args.command == "demo":
        result = asyncio.run(run_demo(args))
    elif args.command == "evaluate":
        if args.repeats < 1 or args.temperature < 0:
            parser.error("repeats must be positive and temperature nonnegative")
        result = asyncio.run(run_evaluation(args))
    elif args.command == "probe":
        if args.tasks < 1 or args.scenarios < 2 or args.policy_samples < 2:
            parser.error("probe requires tasks>=1, scenarios>=2, policy-samples>=2")
        result = asyncio.run(run_probe(args))
    elif args.command == "summarize":
        result = summarize(load_episode_records(args.path))
    elif args.command == "compare":
        result = paired_comparison(load_episode_records(args.left), load_episode_records(args.right))
    else:
        matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
        result = balanced_variance_components(matrix)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
