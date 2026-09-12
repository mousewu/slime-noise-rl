"""Audit real AWM reset prompts and create a context-safe manifest.

AWM renders each scenario's complete tool schema into the initial observation.
That size is only known after contacting the already-running local AWM server;
it cannot be inferred from the seven JSONL source files alone.  This module
uses the trainer-side WebSocket client and local tokenizer only.  It neither
imports OpenEnv nor downloads a model or task dataset.
"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import QwenChatProtocol
from .awm import AWMEnvironment, SYSTEM_PROMPT
from .data import read_records, write_records


@dataclass(frozen=True)
class ContextInspection:
    task_id: str
    scenario: str
    task_idx: int
    prompt_tokens: int | None
    error: str | None = None


def load_local_tokenizer(model: str | Path):
    """Load the exact local tokenizer without permitting hub access."""
    model_path = Path(model).expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model/tokenizer path must be a directory: {model_path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False, local_files_only=True)


async def _fetch_initial_observation(task: dict, url: str, timeout: float) -> str:
    """Reset one disposable session and return exactly the rollout's initial observation."""
    environment = AWMEnvironment(task, url, timeout=timeout)
    try:
        # ``_reset`` is the async implementation used by AWMEnvironment.reset.
        # Calling it directly avoids a thread per audit task while retaining the
        # same reset, verifier and tool-filtering semantics as rollout.
        result = await environment._reset()
        return result.observation
    finally:
        client, environment.client = environment.client, None
        if client is not None:
            await client.close()


async def inspect_awm_contexts(
    records: list[dict],
    *,
    tokenizer,
    url: str,
    timeout: float,
    concurrency: int,
    observation_fetcher=None,
) -> list[ContextInspection]:
    """Measure initial prompt tokens for AWM records, preserving input order."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    protocol = QwenChatProtocol(tokenizer)
    semaphore = asyncio.Semaphore(concurrency)

    async def inspect(record: dict) -> ContextInspection:
        task = record["metadata"]["task"]
        try:
            async with semaphore:
                if observation_fetcher is None:
                    observation = await _fetch_initial_observation(task, url, timeout)
                else:
                    observation = await observation_fetcher(task)
            _prompt, tokens = protocol.initial(observation, SYSTEM_PROMPT)
            return ContextInspection(
                task_id=task["id"],
                scenario=task["scenario"],
                task_idx=task["task_idx"],
                prompt_tokens=len(tokens),
            )
        except Exception as exc:
            return ContextInspection(
                task_id=task["id"],
                scenario=task["scenario"],
                task_idx=task["task_idx"],
                prompt_tokens=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    return list(await asyncio.gather(*(inspect(record) for record in records)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def build_context_checked_manifest(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: str | Path | None,
    url: str,
    max_context_tokens: int,
    timeout: float = 180,
    concurrency: int = 16,
    report_path: str | Path | None = None,
    tokenizer=None,
    observation_fetcher=None,
) -> dict:
    """Write a non-overwriting manifest that cannot fail the initial context check.

    Infrastructure errors are fatal rather than silently treated as excluded
    tasks.  Otherwise a temporary AWM-server failure could change the training
    distribution without the researcher noticing.
    """
    if type(max_context_tokens) is not int or max_context_tokens < 2:
        raise ValueError("max_context_tokens must be an integer of at least two")
    if type(timeout) not in {int, float} or timeout <= 0:
        raise ValueError("timeout must be positive")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")

    input_path = Path(input_path).expanduser().resolve(strict=True)
    output_path = Path(output_path).expanduser()
    report = Path(report_path).expanduser() if report_path else None
    artifacts = [output_path] + ([report] if report else [])
    if len({path.resolve() for path in artifacts}) != len(artifacts):
        raise ValueError("Output manifest and report paths must be distinct")
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    records = read_records(input_path)
    if any(record["metadata"]["task"]["environment"] != "awm" for record in records):
        raise ValueError("Context audit accepts an AWM-only manifest")
    tokenizer = tokenizer or load_local_tokenizer(model)
    inspections = asyncio.run(
        inspect_awm_contexts(
            records,
            tokenizer=tokenizer,
            url=url,
            timeout=float(timeout),
            concurrency=concurrency,
            observation_fetcher=observation_fetcher,
        )
    )
    failures = [inspection for inspection in inspections if inspection.error]
    if failures:
        examples = "; ".join(
            f"{item.task_id}: {item.error}" for item in failures[:3]
        )
        raise RuntimeError(
            f"AWM context audit failed for {len(failures)} task(s); no output was written. {examples}"
        )

    accepted = [
        record
        for record, inspection in zip(records, inspections, strict=True)
        if inspection.prompt_tokens is not None and inspection.prompt_tokens < max_context_tokens
    ]
    excluded = [
        inspection
        for inspection in inspections
        if inspection.prompt_tokens is not None and inspection.prompt_tokens >= max_context_tokens
    ]
    if not accepted:
        raise ValueError("No AWM task fits the requested initial-context budget")

    write_records(output_path, accepted)
    token_counts = [inspection.prompt_tokens for inspection in inspections if inspection.prompt_tokens is not None]
    excluded_by_scenario: dict[str, int] = {}
    for inspection in excluded:
        excluded_by_scenario[inspection.scenario] = excluded_by_scenario.get(inspection.scenario, 0) + 1
    result = {
        "schema": 1,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(output_path.resolve()),
        "model": str(Path(model).expanduser().resolve()) if model else "injected-tokenizer",
        "awm_url": url,
        "max_context_tokens": max_context_tokens,
        "acceptance_rule": "initial_prompt_tokens < max_context_tokens",
        "timeout_seconds": timeout,
        "concurrency": concurrency,
        "task_count": len(records),
        "accepted_task_count": len(accepted),
        "excluded_context_task_count": len(excluded),
        "initial_prompt_tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
        },
        "excluded_by_scenario": dict(sorted(excluded_by_scenario.items())),
        "excluded": [asdict(inspection) for inspection in excluded],
    }
    if report:
        _write_report(report, result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure real AWM reset prompts and create a context-safe local manifest"
    )
    parser.add_argument("--input", required=True, help="Existing AWM-only JSONL manifest")
    parser.add_argument("--output", required=True, help="New non-overwriting context-safe JSONL manifest")
    parser.add_argument("--model", required=True, help="Local Qwen tokenizer/checkpoint directory")
    parser.add_argument("--url", required=True, help="Existing local AWM server HTTP URL")
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--report", help="Optional non-overwriting JSON audit report")
    args = parser.parse_args(argv)
    result = build_context_checked_manifest(
        args.input,
        args.output,
        model=args.model,
        url=args.url,
        max_context_tokens=args.max_context_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
