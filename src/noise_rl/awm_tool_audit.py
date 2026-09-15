"""Dynamically confirm broken AWM business tools without model rollouts.

The AWM dataset contains generated FastAPI applications.  A lifecycle check
(``reset -> list_tools -> done``) proves that an application boots, but it
does not exercise its business tools.  This module first identifies static
routes which may be swallowed by an earlier dynamic route, then invokes the
matching MCP tool once with a small, schema-derived argument object.

The resulting JSONL is intentionally an *evidence* artifact.  A row is
marked ``confirmed_http_422`` or ``confirmed_http_500`` only when the actual
tool response contains that status code.  A schema which this deterministic
builder cannot represent is marked ``unconstructible_parameters``; it is not
silently converted into a fake successful tool call.

Each attempted call is a direct diagnostic sequence, not an LLM trajectory:
``reset -> list_tools -> call_tool -> done``.  Sessions are discarded after
the call, so writes made by a mutating diagnostic tool never alter the source
dataset or another task's initial database.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HTTP_DECORATORS = {"get", "post", "put", "patch", "delete", "head", "options"}
STATUS_PATTERN = re.compile(r"(?:status(?:\s+code)?|http)\D{0,20}(422|500)\b", re.IGNORECASE)


class ParameterConstructionError(ValueError):
    """The public MCP schema cannot be reduced to a safe diagnostic input."""


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    line: int
    handler: str
    parameter_types: dict[str, str | None]
    operation_id: str | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{number}")
            rows.append(row)
    return rows


def _write_new_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _write_new_json(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _annotation_name(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _route_from_decorator(decorator: ast.expr) -> tuple[str, str, str | None] | None:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    method = decorator.func.attr.lower()
    if method not in HTTP_DECORATORS:
        return None
    path = _string_literal(decorator.args[0]) if decorator.args else None
    if path is None:
        for keyword in decorator.keywords:
            if keyword.arg in {"path", "url"}:
                path = _string_literal(keyword.value)
                break
    operation_id = next(
        (_string_literal(keyword.value) for keyword in decorator.keywords if keyword.arg == "operation_id"), None
    )
    return (method, path, operation_id) if path else None


def _extract_routes(code: str) -> list[Route]:
    routes: list[Route] = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = {
            argument.arg: _annotation_name(argument.annotation)
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        }
        for decorator in node.decorator_list:
            route = _route_from_decorator(decorator)
            if route is not None:
                method, path, operation_id = route
                routes.append(Route(method, path, node.lineno, node.name, parameters, operation_id))
    return sorted(routes, key=lambda item: item.line)


def _path_parts(path: str) -> list[str]:
    return [part for part in path.strip("/").split("/") if part]


def _dynamic_parameter(segment: str) -> str | None:
    if segment.startswith("{") and segment.endswith("}"):
        return segment[1:-1].split(":", 1)[0].strip() or None
    return None


def _route_shadow(earlier: Route, later: Route) -> list[tuple[str, str | None]] | None:
    if earlier.method != later.method:
        return None
    earlier_parts, later_parts = _path_parts(earlier.path), _path_parts(later.path)
    if len(earlier_parts) != len(later_parts):
        return None
    captures: list[tuple[str, str | None]] = []
    for first, second in zip(earlier_parts, later_parts):
        name = _dynamic_parameter(first)
        if name is None:
            if first != second:
                return None
        else:
            captures.append((name, earlier.parameter_types.get(name)))
    return captures or None


def find_route_targets(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find every static route that an earlier dynamic FastAPI route can catch."""
    env_rows = _read_jsonl(data_dir / "gen_envs.jsonl")
    targets: list[dict[str, Any]] = []
    parse_failures = 0
    route_count = 0
    for source in env_rows:
        scenario, code = source.get("scenario"), source.get("full_code")
        if not isinstance(scenario, str) or not isinstance(code, str):
            parse_failures += 1
            continue
        try:
            routes = _extract_routes(code)
        except SyntaxError:
            parse_failures += 1
            continue
        route_count += len(routes)
        for first_index, earlier in enumerate(routes):
            for later in routes[first_index + 1 :]:
                captures = _route_shadow(earlier, later)
                if captures is None:
                    continue
                # FastAPI usually exposes explicit operation IDs as MCP tool names.
                # The handler name is retained as a transparent fallback for data
                # sources that rely on FastAPI's generated operation ID.
                candidates = list(dict.fromkeys(name for name in (later.operation_id, later.handler) if name))
                targets.append(
                    {
                        "scenario": scenario,
                        "task_idx": 0,
                        "method": earlier.method.upper(),
                        "earlier_dynamic_path": earlier.path,
                        "earlier_dynamic_line": earlier.line,
                        "later_static_path": later.path,
                        "later_static_line": later.line,
                        "later_operation_id": later.operation_id,
                        "later_handler": later.handler,
                        "tool_name_candidates": candidates,
                        "captured_path_parameters": [
                            {"name": name, "annotation": annotation} for name, annotation in captures
                        ],
                        "likely_http_422": any(annotation in {"int", "float", "UUID"} for _, annotation in captures),
                        "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    }
                )
    summary = {
        "scenarios_scanned": len(env_rows),
        "routes_scanned": route_count,
        "route_shadow_targets": len(targets),
        "parse_failures": parse_failures,
    }
    return targets, summary


def _scenario_names(data_dir: Path) -> list[str]:
    """Read one representative task per scenario for schema-only inspection."""
    rows = _read_jsonl(data_dir / "gen_tasks.jsonl")
    names: list[str] = []
    for row in rows:
        scenario, tasks = row.get("scenario"), row.get("tasks")
        if not isinstance(scenario, str) or not isinstance(tasks, list) or not tasks:
            raise ValueError(f"Invalid task catalog entry for scenario {scenario!r}")
        names.append(scenario)
    return names


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    if not reference.startswith("#/"):
        raise ParameterConstructionError(f"external JSON-schema reference is unsupported: {reference}")
    current: Any = root
    for token in reference[2:].split("/"):
        if not isinstance(current, dict) or token not in current:
            raise ParameterConstructionError(f"unresolvable JSON-schema reference: {reference}")
        current = current[token]
    if not isinstance(current, dict):
        raise ParameterConstructionError(f"JSON-schema reference is not an object: {reference}")
    merged = dict(current)
    merged.update({key: value for key, value in schema.items() if key != "$ref"})
    return merged


def _merge_all_of(parts: list[dict[str, Any]], root: dict[str, Any]) -> dict[str, Any]:
    """Merge the object-shaped ``allOf`` subset emitted by FastAPI/Pydantic."""
    merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    for part in parts:
        part = _resolve_ref(part, root)
        if part.get("type") not in {None, "object"}:
            raise ParameterConstructionError("non-object allOf cannot be safely merged")
        properties = part.get("properties", {})
        if not isinstance(properties, dict):
            raise ParameterConstructionError("allOf properties is not an object")
        merged["properties"].update(properties)
        required = part.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ParameterConstructionError("allOf required is invalid")
        merged["required"].extend(item for item in required if item not in merged["required"])
    return merged


def build_minimal_arguments(schema: dict[str, Any]) -> dict[str, Any]:
    """Build a schema-valid, deliberately tiny MCP argument object.

    Values are syntactic probes, not semantic task solutions.  For example,
    an ID of ``1`` may correctly receive a normal application-level 404; that
    is not labelled an infrastructure failure by the dynamic audit.
    """
    if not isinstance(schema, dict):
        raise ParameterConstructionError("tool input schema is not an object")
    value = _build_schema_value(schema, schema)
    if not isinstance(value, dict):
        raise ParameterConstructionError("tool input schema does not produce an argument object")
    return value


def _build_schema_value(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    schema = _resolve_ref(schema, root)
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        if not all(isinstance(item, dict) for item in all_of):
            raise ParameterConstructionError("allOf contains a non-object schema")
        return _build_schema_value(_merge_all_of(all_of, root), root)
    for union_name in ("anyOf", "oneOf"):
        alternatives = schema.get(union_name)
        if isinstance(alternatives, list):
            errors: list[str] = []
            for alternative in alternatives:
                if not isinstance(alternative, dict):
                    continue
                # Prefer a non-null branch.  An optional field can be omitted
                # by its parent; a required field needs an actual representative.
                if alternative.get("type") == "null":
                    continue
                try:
                    return _build_schema_value(alternative, root)
                except ParameterConstructionError as exc:
                    errors.append(str(exc))
            raise ParameterConstructionError(f"{union_name} has no constructible branch: {'; '.join(errors[:2])}")
    kind = schema.get("type")
    if kind is None and "properties" in schema:
        kind = "object"
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ParameterConstructionError("object schema has invalid properties or required")
        values: dict[str, Any] = {}
        for name in required:
            if not isinstance(name, str) or name not in properties or not isinstance(properties[name], dict):
                raise ParameterConstructionError(f"required parameter {name!r} has no usable schema")
            values[name] = _build_schema_value(properties[name], root)
        return values
    if kind == "array":
        items = schema.get("items", {})
        if not isinstance(items, dict):
            raise ParameterConstructionError("array items schema is invalid")
        minimum = schema.get("minItems", 0)
        if not isinstance(minimum, int) or minimum < 0:
            raise ParameterConstructionError("array minItems is invalid")
        return [_build_schema_value(items, root) for _ in range(minimum)]
    if kind == "string" or kind is None:
        minimum = schema.get("minLength", 0)
        if not isinstance(minimum, int) or minimum < 0:
            raise ParameterConstructionError("string minLength is invalid")
        return "x" * max(1, minimum)
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "number":
        return float(schema.get("minimum", 1.0))
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    raise ParameterConstructionError(f"unsupported JSON-schema type: {kind!r}")


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Keep JSONL diagnostics bounded while retaining the actual error signal."""
    if depth > 3:
        return "<nested value omitted>"
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:4000] + ("…" if len(value) > 4000 else "")
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _compact(item, depth=depth + 1) for key, item in list(value.items())[:30]}
    return repr(value)[:4000]


def _observation_details(observation: Any) -> dict[str, Any]:
    return {
        name: _compact(getattr(observation, name))
        for name in ("reward_type", "error", "warning", "tool_name", "tool_result", "steps_taken")
        if getattr(observation, name, None) is not None
    }


def _classify_tool_result(observation: Any) -> str:
    error = str(getattr(observation, "error", "") or "")
    status = STATUS_PATTERN.search(error)
    if status and status.group(1) == "422":
        return "confirmed_http_422"
    if status and status.group(1) == "500":
        return "confirmed_http_500"
    if getattr(observation, "reward_type", None) == "server_error":
        return "unclassified_server_error"
    return "call_completed"


async def _probe_one(
    target: dict[str, Any],
    *,
    url: str,
    connect_timeout: float,
    message_timeout: float,
) -> dict[str, Any]:
    # Imports remain inside this function: the static phase needs only Python's
    # standard library, while the dynamic phase must run in the AWM server venv.
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    row = {"kind": "route_shadow_tool_probe", **target, "trace": []}
    started = time.perf_counter()

    def record(action: str, **details: Any) -> None:
        row["trace"].append({"step": len(row["trace"]) + 1, "action": action, **details})

    try:
        async with AWMEnv(
            base_url=url,
            connect_timeout_s=connect_timeout,
            message_timeout_s=message_timeout,
        ) as env:
            # Keep ``done`` inside the context manager.  Its subprocess and
            # per-session SQLite cleanup must happen before AWMEnv closes its
            # websocket transport, including when one of the early checks
            # below returns.
            try:
                reset = await env.reset(scenario=target["scenario"], task_idx=target["task_idx"])
                record("reset", arguments={"scenario": target["scenario"], "task_idx": target["task_idx"]}, response=_observation_details(reset.observation))
                if getattr(reset.observation, "reward_type", None) not in {"reset_ok", "reset_warning"}:
                    row.update(status="reset_failed", excluded_from_training=True)
                    return row

                listed = await env.step(ListToolsAction())
                tools = list(getattr(listed.observation, "tools", []) or [])
                tool_map = {str(getattr(tool, "name", "")): tool for tool in tools}
                record("list_tools", tool_names=sorted(tool_map), response=_observation_details(listed.observation))
                selected_name = next((name for name in target["tool_name_candidates"] if name in tool_map), None)
                if selected_name is None:
                    row.update(
                        status="target_tool_not_discoverable",
                        excluded_from_training=True,
                        error="No route operation-id/handler candidate appeared in the MCP tool list",
                    )
                    return row

                tool = tool_map[selected_name]
                schema = getattr(tool, "input_schema", None)
                if schema is None:
                    schema = getattr(tool, "inputSchema", None)
                try:
                    arguments = build_minimal_arguments(schema)
                except ParameterConstructionError as exc:
                    row.update(
                        status="unconstructible_parameters",
                        excluded_from_training=True,
                        selected_tool=selected_name,
                        input_schema=_compact(schema),
                        error=str(exc),
                    )
                    return row

                result = await env.step(CallToolAction(tool_name=selected_name, arguments=arguments))
                response = _observation_details(result.observation)
                record("call_tool", tool_name=selected_name, arguments=arguments, response=response, done=bool(result.done))
                status = _classify_tool_result(result.observation)
                row.update(
                    status=status,
                    excluded_from_training=status in {"confirmed_http_422", "confirmed_http_500"},
                    selected_tool=selected_name,
                    arguments=arguments,
                    input_schema=_compact(schema),
                )
            finally:
                try:
                    done = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
                    record("done", arguments={"keep_session": False}, response=_observation_details(done.observation), done=bool(done.done))
                except Exception as exc:
                    record("done", arguments={"keep_session": False}, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        row.update(status="probe_exception", excluded_from_training=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        row["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return row


async def probe_targets(
    targets: list[dict[str, Any]],
    *,
    url: str,
    workers: int,
    connect_timeout: float,
    message_timeout: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(workers)

    async def bounded(target: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _probe_one(
                target,
                url=url,
                connect_timeout=connect_timeout,
                message_timeout=message_timeout,
            )

    results: list[dict[str, Any]] = []
    for index in range(0, len(targets), workers):
        batch = targets[index : index + workers]
        rows = await asyncio.gather(*(bounded(target) for target in batch))
        results.extend(rows)
        excluded = sum(bool(row.get("excluded_from_training")) for row in results)
        print(f"tool probe: {len(results)}/{len(targets)} targets; excluded={excluded}", file=sys.stderr, flush=True)
    return results


async def _audit_schemas_one(
    scenario: str,
    *,
    url: str,
    connect_timeout: float,
    message_timeout: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """List one scenario's tools and audit only public parameter schemas.

    This deliberately never calls a business tool.  It can therefore cover
    every schema without turning synthetic values into application writes or
    mistaking a missing database ID for an unconstructible parameter shape.
    """
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    failures: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    trace: list[dict[str, Any]] = []

    def record(action: str, **details: Any) -> None:
        trace.append({"step": len(trace) + 1, "action": action, **details})

    try:
        async with AWMEnv(
            base_url=url,
            connect_timeout_s=connect_timeout,
            message_timeout_s=message_timeout,
        ) as env:
            try:
                reset = await env.reset(scenario=scenario, task_idx=0)
                record("reset", arguments={"scenario": scenario, "task_idx": 0}, response=_observation_details(reset.observation))
                if getattr(reset.observation, "reward_type", None) not in {"reset_ok", "reset_warning"}:
                    counts["reset_failed"] += 1
                    failures.append(
                        {
                            "kind": "schema_construction_audit",
                            "scenario": scenario,
                            "task_idx": 0,
                            "status": "schema_enumeration_failed",
                            "excluded_from_training": False,
                            "error": "reset did not succeed",
                            "trace": trace,
                        }
                    )
                    return failures, counts
                listed = await env.step(ListToolsAction())
                tools = list(getattr(listed.observation, "tools", []) or [])
                record(
                    "list_tools",
                    tool_names=sorted(str(getattr(tool, "name", "")) for tool in tools),
                    response=_observation_details(listed.observation),
                )
                if getattr(listed.observation, "error", None):
                    counts["list_tools_failed"] += 1
                    failures.append(
                        {
                            "kind": "schema_construction_audit",
                            "scenario": scenario,
                            "task_idx": 0,
                            "status": "schema_enumeration_failed",
                            "excluded_from_training": False,
                            "error": str(getattr(listed.observation, "error")),
                            "trace": trace,
                        }
                    )
                    return failures, counts
                for tool in tools:
                    name = str(getattr(tool, "name", ""))
                    schema = getattr(tool, "input_schema", None)
                    if schema is None:
                        schema = getattr(tool, "inputSchema", None)
                    try:
                        build_minimal_arguments(schema)
                    except ParameterConstructionError as exc:
                        counts["unconstructible_parameters"] += 1
                        failures.append(
                            {
                                "kind": "schema_construction_audit",
                                "scenario": scenario,
                                "task_idx": 0,
                                "status": "unconstructible_parameters",
                                "excluded_from_training": True,
                                "selected_tool": name,
                                "input_schema": _compact(schema),
                                "error": str(exc),
                                # Keep the reference so the enclosing finally
                                # block appends the terminal ``done`` action
                                # before this evidence row is serialized.
                                "trace": trace,
                            }
                        )
                    else:
                        counts["constructible_parameters"] += 1
            finally:
                try:
                    done = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
                    record("done", arguments={"keep_session": False}, response=_observation_details(done.observation), done=bool(done.done))
                except Exception as exc:
                    record("done", arguments={"keep_session": False}, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        counts["probe_exception"] += 1
        failures.append(
            {
                "kind": "schema_construction_audit",
                "scenario": scenario,
                "task_idx": 0,
                "status": "schema_enumeration_failed",
                "excluded_from_training": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": trace,
            }
        )
    return failures, counts


async def audit_all_schemas(
    scenarios: list[str],
    *,
    url: str,
    workers: int,
    connect_timeout: float,
    message_timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semaphore = asyncio.Semaphore(workers)

    async def bounded(scenario: str) -> tuple[list[dict[str, Any]], Counter[str]]:
        async with semaphore:
            return await _audit_schemas_one(
                scenario,
                url=url,
                connect_timeout=connect_timeout,
                message_timeout=message_timeout,
            )

    failures: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for index in range(0, len(scenarios), workers):
        rows = await asyncio.gather(*(bounded(scenario) for scenario in scenarios[index : index + workers]))
        for row_failures, row_counts in rows:
            failures.extend(row_failures)
            totals.update(row_counts)
        complete = min(index + workers, len(scenarios))
        print(
            f"schema audit: {complete}/{len(scenarios)} scenarios; "
            f"unconstructible={totals['unconstructible_parameters']}",
            file=sys.stderr,
            flush=True,
        )
    summary = {
        "scenarios_requested": len(scenarios),
        "scenarios_run": len(scenarios),
        "tools_constructible": totals["constructible_parameters"],
        "tools_unconstructible": totals["unconstructible_parameters"],
        "enumeration_failures": totals["reset_failed"] + totals["list_tools_failed"] + totals["probe_exception"],
    }
    return failures, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="local AgentWorldModel-1K directory")
    parser.add_argument("--url", required=True, help="already-running local AWM server URL")
    parser.add_argument("--output", required=True, type=Path, help="new JSONL evidence output")
    parser.add_argument("--workers", type=int, default=2, help="bounded simultaneous AWM sessions")
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--message-timeout", type=float, default=120.0)
    parser.add_argument("--max-targets", type=int, help="bounded debugging subset; omit for every route-shadow target")
    parser.add_argument(
        "--schema-all",
        action="store_true",
        help="also enumerate every scenario's public tool schemas without calling business tools",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_targets is not None and args.max_targets < 1:
        raise ValueError("--max-targets must be positive")
    data_dir = args.data_dir.expanduser().resolve(strict=True)
    if not (data_dir / "gen_envs.jsonl").is_file():
        raise FileNotFoundError(f"Missing {data_dir / 'gen_envs.jsonl'}")
    output = args.output.expanduser()
    summary_path = Path(str(output) + ".summary.json")
    # Validate outputs before opening any AWM session.  The audit can make
    # temporary, session-local writes while calling a tool, so discovering an
    # existing destination only after those calls would be surprising.
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")
    targets, static_summary = find_route_targets(data_dir)
    # A duplicate can arise when multiple earlier dynamic routes shadow the
    # same operation.  One actual MCP call per scenario/tool is sufficient.
    unique: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for target in targets:
        unique.setdefault((target["scenario"], tuple(target["tool_name_candidates"])), target)
    selected = list(unique.values())
    if args.max_targets is not None:
        selected = selected[: args.max_targets]
    started = time.perf_counter()
    route_rows = asyncio.run(
        probe_targets(
            selected,
            url=args.url,
            workers=args.workers,
            connect_timeout=args.connect_timeout,
            message_timeout=args.message_timeout,
        )
    )
    schema_rows: list[dict[str, Any]] = []
    schema_summary: dict[str, Any] | None = None
    if args.schema_all:
        schema_rows, schema_summary = asyncio.run(
            audit_all_schemas(
                _scenario_names(data_dir),
                url=args.url,
                workers=args.workers,
                connect_timeout=args.connect_timeout,
                message_timeout=args.message_timeout,
            )
        )
    rows = route_rows + schema_rows
    counts = Counter(row["status"] for row in rows)
    summary = {
        "kind": "awm_route_shadow_tool_probe",
        "data_dir": str(data_dir),
        "url": args.url,
        "workers": args.workers,
        "static": static_summary,
        "unique_targets": len(unique),
        "targets_requested": len(selected),
        "targets_run": len(route_rows),
        "status_counts": dict(sorted(counts.items())),
        "scenarios_to_exclude": sorted({row["scenario"] for row in rows if row.get("excluded_from_training")}),
        "output": str(output),
        "summary": str(summary_path),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "note": "Only explicit HTTP 422/500, unavailable target tools, and schemas this builder cannot construct are marked for exclusion. Other application-level errors remain evidence, not confirmed infrastructure defects.",
    }
    if schema_summary is not None:
        summary["schema_all"] = schema_summary
    _write_new_jsonl(output, rows)
    _write_new_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
