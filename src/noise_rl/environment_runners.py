"""Process-isolated, stateful environment runners for unsafe TextWorld backends.

TextWorld's parser is not safe to share between Python threads.  A runner owns
one live environment at a time in its own OS process; a parent-side proxy keeps
that runner leased for the whole trajectory.  Consequently an episode remains
strictly sequential while separate episodes can execute their steps in parallel.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Lock

from .envs import StepResult, make_environment


# These globals exist only in a runner child process.  Environment instances
# are never serialized back to the rollout worker.
_RUNNER_ENVIRONMENTS: dict[str, object] = {}


def _runner_initializer() -> None:
    """Ensure CPU-only TextWorld children never initialize a rollout GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _runner_create(episode_id: str, task: dict) -> None:
    if episode_id in _RUNNER_ENVIRONMENTS:
        raise RuntimeError(f"Environment runner already owns episode {episode_id}")
    _RUNNER_ENVIRONMENTS[episode_id] = make_environment(task)


def _runner_reset(episode_id: str) -> StepResult:
    return _runner_environment(episode_id).reset()


def _runner_step(episode_id: str, action: str) -> StepResult:
    return _runner_environment(episode_id).step(action)


def _runner_close(episode_id: str) -> None:
    environment = _RUNNER_ENVIRONMENTS.pop(episode_id, None)
    if environment is not None:
        environment.close()


def _runner_environment(episode_id: str):
    try:
        return _RUNNER_ENVIRONMENTS[episode_id]
    except KeyError as exc:
        raise RuntimeError(f"Unknown or closed environment episode {episode_id}") from exc


@dataclass(frozen=True)
class _RunnerSlot:
    index: int
    executor: ProcessPoolExecutor


class ProcessIsolatedALFWorldEnvironment:
    """A synchronous TextEnvironment proxy whose state stays in one child process."""

    def __init__(self, pool: "ProcessEnvironmentPool", slot: _RunnerSlot, episode_id: str):
        self._pool = pool
        self._slot = slot
        self._episode_id = episode_id
        self._released = False

    @property
    def runner_index(self) -> int:
        """Stable slot identifier for diagnostics; it has no task semantics."""
        return self._slot.index

    @property
    def runner_count(self) -> int:
        return self._pool.processes

    def _submit(self, function, *args):
        if self._released:
            raise RuntimeError("Cannot use a released environment runner")
        return self._slot.executor.submit(function, self._episode_id, *args)

    async def _await(self, function, *args):
        return await asyncio.wrap_future(self._submit(function, *args))

    async def acreate(self) -> None:
        await self._await(_runner_create, self._pool.task_for(self._episode_id))

    def reset(self) -> StepResult:
        return self._submit(_runner_reset).result()

    def step(self, action: str) -> StepResult:
        return self._submit(_runner_step, action).result()

    def is_read_only(self, action: str) -> bool:
        # This is the exact ALFWorld implementation rule.  Keeping it in the
        # parent avoids a second RPC before every genuine environment action.
        return action in {"look", "inventory"} or action.startswith("examine ")

    def close(self) -> None:
        if self._released:
            return
        try:
            self._submit(_runner_close).result()
        finally:
            self._release()

    async def aclose(self) -> None:
        if self._released:
            return
        try:
            await self._await(_runner_close)
        finally:
            self._release()

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._pool.release(self._slot, self._episode_id)


class ProcessEnvironmentPool:
    """A bounded pool of one-environment-per-process TextWorld runners.

    ``ProcessPoolExecutor`` normally load-balances independent calls, which is
    wrong for a stateful text game.  Instead, each slot owns a separate
    one-worker executor.  Acquiring a slot leases it to exactly one trajectory
    until ``close`` returns it to the pool.
    """

    def __init__(self, processes: int):
        if type(processes) is not int or processes < 1:
            raise ValueError("processes must be a positive integer")
        self.processes = processes
        context = mp.get_context("spawn")
        self._slots = tuple(
            _RunnerSlot(
                index=index,
                executor=ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                    initializer=_runner_initializer,
                ),
            )
            for index in range(processes)
        )
        self._available: Queue[_RunnerSlot] = Queue(maxsize=processes)
        for slot in self._slots:
            self._available.put(slot)
        # Waiting for a free lease must not block the asyncio rollout loop.
        # This executor never executes environment work, so queued waiters
        # cannot deadlock runner cleanup or RPC dispatch.
        self._wait_executor = ThreadPoolExecutor(
            max_workers=processes, thread_name_prefix="noise-rl-env-lease"
        )
        self._tasks: dict[str, dict] = {}
        self._task_lock = Lock()
        self._closed = False

    async def open(self, task: dict) -> tuple[ProcessIsolatedALFWorldEnvironment, float]:
        if self._closed:
            raise RuntimeError("Environment runner pool is closed")
        loop = asyncio.get_running_loop()
        started = loop.time()
        slot = await loop.run_in_executor(self._wait_executor, self._available.get)
        wait_seconds = loop.time() - started
        episode_id = uuid.uuid4().hex
        with self._task_lock:
            self._tasks[episode_id] = dict(task)
        environment = ProcessIsolatedALFWorldEnvironment(self, slot, episode_id)
        try:
            await environment.acreate()
        except BaseException:
            environment._release()
            raise
        return environment, wait_seconds

    def task_for(self, episode_id: str) -> dict:
        with self._task_lock:
            try:
                return self._tasks[episode_id]
            except KeyError as exc:
                raise RuntimeError(f"Unknown environment episode {episode_id}") from exc

    def release(self, slot: _RunnerSlot, episode_id: str) -> None:
        with self._task_lock:
            self._tasks.pop(episode_id, None)
        self._available.put(slot)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wait_executor.shutdown(wait=False, cancel_futures=True)
        for slot in self._slots:
            slot.executor.shutdown(wait=False, cancel_futures=True)


_POOLS: dict[int, ProcessEnvironmentPool] = {}
_POOLS_LOCK = Lock()


def get_process_environment_pool(processes: int) -> ProcessEnvironmentPool:
    """Return a rollout-worker-local pool with the requested fixed capacity."""
    with _POOLS_LOCK:
        pool = _POOLS.get(processes)
        if pool is None:
            pool = ProcessEnvironmentPool(processes)
            _POOLS[processes] = pool
        return pool


def shutdown_process_environment_pools() -> None:
    """Best-effort cleanup for normal rollout-worker shutdown."""
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.shutdown()


atexit.register(shutdown_process_environment_pools)
