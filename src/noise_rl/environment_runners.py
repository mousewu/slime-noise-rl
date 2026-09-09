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
from dataclasses import dataclass, field
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


@dataclass
class _RunnerSlot:
    index: int
    executor: ProcessPoolExecutor
    completed_leases: int = 0
    restarts: int = 0
    lock: Lock = field(default_factory=Lock)


class ProcessIsolatedALFWorldEnvironment:
    """A synchronous TextEnvironment proxy whose state stays in one child process."""

    def __init__(self, pool: "ProcessEnvironmentPool", slot: _RunnerSlot, episode_id: str):
        self._pool = pool
        self._slot = slot
        self._episode_id = episode_id
        self._released = False
        self._recycled_on_release = False

    @property
    def runner_index(self) -> int:
        """Stable slot identifier for diagnostics; it has no task semantics."""
        return self._slot.index

    @property
    def runner_count(self) -> int:
        return self._pool.processes

    @property
    def recycled_on_release(self) -> bool:
        """Whether closing this lease replaced its now-idle child process."""
        return self._recycled_on_release

    @property
    def runner_restarts_total(self) -> int:
        return self._pool.statistics()["restarts_total"]

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
        force_recycle = False
        try:
            self._submit(_runner_close).result()
        except BaseException:
            force_recycle = True
            raise
        finally:
            self._release(force_recycle=force_recycle)

    async def aclose(self) -> None:
        if self._released:
            return
        force_recycle = False
        try:
            await self._await(_runner_close)
        except BaseException:
            force_recycle = True
            raise
        finally:
            self._release(force_recycle=force_recycle)

    def _release(self, *, force_recycle: bool = False) -> None:
        if not self._released:
            self._released = True
            self._recycled_on_release = self._pool.release(
                self._slot, self._episode_id, force_recycle=force_recycle
            )


class ProcessEnvironmentPool:
    """A bounded pool of one-environment-per-process TextWorld runners.

    ``ProcessPoolExecutor`` normally load-balances independent calls, which is
    wrong for a stateful text game.  Instead, each slot owns a separate
    one-worker executor.  Acquiring a slot leases it to exactly one trajectory
    until ``close`` returns it to the pool.
    """

    def __init__(self, processes: int, recycle_episodes: int = 32):
        if type(processes) is not int or processes < 1:
            raise ValueError("processes must be a positive integer")
        if type(recycle_episodes) is not int or recycle_episodes < 0:
            raise ValueError("recycle_episodes must be a nonnegative integer")
        self.processes = processes
        self.recycle_episodes = recycle_episodes
        context = mp.get_context("spawn")
        self._mp_context = context
        self._slots = tuple(
            _RunnerSlot(
                index=index,
                executor=self._new_executor(),
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
        self._statistics_lock = Lock()
        self._completed_leases_total = 0
        self._restarts_total = 0
        self._closed = False

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=self._mp_context,
            initializer=_runner_initializer,
        )

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
            # A failed child initialization can itself leave parser/native
            # state behind; do not reuse that process for an unrelated task.
            environment._release(force_recycle=True)
            raise
        return environment, wait_seconds

    def task_for(self, episode_id: str) -> dict:
        with self._task_lock:
            try:
                return self._tasks[episode_id]
            except KeyError as exc:
                raise RuntimeError(f"Unknown environment episode {episode_id}") from exc

    def release(self, slot: _RunnerSlot, episode_id: str, *, force_recycle: bool = False) -> bool:
        """Return an idle slot and periodically replace its child process.

        A slot is leased to exactly one trajectory, and ``_runner_close`` has
        completed before this method is called.  Replacing the executor here
        therefore cannot interrupt an active TextWorld environment or change
        episode semantics.  ``shutdown(wait=False)`` lets the already-idle old
        child exit while the replacement starts lazily on its next lease.
        """
        with self._task_lock:
            self._tasks.pop(episode_id, None)
        old_executor = None
        recycled = False
        with slot.lock:
            slot.completed_leases += 1
            should_recycle = force_recycle or (
                self.recycle_episodes > 0 and slot.completed_leases >= self.recycle_episodes
            )
            if should_recycle:
                old_executor = slot.executor
                slot.executor = self._new_executor()
                slot.completed_leases = 0
                slot.restarts += 1
                recycled = True
        with self._statistics_lock:
            self._completed_leases_total += 1
            if recycled:
                self._restarts_total += 1
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        if self._closed:
            # The replacement is unused when the parent is already shutting
            # down.  Do not put the slot back into a pool nobody can acquire.
            if recycled:
                slot.executor.shutdown(wait=False, cancel_futures=True)
            return recycled
        self._available.put(slot)
        return recycled

    def statistics(self) -> dict[str, int]:
        """Return cheap counters suitable for traces and scalar telemetry."""
        with self._statistics_lock:
            return {
                "processes": self.processes,
                "recycle_episodes": self.recycle_episodes,
                "completed_leases_total": self._completed_leases_total,
                "restarts_total": self._restarts_total,
            }

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wait_executor.shutdown(wait=False, cancel_futures=True)
        for slot in self._slots:
            slot.executor.shutdown(wait=False, cancel_futures=True)


_POOLS: dict[tuple[int, int], ProcessEnvironmentPool] = {}
_POOLS_LOCK = Lock()


def get_process_environment_pool(processes: int, recycle_episodes: int = 32) -> ProcessEnvironmentPool:
    """Return a rollout-worker-local pool with the requested fixed capacity."""
    key = (processes, recycle_episodes)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = ProcessEnvironmentPool(processes, recycle_episodes)
            _POOLS[key] = pool
        return pool


def shutdown_process_environment_pools() -> None:
    """Best-effort cleanup for normal rollout-worker shutdown."""
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.shutdown()


atexit.register(shutdown_process_environment_pools)
