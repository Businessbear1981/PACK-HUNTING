"""
ArtifactStore — the shared environment every stigmergy agent reads and
writes. This is the center of gravity of the whole system: agents hold no
state and never call each other, so whatever safety property the system has
comes entirely from what this file guarantees.

CONCURRENCY MODEL (see docs/reference/stigmergy-research.md for the full
writeup)
    Every artifact is append-only: write() never overwrites a version, it
    adds a new one under the same key. That single mechanism implements both
    patterns the spec requires:

    1. Optimistic concurrency — pass `expected_version`. If the key's current
       head isn't at that version, the write is rejected with
       WriteOutcome.CONFLICT (and, if the caller asked, a ConflictRecord is
       deposited into the store for a Critic/MetaController to resolve
       later) instead of silently clobbering the other writer.
    2. Append-only / event-sourced — pass no `expected_version`. The new
       version is always accepted as another proposal. A Synthesizer reads
       the full version history (`read_all_versions`) to reconcile multiple
       agents' proposals into a canonical view. This is exactly how the
       swarm mode is safe: concurrent Scouts/Analysts writing distinct or
       overlapping keys never corrupt each other, they just accumulate.

    Conflict resolution strategy is explicit, not last-write-wins: a losing
    write is told it lost and gets the winner's version back so it can
    decide to retry (re-read, reapply, write again) or deposit a CONFLICT
    artifact. `write_with_retry` below implements the retry half; agents
    that want the "loud" behavior call `write(..., on_conflict="deposit")`.

GRACEFUL DEGRADATION
    FileArtifactStore is backed by NEST's existing StateStore (atomic
    temp-file + os.replace, same posture as ORCA/The Field). If the durable
    backend stops accepting writes, it degrades to in-memory and every write
    result says so via WriteOutcome.DEGRADED — never a silent success that
    quietly stops persisting.

WHAT THIS DOES NOT COVER (documented, not hidden)
    True cross-process mutual exclusion. The per-store lock is
    process-local. Two separate OS processes hammering the same
    FileArtifactStore file can still race between "load" and "atomic
    replace" — the same boundary StateStore itself documents. Multi-process
    fan-out needs a real broker (Redis/Postgres) behind the same
    ArtifactStore interface; see the docs for the swap-in contract.
"""
from __future__ import annotations

import abc
import asyncio
import itertools
import logging
import pathlib
import threading
from collections import defaultdict
from typing import Callable, Literal, Optional

from contracts.stigmergy_contracts import (
    ArtifactRecord,
    ConflictRecord,
    StoreQuery,
    WriteOutcome,
    WriteRequest,
    WriteResult,
)
from services.state_store import StateStore

logger = logging.getLogger(__name__)

OnConflict = Literal["reject", "deposit"]


class ArtifactStore(abc.ABC):
    """
    Interface every backend (in-memory, file, and — not implemented here —
    Redis/Postgres/vector-DB) must satisfy. Swapping the backend means
    writing one class against this ABC; nothing above this layer changes.
    """

    @abc.abstractmethod
    async def write(self, request: WriteRequest, *, on_conflict: OnConflict = "reject") -> WriteResult: ...

    @abc.abstractmethod
    async def read(self, key: str) -> Optional[ArtifactRecord]:
        """Latest version of `key`, or None if it has never been written."""

    @abc.abstractmethod
    async def read_all_versions(self, key: str) -> list[ArtifactRecord]: ...

    @abc.abstractmethod
    async def query(self, q: StoreQuery) -> list[ArtifactRecord]: ...

    @abc.abstractmethod
    async def audit_log(self) -> list[ArtifactRecord]:
        """Every version of every artifact, in write order. The full,
        auditable trail: who wrote what, when, at what confidence, off what
        parents, and why."""

    @abc.abstractmethod
    async def conflicts(self, *, unresolved_only: bool = True) -> list[ConflictRecord]: ...

    @abc.abstractmethod
    def status(self) -> dict:
        """Health/degradation signal — never raises."""

    async def write_with_retry(
        self,
        key: str,
        build_request: "Callable[[Optional[ArtifactRecord]], WriteRequest]",
        *,
        max_attempts: int = 5,
    ) -> WriteResult:
        """
        Optimistic-concurrency retry helper: re-read `key`'s current head,
        let the caller rebuild its write against that head (or against None
        if the key has never been written), attempt the write, and retry on
        CONFLICT up to `max_attempts`. This is the "retry" half of the
        required optimistic-concurrency pattern; the "deposit a conflict
        artifact" half is on_conflict="deposit" in write() directly.
        """
        last: Optional[WriteResult] = None
        for _ in range(max_attempts):
            current = await self.read(key)
            request = build_request(current)
            last = await self.write(request, on_conflict="reject")
            if last.outcome != WriteOutcome.CONFLICT:
                return last
        assert last is not None
        return last


class InMemoryArtifactStore(ArtifactStore):
    """
    Pure in-memory backend. No persistence, no external dependency — the
    default for tests and for swarm demos that don't need durability.
    Concurrency-safe via a single asyncio.Lock guarding read-check-append,
    which is all "safe" needs to mean for an in-process store: every writer
    is a coroutine on the same event loop, and the critical section is a
    dict append, not I/O.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._versions: dict[str, list[ArtifactRecord]] = defaultdict(list)
        self._log: list[ArtifactRecord] = []
        self._conflicts: list[ConflictRecord] = []
        self._seq = itertools.count(1)

    async def write(self, request: WriteRequest, *, on_conflict: OnConflict = "reject") -> WriteResult:
        # asyncio.Lock.acquire() returns without a real suspension when
        # uncontended, and every method here is pure CPU (no I/O). Without
        # an explicit yield, a fully-synchronous workload (heuristic
        # provider + this backend) never actually interleaves: one worker's
        # while-loop can run to completion before asyncio.gather ever
        # schedules the others, silently downgrading "swarm" to sequential.
        # This checkpoint is what makes concurrent workers genuinely
        # concurrent instead of accidentally serialized.
        await asyncio.sleep(0)
        async with self._lock:
            history = self._versions[request.key]
            current_version = history[-1].version if history else 0

            if request.expected_version is not None and request.expected_version != current_version:
                conflict = ConflictRecord(
                    key=request.key,
                    attempted_by=request.agent_id,
                    attempted_role=request.role,
                    expected_version=request.expected_version,
                    actual_version=current_version,
                    attempted_content=request.content,
                )
                if on_conflict == "deposit":
                    self._conflicts.append(conflict)
                return WriteResult(
                    outcome=WriteOutcome.CONFLICT,
                    conflict=conflict,
                    note=f"expected version {request.expected_version}, actual is {current_version}",
                )

            record = ArtifactRecord(
                key=request.key,
                kind=request.kind,
                version=current_version + 1,
                round=request.round,
                role=request.role,
                agent_id=request.agent_id,
                content=request.content,
                confidence=request.confidence,
                rationale=request.rationale,
                parent_versions=request.parent_versions,
            )
            history.append(record)
            self._log.append(record)
            return WriteResult(outcome=WriteOutcome.OK, record=record)

    async def read(self, key: str) -> Optional[ArtifactRecord]:
        async with self._lock:
            history = self._versions.get(key)
            return history[-1] if history else None

    async def read_all_versions(self, key: str) -> list[ArtifactRecord]:
        async with self._lock:
            return list(self._versions.get(key, ()))

    async def query(self, q: StoreQuery) -> list[ArtifactRecord]:
        async with self._lock:
            pool = self._log if not q.only_latest else [h[-1] for h in self._versions.values() if h]
            return [r for r in pool if _matches(r, q)]

    async def audit_log(self) -> list[ArtifactRecord]:
        async with self._lock:
            return list(self._log)

    async def conflicts(self, *, unresolved_only: bool = True) -> list[ConflictRecord]:
        async with self._lock:
            if unresolved_only:
                return [c for c in self._conflicts if not c.resolved]
            return list(self._conflicts)

    def status(self) -> dict:
        return {
            "backend": "in_memory",
            "durable": False,
            "keys": len(self._versions),
            "total_versions": len(self._log),
            "note": "Not persisted. Lost on process exit — expected for this backend.",
        }


def _matches(record: ArtifactRecord, q: StoreQuery) -> bool:
    if q.kinds is not None and record.kind not in q.kinds:
        return False
    if q.roles is not None and record.role not in q.roles:
        return False
    if q.round_eq is not None and record.round != q.round_eq:
        return False
    if q.round_lte is not None and record.round > q.round_lte:
        return False
    if q.key_prefix is not None and not record.key.startswith(q.key_prefix):
        return False
    return True


class FileArtifactStore(ArtifactStore):
    """
    Durable backend on top of NEST's existing StateStore (atomic
    temp-file + os.replace, same file as ORCA/The Field use). One JSON
    namespace holds every artifact's full version history plus the
    conflict log, keyed by artifact key.

    A threading.Lock (not asyncio.Lock) guards the critical section because
    StateStore itself is synchronous file I/O; the async methods below run
    that section in a worker thread via asyncio.to_thread so a slow disk
    never blocks the event loop other agents are running on.
    """

    def __init__(self, namespace: str = "stigmergy_artifacts", *, directory: Optional[pathlib.Path] = None) -> None:
        self._namespace = namespace
        self._state = StateStore(namespace, directory=directory)
        self._lock = threading.Lock()
        self._degraded_logged = False

    # ---- sync core, run off the event loop -------------------------------

    def _write_sync(self, request: WriteRequest, on_conflict: OnConflict) -> WriteResult:
        with self._lock:
            history_raw = self._state.get(f"key:{request.key}", [])
            history = [ArtifactRecord.model_validate(r, strict=False) for r in history_raw]
            current_version = history[-1].version if history else 0

            if request.expected_version is not None and request.expected_version != current_version:
                conflict = ConflictRecord(
                    key=request.key,
                    attempted_by=request.agent_id,
                    attempted_role=request.role,
                    expected_version=request.expected_version,
                    actual_version=current_version,
                    attempted_content=request.content,
                )
                if on_conflict == "deposit":
                    conflicts_raw = self._state.get("conflicts", [])
                    conflicts_raw.append(conflict.model_dump(mode="json"))
                    self._state.put("conflicts", conflicts_raw)
                return WriteResult(
                    outcome=WriteOutcome.CONFLICT,
                    conflict=conflict,
                    note=f"expected version {request.expected_version}, actual is {current_version}",
                )

            record = ArtifactRecord(
                key=request.key,
                kind=request.kind,
                version=current_version + 1,
                round=request.round,
                role=request.role,
                agent_id=request.agent_id,
                content=request.content,
                confidence=request.confidence,
                rationale=request.rationale,
                parent_versions=request.parent_versions,
            )
            history_raw.append(record.model_dump(mode="json"))
            self._state.put(f"key:{request.key}", history_raw)

            index = set(self._state.get("_index", []))
            index.add(request.key)
            self._state.put("_index", sorted(index))

            outcome = WriteOutcome.OK
            note = None
            status = self._state.status()
            if not status["durable"]:
                outcome = WriteOutcome.DEGRADED
                note = status["note"]
                if not self._degraded_logged:
                    logger.error("stigmergy_store_degraded", extra={"namespace": self._namespace, "reason": status["write_error"]})
                    self._degraded_logged = True

            return WriteResult(outcome=outcome, record=record, note=note)

    def _read_all_sync(self, key: str) -> list[ArtifactRecord]:
        with self._lock:
            raw = self._state.get(f"key:{key}", [])
            return [ArtifactRecord.model_validate(r, strict=False) for r in raw]

    def _query_sync(self, q: StoreQuery) -> list[ArtifactRecord]:
        with self._lock:
            index = self._state.get("_index", [])
            keys = [k for k in index if q.key_prefix is None or k.startswith(q.key_prefix)]
            pool: list[ArtifactRecord] = []
            for key in keys:
                history = [ArtifactRecord.model_validate(r, strict=False) for r in self._state.get(f"key:{key}", [])]
                if not history:
                    continue
                pool.extend([history[-1]] if q.only_latest else history)
            return [r for r in pool if _matches(r, q)]

    def _audit_log_sync(self) -> list[ArtifactRecord]:
        with self._lock:
            index = self._state.get("_index", [])
            entries: list[ArtifactRecord] = []
            for key in index:
                entries.extend(ArtifactRecord.model_validate(r, strict=False) for r in self._state.get(f"key:{key}", []))
            entries.sort(key=lambda r: r.written_at)
            return entries

    def _conflicts_sync(self, unresolved_only: bool) -> list[ConflictRecord]:
        with self._lock:
            raw = self._state.get("conflicts", [])
            records = [ConflictRecord.model_validate(r, strict=False) for r in raw]
            return [c for c in records if not c.resolved] if unresolved_only else records

    # ---- async facade -----------------------------------------------------

    async def write(self, request: WriteRequest, *, on_conflict: OnConflict = "reject") -> WriteResult:
        return await asyncio.to_thread(self._write_sync, request, on_conflict)

    async def read(self, key: str) -> Optional[ArtifactRecord]:
        history = await asyncio.to_thread(self._read_all_sync, key)
        return history[-1] if history else None

    async def read_all_versions(self, key: str) -> list[ArtifactRecord]:
        return await asyncio.to_thread(self._read_all_sync, key)

    async def query(self, q: StoreQuery) -> list[ArtifactRecord]:
        return await asyncio.to_thread(self._query_sync, q)

    async def audit_log(self) -> list[ArtifactRecord]:
        return await asyncio.to_thread(self._audit_log_sync)

    async def conflicts(self, *, unresolved_only: bool = True) -> list[ConflictRecord]:
        return await asyncio.to_thread(self._conflicts_sync, unresolved_only)

    def status(self) -> dict:
        status = self._state.status()
        return {
            "backend": "file",
            "durable": status["durable"],
            "path": status["path"],
            "keys": len(self._state.get("_index", [])),
            "note": status["note"],
        }
